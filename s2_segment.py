"""S2: self-supervised smooth segmentation from the S1 dense feature canvases.

1. k-means (cosine) on tissue tokens pooled across all slides -> K_MAX segment classes shared by the
   cohort, no labels used.
2. Soft assignment per token, softmax(cos / TAU), smoothed with a tissue-normalised Gaussian.
2b. Per slide K (ADAPT=1): classes covering < MIN_FRAC of the slide's tissue are folded into their nearest
   present class, then the two least separable classes on this slide are merged while their separation
   d' (token features projected on the line between the two class means) is below D_MIN. A merged
   segment keeps the id of its largest member, so class labels still apply, and lists its members.
3. Upsampled to OUT_UM (4 um/px) and refined with a guided filter that uses the H&E image itself as the
   guide, so class boundaries follow tissue edges instead of the token grid.
4. Argmax inside a 4 um tissue mask, then specks below MIN_UM2 are absorbed by their nearest neighbour.

The pathologist names each segment class afterwards (review UI, tab 1). Synovial sarcoma components
(spindle, glandular, corded / solid epithelial, poorly differentiated round cell) are segment-level
labels; monophasic / biphasic / poorly differentiated are derived per slide from their areas.

Env: ENCODER (conch15 | hoptimus0 | hoptimus1 | uni2h | uni | virchow2, features from s1), K_MAX (default 10, N_SEG is an alias),
     ADAPT (default 1), D_MIN (default 2.0), MIN_FRAC (default 0.01), TAU (default 0.05),
     SEG_OUT (default results/annot/seg_<encoder>_kmax<K>)

Outputs (SEG_OUT)
  clustering.json, summary.csv, exemplars.csv + ex/, centroids.npy
  labels/<slide_id>.png + labels/<slide_id>.json   label map (255 = background), its level-0 placement and the
                                                    per-slide segments (groups of classes)
  slide_k.csv                                       segments per slide, merges and the separations behind them
  geojson/<slide_id>.geojson                        polygons per segment class (level-0 px)
  area_mm2.csv                                      slide x segment class area
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from ann_common import ANN, CLUSTER_COLORS, MAP_DIR, PATTERN, RES, SEED, TISSUE

ENCODER = os.environ.get("ENCODER", "conch15")
K = int(os.environ.get("K_MAX", os.environ.get("N_SEG", 10)))
TAU = float(os.environ.get("TAU", 0.05))
ADAPT = os.environ.get("ADAPT", "1") == "1"
D_MIN = float(os.environ.get("D_MIN", 2.0))      # two classes stay apart on a slide only if d' >= D_MIN
MIN_FRAC = float(os.environ.get("MIN_FRAC", 0.01))  # classes below this share of a slide's tissue are folded in
OUT_UM = 4.0
MIN_UM2 = 4000.0          # specks smaller than this (about 63 x 63 um) are absorbed
INST_UM2 = 10000.0        # connected regions at least this large are used as review exemplars
GF_R, GF_EPS = 3, 1e-2    # guided filter radius (px at 4 um) and regulariser
CROP_UM = 256             # exemplar field of view
FEAT = ANN / "seg" / "feat" if ENCODER == "conch15" else ANN / f"feat_{ENCODER}"
SEG = Path(os.environ.get("SEG_OUT") or ANN / f"seg_{ENCODER}_kmax{K}{'' if ADAPT else '_fixed'}")
for d in ("labels", "geojson", "ex"):
    (SEG / d).mkdir(parents=True, exist_ok=True)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def guided_filter(I, p, r, eps):
    """He et al. guided filter, grey guide I [h, w], inputs p [h, w, c]."""
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1))  # noqa: E731
    mI, mII = box(I), box(I * I)
    varI = mII - mI * mI
    out = np.empty_like(p)
    for c in range(p.shape[2]):
        mp, mIp = box(p[..., c]), box(I * p[..., c])
        a = (mIp - mI * mp) / (varI + eps)
        b = mp - a * mI
        out[..., c] = box(a) * I + box(b)
    return out


def read_guide(slide, x0, y0, w0, h0, out_w, out_h):
    """H&E of the canvas extent at the output resolution (reads the closest pyramid level)."""
    ds = w0 / out_w
    lvl = slide.get_best_level_for_downsample(ds)
    f = slide.level_downsamples[lvl]
    im = slide.read_region((int(x0), int(y0)), lvl, (int(np.ceil(w0 / f)), int(np.ceil(h0 / f)))).convert("RGB")
    return np.asarray(im.resize((out_w, out_h), Image.BILINEAR))


def tissue_at(rgb):
    od = -np.log(np.clip(rgb.astype(np.float32), 1, 255) / 255.0).mean(-1)
    m = od > 0.12
    m = ndi.binary_opening(ndi.binary_closing(m, iterations=2), iterations=1)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.nonzero(sizes >= 200)[0])
    return ndi.binary_fill_holes(m) & (od > 0.05)


def absorb_specks(lab, tissue, min_px):
    """Relabel connected components smaller than min_px with the nearest surviving label."""
    small = np.zeros_like(tissue)
    for k in np.unique(lab[tissue]):
        cc, n = ndi.label(lab == k)
        if n:
            sizes = ndi.sum(np.ones_like(cc), cc, range(1, n + 1))
            small |= np.isin(cc, 1 + np.nonzero(sizes < min_px)[0])
    keep = tissue & ~small
    if small.any() and keep.any():
        _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
        lab = np.where(small, lab[iy, ix], lab)
    return lab


def dprime(xa, xb):
    """Separation of two token sets along the line between their means (pooled-sd units)."""
    w = xa.mean(0) - xb.mean(0)
    pa, pb = xa @ w, xb @ w
    return float((pa.mean() - pb.mean()) / np.sqrt((pa.var() + pb.var()) / 2 + 1e-12))


def adapt_groups(X, a, cent, rng):
    """Per-slide segments as groups of cohort classes. X unit tissue tokens, a their class.
    Returns [(representative, [members])], largest first, and the merge log."""
    if len(X) > 30000:
        i = rng.choice(len(X), 30000, replace=False)
        X, a = X[i], a[i]
    frac = np.bincount(a, minlength=K) / len(a)
    present = [k for k in range(K) if frac[k] >= MIN_FRAC] or [int(frac.argmax())]
    groups = {k: [k] for k in present}
    for k in range(K):                       # small classes join the most similar present class
        if k not in groups and frac[k] > 0:
            groups[present[int(np.argmax(cent[present] @ cent[k]))]].append(k)
    gid = np.full(K, -1)
    log = []
    while len(groups) > 1:
        for r, mem in groups.items():
            gid[mem] = r
        g = gid[a]
        reps = list(groups)
        best = None
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                xa, xb = X[g == reps[i]], X[g == reps[j]]
                if len(xa) < 20 or len(xb) < 20:
                    d = 0.0
                else:
                    d = dprime(xa, xb)
                if best is None or d < best[0]:
                    best = (d, reps[i], reps[j])
        if best[0] >= D_MIN:
            break
        d, r1, r2 = best
        keep, drop = (r1, r2) if frac[groups[r1]].sum() >= frac[groups[r2]].sum() else (r2, r1)
        groups[keep] += groups.pop(drop)
        log.append(f"{keep}+{drop}@{d:.2f}")
    out = []
    for mem in groups.values():
        r = max(mem, key=lambda k: frac[k])
        out.append((int(r), sorted(int(k) for k in mem)))
    out.sort(key=lambda t: -frac[t[1]].sum())
    return out, log


def main():
    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    pkey = man.Unified_ID.fillna(man.patient)
    files = sorted(FEAT.glob("*.npz"))
    rng = np.random.default_rng(SEED)

    # ---- 1. k-means on pooled tissue tokens
    samp = []
    for f in files:
        z = np.load(f)
        X = z["feat"].astype(np.float32).reshape(-1, z["feat"].shape[-1])[z["tissue"].reshape(-1) >= 0.5]
        samp.append(unit(X[rng.choice(len(X), min(6000, len(X)), replace=False)]))
    samp = np.concatenate(samp)
    km = MiniBatchKMeans(K, batch_size=8192, n_init=10, max_iter=300, random_state=SEED).fit(samp)
    cent = unit(km.cluster_centers_)
    cent = cent[np.argsort(PCA(1, random_state=SEED).fit_transform(cent).ravel())]   # similar classes adjacent
    np.save(SEG / "centroids.npy", cent)
    print(f"k-means on {len(samp)} tokens, K={K}", flush=True)

    tcls, pcls = list(TISSUE), list(PATTERN)
    areas, insts, hint_t, hint_p, slide_k = [], [], np.zeros((K, len(tcls))), np.zeros((K, len(pcls))), []
    for f in files:
        sid = f.stem
        z = np.load(f)
        feat, tis = unit(z["feat"].astype(np.float32)), z["tissue"].astype(np.float32)
        x0, y0, ts = int(z["x0"]), int(z["y0"]), float(z["ts"])
        mpp = float(man.loc[sid, "mpp"])
        h, w = tis.shape

        # ---- 2. soft assignment + tissue-normalised smoothing on the token grid
        logits = feat @ cent.T / TAU
        P = np.exp(logits - logits.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        tw = (tis > 0.2).astype(np.float32)
        num = np.stack([ndi.gaussian_filter(P[..., k] * tw, 1.0) for k in range(K)], -1)
        P = num / (ndi.gaussian_filter(tw, 1.0)[..., None] + 1e-6)

        # ---- 2b. segments on this slide: groups of cohort classes, probabilities summed within a group
        T = tis >= 0.5
        if ADAPT and T.sum() >= 50:
            groups, log = adapt_groups(feat[T], P[T].argmax(-1), cent, rng)
        else:
            groups, log = [(k, [k]) for k in range(K)], []
        P = np.stack([P[..., mem].sum(-1) for _, mem in groups], -1)
        rep_of = np.array([r for r, _ in groups], np.uint8)

        # ---- 3. upsample to 4 um and refine with the H&E as guide
        px_l0 = OUT_UM / mpp
        ow, oh = int(np.ceil(w * ts / px_l0)), int(np.ceil(h * ts / px_l0))
        Pu = cv2.resize(P.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
        if Pu.ndim == 2:
            Pu = Pu[..., None]
        s = openslide.OpenSlide(man.loc[sid, "file"])
        rgb = read_guide(s, x0, y0, w * ts, h * ts, ow, oh)
        guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        Pg = guided_filter(guide, Pu, GF_R, GF_EPS)

        # ---- 4. labels inside tissue, specks absorbed
        tissue = tissue_at(rgb) & (cv2.resize(tw, (ow, oh), interpolation=cv2.INTER_LINEAR) > 0.1)
        lab = rep_of[Pg.argmax(-1)]
        lab = absorb_specks(lab, tissue, int(MIN_UM2 / OUT_UM ** 2))
        lab[~tissue] = 255
        seen = set(np.unique(lab[tissue]).tolist())
        groups = [(r, mem) for r, mem in groups if r in seen]
        slide_k.append(dict(slide_id=sid, k=len(groups), merges=" ".join(log),
                            segments=" ".join("+".join(map(str, mem)) for _, mem in groups)))
        Image.fromarray(lab, "L").save(SEG / "labels" / f"{sid}.png")
        json.dump(dict(x0=x0, y0=y0, px_l0=px_l0, w=ow, h=oh, out_um=OUT_UM, k=len(groups),
                       groups=[dict(segment=r, members=mem) for r, mem in groups]), open(SEG / "labels" / f"{sid}.json", "w"))

        # ---- polygons (level 0)
        feats_geo = []
        for k in range(K):
            m = (lab == k).astype(np.uint8)
            if not m.any():
                continue
            cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            polys = {}
            for i, c in enumerate(cs):
                c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :]
                if len(c) < 3:
                    continue
                ring = [[float(x0 + (u + 0.5) * px_l0), float(y0 + (v + 0.5) * px_l0)] for u, v in c]
                ring.append(ring[0])
                parent = hier[0][i][3]
                if parent < 0:
                    polys.setdefault(i, [ring])
                else:
                    polys.setdefault(parent, []).append(ring)
            for rings in polys.values():
                feats_geo.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings},
                                  "properties": {"objectType": "annotation", "segment": int(k), "slide_id": sid}})
        json.dump({"type": "FeatureCollection", "features": feats_geo}, open(SEG / "geojson" / f"{sid}.geojson", "w"))

        # ---- area, CONCH hint (patch-level calls under each segment), instances for exemplars
        a = np.bincount(lab[tissue].ravel(), minlength=K)[:K] * (OUT_UM ** 2) / 1e6
        areas.append(pd.Series(a, index=range(K), name=sid))
        cz = np.load(MAP_DIR / f"{sid}_conch.npz")
        pstep = int(cz["step"])
        gx = ((x0 + (np.arange(ow) + 0.5) * px_l0) // pstep).astype(int)
        gy = ((y0 + (np.arange(oh) + 0.5) * px_l0) // pstep).astype(int)
        cell = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(cz["gx"], cz["gy"]))}
        tt, pt = cz["probs"].argmax(1), cz["pattern_probs"].argmax(1)
        yy, xx = np.nonzero(tissue)
        sel = rng.choice(len(yy), min(20000, len(yy)), replace=False)
        for v, u in zip(yy[sel], xx[sel]):
            i = cell.get((gx[u], gy[v]))
            if i is not None:
                hint_t[lab[v, u], tt[i]] += 1
                hint_p[lab[v, u], pt[i]] += 1
        for k in range(K):
            cc, n = ndi.label(lab == k)
            if not n:
                continue
            sizes = ndi.sum(np.ones_like(cc), cc, range(1, n + 1)) * OUT_UM ** 2
            coms = ndi.center_of_mass(np.ones_like(cc), cc, range(1, n + 1))
            for size, (cy, cx) in zip(sizes, coms):
                if size >= INST_UM2:
                    insts.append(dict(cluster=k, slide_id=sid, area_um2=float(size),
                                      cx=float(x0 + (cx + 0.5) * px_l0), cy=float(y0 + (cy + 0.5) * px_l0)))
        print(sid, f"{ow}x{oh}px", "segments", len(groups), " ".join(log), flush=True)

    A = pd.DataFrame(areas).fillna(0)
    A.index.name = "slide_id"
    A.to_csv(SEG / "area_mm2.csv")
    SK = pd.DataFrame(slide_k)
    SK.to_csv(SEG / "slide_k.csv", index=False)
    print("segments per slide:", SK.k.value_counts().sort_index().to_dict(), "median", SK.k.median(), flush=True)
    I = pd.DataFrame(insts)

    # ---- summary
    cohort = A.index.map(man.cohort)
    tot = A.sum(0)
    base = pd.Series(A.sum(1).values, index=cohort).groupby(level=0).sum() / A.values.sum()
    summ = []
    for k in range(K):
        col = A[k]
        cf = pd.Series(col.values, index=cohort).groupby(level=0).sum() / max(col.sum(), 1e-9)
        ht, hp = hint_t[k] / max(hint_t[k].sum(), 1), hint_p[k] / max(hint_p[k].sum(), 1)
        o = np.argsort(-ht)
        summ.append(dict(cluster=k, area_mm2=float(tot[k]), frac_tokens=float(tot[k] / tot.sum()),
                         n_tokens=int(round(tot[k] * 1e6 / OUT_UM ** 2)), n_slides=int((col > 0.01).sum()),
                         n_patients=int(pd.Series(pkey.loc[A.index[col > 0.01]]).nunique()),
                         **{f"frac_{c}": float(cf.get(c, 0)) for c in base.index},
                         cohort_enrichment=float((cf / base).max()), max_slide_share=float(col.max() / max(col.sum(), 1e-9)),
                         top_slide=col.idxmax(), conch_hint=tcls[o[0]], conch_hint_frac=float(ht[o[0]]),
                         conch_hint_2=tcls[o[1]], conch_pattern_hint=pcls[int(hp.argmax())],
                         conch_pattern_frac=float(hp.max()), color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)]))
    S = pd.DataFrame(summ)
    S.to_csv(SEG / "summary.csv", index=False)
    enc = {"conch15": "CONCH v1.5", "hoptimus0": "H-optimus-0", "hoptimus1": "H-optimus-1", "uni2h": "UNI2-h", "uni": "UNI", "virchow2": "Virchow2"}[ENCODER]
    rid = f"seg_{ENCODER}_k{K}" + (f"_adapt_d{D_MIN}_f{MIN_FRAC}" if ADAPT else "") + f"_tau{TAU}_seed{SEED}"
    json.dump(dict(id=rid, kind="seg", k=K, encoder=ENCODER, adaptive=ADAPT, d_min=D_MIN, min_frac=MIN_FRAC,
                   k_slide_median=float(SK.k.median()), k_slide_max=int(SK.k.max()), token_um=round(float(z["ts"]) * mpp, 1),
                   out_um=OUT_UM, n_tokens=int(S.n_tokens.sum()), n_slides=len(A), area_mm2=round(float(tot.sum()), 1),
                   method=f"{enc} dense tokens (50 % overlapping windows, Hann blend) -> k-means (K={K}) -> soft assignment "
                          + (f"-> per slide, classes < {MIN_FRAC:.0%} folded in and classes with d' < {D_MIN} merged " if ADAPT else "")
                          + "-> Gaussian + H&E-guided filter -> 4 um labels, specks < 4000 um2 absorbed"),
              open(SEG / "clustering.json", "w"), indent=1)
    print(S[["cluster", "area_mm2", "n_slides", "n_patients", "max_slide_share", "cohort_enrichment",
             "conch_hint", "conch_hint_frac", "conch_pattern_hint"]].round(2).to_string(index=False), flush=True)

    # ---- exemplars: segment outline inside a 256 um field
    ex = []
    for k, g in I.groupby("cluster"):
        typ = g.sort_values("area_um2", ascending=False).groupby("slide_id").head(3).head(16)
        rest = g.drop(typ.index)
        rnd = rest.sample(min(32, len(rest)), random_state=SEED)
        for kind, part in [("central", typ), ("random", rnd.head(8)), ("more", rnd.iloc[8:])]:
            for r in part.itertuples():
                ex.append(dict(cluster=k, kind=kind, slide_id=r.slide_id, cx=r.cx, cy=r.cy, area_um2=r.area_um2,
                               name=f"s{k}_{r.slide_id}_{int(r.cx)}_{int(r.cy)}"))
    E = pd.DataFrame(ex)
    for sid, g in E.groupby("slide_id"):
        s = openslide.OpenSlide(man.loc[sid, "file"])
        mpp = float(man.loc[sid, "mpp"])
        lab = np.asarray(Image.open(SEG / "labels" / f"{sid}.png"))
        meta = json.load(open(SEG / "labels" / f"{sid}.json"))
        side = int(round(CROP_UM / mpp))
        for r in g.itertuples():
            p = SEG / "ex" / f"{r.name}.jpg"
            if p.exists():
                continue
            X0, Y0 = int(r.cx - side / 2), int(r.cy - side / 2)
            im = np.asarray(s.read_region((X0, Y0), 0, (side, side)).convert("RGB").resize((320, 320))).copy()
            u0, v0 = (X0 - meta["x0"]) / meta["px_l0"], (Y0 - meta["y0"]) / meta["px_l0"]
            n = side / meta["px_l0"]
            M = np.float32([[n / 320, 0, u0], [0, n / 320, v0]])
            m = cv2.warpAffine((lab == r.cluster).astype(np.uint8), M, (320, 320),
                               flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (0, 0, 0), 3)
            cv2.drawContours(im, cs, -1, (255, 255, 255), 1)
            Image.fromarray(im).save(p, quality=88)
    side = (CROP_UM / E.slide_id.map(man.mpp)).round()          # exemplar field in level-0 px
    E.assign(x0=(E.cx - side / 2).astype(int), y0=(E.cy - side / 2).astype(int), size=side).to_csv(SEG / "exemplars.csv", index=False)


if __name__ == "__main__":
    main()
