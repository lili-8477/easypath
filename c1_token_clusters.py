"""C1: unsupervised clustering of dense CONCH v1.5 tokens for cluster-level human review.

Tokens come from c0 (14 x 14 per 512 px / 20x patch). They are pooled to TOKEN_GRID per patch
(default 7 -> about 37 um, a few cells), background tokens (tissue < 50 %) are dropped, then
L2 + PCA64 + mini-batch k-means (cosine geometry). K deliberately over-clusters: reviewers merge
clusters by giving them the same label, which is cheaper than splitting a mixed cluster.

For review each cluster gets exemplars (nearest the centroid, at most 3 per slide so batch-driven
clusters show up, plus a random spread) drawn as the token box inside its surrounding tissue, its
cohort / slide make-up, and the patch-level CONCH zero-shot call as a hint.

Env: TOKEN_GRID (14, 7 or 2; default 7), N_CLUSTERS (default 20)

Outputs (results/annot/clusters/)
  clustering.json         id used to key reviews, settings
  tokens.parquet          slide_id, px, py, ti, tj, x0, y0, size, cluster, dist, conch_top
  summary.csv, k_scan.csv, exemplars.csv + ex/<name>.jpg
  ../tissue_map/<slide>_clusters.npz   tx, ty (token grid), grid, tsize, cluster (viewer overlay)
  figures/token_clusters_umap.{svg,pdf,png}
"""
import json
import os

import h5py
import numpy as np
import openslide
import pandas as pd
import umap
from PIL import ImageDraw
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from ann_common import ANN, CLUSTER_COLORS, MAP_DIR, RES, SEED, TISSUE
from common import COHORT_COLORS, figure_setup, save

G = int(os.environ.get("TOKEN_GRID", 7))
K = int(os.environ.get("N_CLUSTERS", 20))
MIN_TISSUE = 0.5
N_CENTRAL, N_RANDOM, PER_SLIDE = 16, 8, 3
CONTEXT = 4            # exemplar crop = 4 x token size, token box drawn in the middle
TOK = ANN / "dense_tokens"
OUT = ANN / "clusters"
(OUT / "ex").mkdir(parents=True, exist_ok=True)


def pool(a, f):
    """Average f x f blocks over the two grid axes of [N, g, g, ...]."""
    n, g = a.shape[:2]
    return a.reshape(n, g // f, f, g // f, f, *a.shape[3:]).mean((2, 4))


def main():
    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    pkey = man.Unified_ID.fillna(man.patient)
    tcls = list(TISSUE)
    feats, recs = [], []
    for h5 in sorted(TOK.glob("*.h5")):
        sid = h5.stem
        with h5py.File(h5) as f:
            t, tis, coords = f["tokens"][:].astype(np.float32), f["tissue"][:].astype(np.float32), f["coords"][:]
            step, g0 = int(f.attrs["patch_size_level0"]), int(f.attrs["grid"])
        fct = g0 // G
        t, tis = pool(t, fct), pool(tis, fct)
        conch_top = np.array(tcls)[np.load(MAP_DIR / f"{sid}_conch.npz")["probs"].argmax(1)]
        p, ti, tj = np.nonzero(tis >= MIN_TISSUE)
        feats.append(t[p, ti, tj])
        size = step / G
        recs.append(pd.DataFrame(dict(slide_id=sid, px=coords[p, 0], py=coords[p, 1], ti=ti, tj=tj,
                                      x0=(coords[p, 0] + tj * size).round().astype(int),
                                      y0=(coords[p, 1] + ti * size).round().astype(int),
                                      size=size, step=step, tissue=tis[p, ti, tj], conch_top=conch_top[p])))
    T = pd.concat(recs, ignore_index=True)
    X = normalize(np.concatenate(feats))
    del feats
    um = float(T["size"].iloc[0] * man.mpp.iloc[0])
    print(f"{len(T)} tissue tokens ({G} x {G} per patch, {um:.0f} um) from {T.slide_id.nunique()} slides", flush=True)

    rng = np.random.default_rng(SEED)
    fit = rng.choice(len(X), min(60000, len(X)), replace=False)
    pca = PCA(64, random_state=SEED).fit(X[fit])
    Z = normalize(np.vstack([pca.transform(X[i:i + 50000]) for i in range(0, len(X), 50000)]).astype(np.float32))
    del X

    sub = fit[:10000]
    scan = []
    for k in range(8, 41, 4):
        km = MiniBatchKMeans(k, batch_size=8192, n_init=3, random_state=SEED).fit(Z[fit])
        scan.append(dict(k=k, silhouette=silhouette_score(Z[sub], km.predict(Z[sub]), metric="cosine")))
    pd.DataFrame(scan).to_csv(OUT / "k_scan.csv", index=False)
    print(pd.DataFrame(scan).round(3).to_string(index=False), flush=True)

    km = MiniBatchKMeans(K, batch_size=8192, n_init=10, max_iter=300, random_state=SEED).fit(Z)
    cent = normalize(km.cluster_centers_)
    order = np.argsort(PCA(1, random_state=SEED).fit_transform(cent).ravel())   # similar clusters adjacent
    remap = np.empty(K, int)
    remap[order] = np.arange(K)
    T["cluster"] = remap[km.labels_]
    cent = cent[order]
    T["dist"] = 1 - (Z * cent[T.cluster.values]).sum(1)
    T["cohort"] = T.slide_id.map(man.cohort)
    T["patient"] = T.slide_id.map(pkey)
    T.to_parquet(OUT / "tokens.parquet", index=False)

    for sid, g in T.groupby("slide_id"):
        step = int(g.step.iloc[0])
        np.savez(MAP_DIR / f"{sid}_clusters.npz", tx=((g.px // step) * G + g.tj).values, ty=((g.py // step) * G + g.ti).values,
                 grid=G, tsize=step / G, cluster=g.cluster.values.astype(np.int16))

    # ---- summary
    base = T.cohort.value_counts(normalize=True)
    summ = []
    for k, g in T.groupby("cluster"):
        cf = g.cohort.value_counts(normalize=True)
        ch = g.conch_top.value_counts(normalize=True)
        summ.append(dict(cluster=k, n_tokens=len(g), frac_tokens=len(g) / len(T), n_slides=g.slide_id.nunique(),
                         n_patients=g.patient.nunique(), **{f"frac_{c}": float(cf.get(c, 0)) for c in base.index},
                         cohort_enrichment=float((cf / base).max()),
                         max_slide_share=float(g.slide_id.value_counts(normalize=True).iloc[0]),
                         top_slide=g.slide_id.value_counts().index[0],
                         conch_hint=ch.index[0], conch_hint_frac=float(ch.iloc[0]),
                         conch_hint_2=ch.index[1] if len(ch) > 1 else "", mean_dist=float(g.dist.mean()),
                         color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)]))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "summary.csv", index=False)
    json.dump(dict(id=f"conch15_dense{G}_k{K}_seed{SEED}", k=K, grid=G, token_um=round(um, 1), n_tokens=len(T),
                   n_slides=int(T.slide_id.nunique()), min_tissue=MIN_TISSUE,
                   tokens="CONCH v1.5 final-layer ViT tokens (raw tiles), pooled per patch, L2 + PCA64 + mini-batch k-means"),
              open(OUT / "clustering.json", "w"), indent=1)
    print(S[["cluster", "n_tokens", "n_slides", "n_patients", "max_slide_share", "cohort_enrichment", "conch_hint", "conch_hint_frac"]]
          .round(2).to_string(index=False), flush=True)

    # ---- exemplars: token box inside its surrounding tissue
    ex = []
    for k, g in T.groupby("cluster"):
        central = g.sort_values("dist").groupby("slide_id").head(PER_SLIDE).head(N_CENTRAL)
        rest = g.drop(central.index)
        spread = rest.sample(min(N_RANDOM, len(rest)), random_state=SEED)
        for kind, part in [("central", central), ("random", spread)]:
            for r in part.itertuples():
                ex.append(dict(cluster=k, kind=kind, slide_id=r.slide_id, x0=r.x0, y0=r.y0, size=r.size, dist=r.dist,
                               name=f"{k}_{r.slide_id}_{r.x0}_{r.y0}"))
    E = pd.DataFrame(ex)
    for sid, g in E.groupby("slide_id"):
        s = openslide.OpenSlide(man.loc[sid, "file"])
        for r in g.itertuples():
            p = OUT / "ex" / f"{r.name}.jpg"
            if p.exists():
                continue
            ctx = int(round(r.size * CONTEXT))
            cx, cy = int(r.x0 + r.size / 2 - ctx / 2), int(r.y0 + r.size / 2 - ctx / 2)
            im = s.read_region((cx, cy), 0, (ctx, ctx)).convert("RGB").resize((256, 256))
            a, b = 128 - 128 / CONTEXT, 128 + 128 / CONTEXT
            d = ImageDraw.Draw(im)
            d.rectangle([a - 1, a - 1, b + 1, b + 1], outline=(0, 0, 0), width=1)
            d.rectangle([a, a, b, b], outline=(255, 255, 255), width=2)
            im.save(p, quality=88)
    E.to_csv(OUT / "exemplars.csv", index=False)

    # ---- figure: token UMAP (subsample) by cluster and by cohort
    usub = rng.choice(len(Z), min(40000, len(Z)), replace=False)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=SEED).fit_transform(Z[usub])
    cl, co = T.cluster.values[usub], T.cohort.values[usub]
    figure_setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))
    for k in range(K):
        m = cl == k
        axes[0].scatter(emb[m, 0], emb[m, 1], s=0.6, color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], linewidths=0, rasterized=True)
        cx, cy = np.median(emb[m], 0)
        axes[0].text(cx, cy, f"C{k}", fontsize=6, ha="center", va="center", weight="bold")
    for c, col in COHORT_COLORS.items():
        m = co == c
        axes[1].scatter(emb[m, 0], emb[m, 1], s=0.6, color=col, linewidths=0, label=c, rasterized=True)
    axes[1].legend(frameon=False, markerscale=6, fontsize=6)
    for ax, t in zip(axes, [f"{K} token clusters", "Cohort"]):
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    fig.suptitle(f"CONCH v1.5 dense tokens ({um:.0f} µm), {len(T):,} tokens from {T.slide_id.nunique()} H&E slides "
                 f"(UMAP of {len(usub):,})", fontsize=7)
    fig.tight_layout()
    save(fig, "token_clusters_umap")


if __name__ == "__main__":
    main()
