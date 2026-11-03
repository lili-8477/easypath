"""A2: tissue probability map, candidate ROIs and zero-shot ROI annotation (model version zs-v0).

TITAN is a region/slide encoder, not a segmenter, so the tissue map is built by encoding each
patch's 5 x 5 neighbourhood of CONCH v1.5 features with TITAN and scoring it against tissue
prompts. This is a bootstrap: once pathologists have reviewed ROIs, a3_refine.py replaces the
zero-shot heads with linear probes on the same TITAN embeddings.

ROIs tile the tissue grid in 4 x 4 patch blocks (2048 px at 20x, about 1 mm2). Each ROI gets a
TITAN embedding, tumour / pattern / necrosis predictions, a relative confidence, an OOD flag and
a triage state for the review UI.

Outputs (results/annot/)
  tissue_map/<slide_id>.npz      gx, gy, step, probs [N, n_tissue_classes]
  tissue_map_scale.json          per-class 5th / 95th percentiles used for display
  roi_embeddings.npz             roi_id, emb (L2-normalised TITAN embeddings)
  rois_v000.csv                  ROI table with predictions and triage
  roi_thumbs/<roi_id>.jpg        256 px ROI thumbnails
"""
import json

import numpy as np
import openslide
import pandas as pd
import torch
from transformers import AutoModel

from ann_common import (ANN, MAP_DIR, MIN_ROI_PATCHES, PATCH_DIR, PATTERN, RES, ROI_PATCHES, TEMPLATES,
                        THUMB_DIR, TISSUE, TITAN_REV, WIN_RADIUS, load_patches, triage)

MODEL_VERSION = "zs-v0"
OOD_PCT = 5          # ROIs whose neighbour similarity is in the bottom 5 % are out-of-distribution


def rank01(x):
    return pd.Series(x).rank(pct=True).values


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


@torch.inference_mode()
def main():
    dev = "cuda"
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, revision=TITAN_REV).to(dev).eval()
    clf_t = titan.zero_shot_classifier(list(TISSUE.values()), TEMPLATES, device=dev)
    clf_p = titan.zero_shot_classifier(list(PATTERN.values()), TEMPLATES, device=dev)

    def encode(f, c, step):
        if len(f) == 1:  # TITAN fails on a 1 x 1 grid; an all-zero neighbour is masked out as background
            f = np.concatenate([f, np.zeros_like(f)])
            c = np.concatenate([c, c + [step, 0]])
        with torch.autocast("cuda", torch.float16):
            e = titan.encode_slide_from_patch_features(torch.from_numpy(f)[None].to(dev),
                                                       torch.from_numpy(c)[None].to(dev), step)
        e = e.float()
        return e, titan.zero_shot(e, clf_t).cpu().numpy().ravel(), titan.zero_shot(e, clf_p).cpu().numpy().ravel()

    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    qc = pd.read_csv(ANN / "qc_patches.csv").set_index(["slide_id", "x", "y"])
    tcls = list(TISSUE)
    rois, embs, allmap = [], [], []
    for h5 in sorted(PATCH_DIR.glob("*.h5")):
        sid = h5.stem
        feats, coords, step = load_patches(sid)
        gx, gy = coords[:, 0] // step, coords[:, 1] // step

        # ---- tissue map: TITAN on each patch's neighbourhood
        probs = np.zeros((len(coords), len(tcls)), np.float32)
        for i in range(len(coords)):
            nb = np.where((np.abs(gx - gx[i]) <= WIN_RADIUS) & (np.abs(gy - gy[i]) <= WIN_RADIUS))[0]
            probs[i] = encode(feats[nb], coords[nb], step)[1]
        np.savez(MAP_DIR / f"{sid}.npz", gx=gx, gy=gy, step=step, probs=probs, classes=np.array(tcls))
        allmap.append(probs)

        # ---- ROIs: 4 x 4 patch blocks with enough tissue
        bad = qc.loc[[(sid, int(x), int(y)) for x, y in coords], "qc_fail"].values
        bx, by = gx // ROI_PATCHES, gy // ROI_PATCHES
        s = openslide.OpenSlide(man.loc[sid, "file"])
        for (i, j) in sorted(set(zip(bx, by))):
            idx = np.where((bx == i) & (by == j))[0]
            if len(idx) < MIN_ROI_PATCHES:
                continue
            e, pt, pp = encode(feats[idx], coords[idx], step)
            x0, y0, side = int(i * ROI_PATCHES * step), int(j * ROI_PATCHES * step), int(ROI_PATCHES * step)
            rid = f"{sid}__{i}_{j}"
            pm = probs[idx]
            rois.append(dict(roi_id=rid, slide_id=sid, source="model", x=x0, y=y0, w=side, h=side,
                             n_patches=len(idx), frac_qc_fail=float(bad[idx].mean()),
                             **{f"p_tissue:{c}": float(v) for c, v in zip(tcls, pt)},
                             **{f"p_pattern:{c}": float(v) for c, v in zip(PATTERN, pp)},
                             necrosis_map_max=float(pm[:, tcls.index("necrosis")].max()),
                             necrosis_map_top=int((pm.argmax(1) == tcls.index("necrosis")).sum())))
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().ravel())
            th = s.read_region((x0, y0), 0, (side, side)).convert("RGB").resize((256, 256))
            th.save(THUMB_DIR / f"{rid}.jpg", quality=88)
        print(sid, len(coords), "patches", sum(r["slide_id"] == sid for r in rois), "rois", flush=True)

    allmap = np.concatenate(allmap)
    scale = {c: [float(np.percentile(allmap[:, k], 5)), float(np.percentile(allmap[:, k], 95))] for k, c in enumerate(tcls)}
    json.dump(scale, open(ANN / "tissue_map_scale.json", "w"), indent=1)

    R = pd.DataFrame(rois)
    E = np.stack(embs)
    np.savez(ANN / "roi_embeddings.npz", roi_id=R.roi_id.values, emb=E)

    # ---- predictions
    pt = R[[f"p_tissue:{c}" for c in tcls]].values
    pp = R[[f"p_pattern:{c}" for c in PATTERN]].values
    R["pred_tumor"] = np.where(pt.argmax(1) == tcls.index("tumor"), "tumor", "non_tumor")
    R["pred_pattern"] = np.array(list(PATTERN))[pp.argmax(1)]
    R["pred_necrosis"] = np.where(R.necrosis_map_top > 0, "present", "absent")
    # zero-shot probabilities are close to uniform, so confidence is the margin's rank in the cohort
    R["conf_tumor"] = rank01(margin(pt))
    R["conf_pattern"] = rank01(margin(pp))
    R["conf_necrosis"] = rank01(np.abs(R.necrosis_map_max - np.median(R.necrosis_map_max)))

    # OOD: mean cosine similarity to the 5 nearest ROIs from other slides
    S = E @ E.T
    S[R.slide_id.values[:, None] == R.slide_id.values[None, :]] = -1
    nn_sim = np.sort(S, 1)[:, -5:].mean(1)
    R["nn_sim"] = nn_sim
    R["ood"] = nn_sim < np.percentile(nn_sim, OOD_PCT)

    qcs = pd.read_csv(ANN / "qc_slides.csv").set_index("slide_id")
    blocked = (R.frac_qc_fail > 0.5) | (R.slide_id.map(qcs.qc_status) == "fail")
    conf = np.where(R.pred_tumor == "tumor", np.minimum(R.conf_tumor, R.conf_pattern), R.conf_tumor)
    R["confidence"] = conf
    R["uncertainty"] = 1 - conf
    R["state"] = triage(conf, R.ood.values, blocked.values, 1 / 3, 2 / 3)
    R["pred_artifact"] = blocked.values
    R["model_version"] = MODEL_VERSION
    R.to_csv(ANN / "rois_v000.csv", index=False)
    print(len(R), "ROIs from", R.slide_id.nunique(), "slides")
    print(R.state.value_counts().to_string())
    print(pd.crosstab(R.pred_tumor, R.pred_pattern).to_string())


if __name__ == "__main__":
    main()
