"""A1: patch-level image QC on the TITAN patch grid (512 px at 20x).

Flags per patch: blur (Laplacian variance), pen ink (green/blue/black pixels), fold (very dark
over-stained tissue), weak stain (low optical density), little tissue. Slide QC summarises them.

Outputs (results/annot/)
  qc_patches.csv   slide_id, x, y, metrics, flag columns, qc_fail
  qc_slides.csv    per-slide counts, flagged fraction and status ok / warn / fail
"""
from multiprocessing import Pool

import cv2
import numpy as np
import openslide
import pandas as pd

from ann_common import ANN, PATCH_DIR, RES, load_patches

BLUR_MIN = 100.0     # Laplacian variance on 512 px grey patch; below = out of focus (cohort 5th pct is about 340)
PEN_MAX = 0.05       # fraction of pen-coloured pixels
FOLD_MAX = 0.25      # fraction of tissue pixels with mean OD > 1.2 (folded / overlapping)
WEAK_OD = 0.20       # mean OD of tissue pixels below this = weak stain
TISSUE_MIN = 0.25


def patch_metrics(rgb):
    rgbf = rgb.astype(np.float32)
    od = -np.log(np.clip(rgbf, 1, 255) / 255.0)
    tissue = od.mean(-1) > 0.12
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / [180, 255, 255]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pen = ((h > 0.15) & (h < 0.62) & (s > 0.3) & (v > 0.15)) | ((v < 0.15) & (s < 0.35))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tf = float(tissue.mean())
    return dict(tissue_frac=tf,
                blur=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                pen_frac=float(pen.mean()),
                fold_frac=float((od.mean(-1)[tissue] > 1.2).mean()) if tf > 0 else 0.0,
                tissue_od=float(od.mean(-1)[tissue].mean()) if tf > 0 else 0.0)


def run(slide):
    sid, path = slide
    _, coords, step = load_patches(sid)
    s = openslide.OpenSlide(path)
    rows = []
    for x, y in coords:
        rgb = np.asarray(s.read_region((int(x), int(y)), 0, (step, step)).convert("RGB").resize((512, 512)))
        rows.append(dict(slide_id=sid, x=int(x), y=int(y), **patch_metrics(rgb)))
    return rows


def main():
    man = pd.read_csv(RES / "slide_manifest.csv")
    man = man[man.slide_id.map(lambda s: (PATCH_DIR / f"{s}.h5").exists())]
    with Pool(12) as p:
        q = pd.DataFrame([r for rows in p.map(run, list(zip(man.slide_id, man.file))) for r in rows])
    q["flag_blur"] = q.blur < BLUR_MIN
    q["flag_pen"] = q.pen_frac > PEN_MAX
    q["flag_fold"] = q.fold_frac > FOLD_MAX
    q["flag_weak"] = q.tissue_od < WEAK_OD
    q["flag_low_tissue"] = q.tissue_frac < TISSUE_MIN
    flags = [c for c in q.columns if c.startswith("flag_")]
    q["qc_fail"] = q[flags].any(axis=1)
    q.to_csv(ANN / "qc_patches.csv", index=False)

    g = q.groupby("slide_id")
    sq = g[flags + ["qc_fail"]].mean().add_prefix("frac_")
    sq["n_patches"] = g.size()
    sq["n_usable"] = g.qc_fail.apply(lambda f: int((~f).sum()))
    sq["qc_status"] = np.select([(sq.n_usable < 8) | (sq.frac_qc_fail > 0.5), sq.frac_qc_fail > 0.2], ["fail", "warn"], "ok")
    sq.to_csv(ANN / "qc_slides.csv")
    print(q[["blur", "pen_frac", "fold_frac", "tissue_od", "tissue_frac"]].describe(percentiles=[.05, .5, .95]).round(3).to_string())
    print(q[flags].mean().round(3).to_string())
    print(sq.qc_status.value_counts().to_string())


if __name__ == "__main__":
    main()
