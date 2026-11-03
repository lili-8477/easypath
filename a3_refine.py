"""A3: periodic classifier refinement from pathologist decisions.

Reads the latest review of every ROI, embeds pathologist-drawn ROIs with TITAN, fits a logistic
regression (linear probe on frozen TITAN embeddings) per task that has enough labels, reports
patient-grouped cross-validation, and writes the next ROI table rois_v<NNN>.csv that the review
UI picks up on reload. Tasks without enough labels keep their previous predictions.

Outputs (results/annot/)
  rois_v<NNN>.csv              predictions, probabilities, confidence, triage (model lp-v<NNN>)
  refine_report_v<NNN>.csv     per task: labels used, class counts, CV balanced accuracy / AUC
  manual_embeddings.npz        cached TITAN embeddings of pathologist-drawn ROIs
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from ann_common import ANN, RES, latest_rois, load_patches, triage
from ann_db import connect, latest_reviews, manual_rois

MIN_PER_CLASS = 5
LO, HI = 0.6, 0.9    # probability thresholds for needs_review / review / prelabel


def embed_manual(man_rois):
    """TITAN embeddings for pathologist-drawn ROIs, from the patches whose top-left lies inside."""
    cache = ANN / "manual_embeddings.npz"
    have = dict(zip(*np.load(cache, allow_pickle=True).values())) if cache.exists() else {}
    todo = man_rois[~man_rois.roi_id.isin(have)]
    if len(todo):
        import torch
        from transformers import AutoModel
        from ann_common import TITAN_REV
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True, revision=TITAN_REV).to(dev).eval()
        for r in todo.itertuples():
            f, c, step = load_patches(r.slide_id)
            m = (c[:, 0] >= r.x) & (c[:, 0] < r.x + r.w) & (c[:, 1] >= r.y) & (c[:, 1] < r.y + r.h)
            if not m.any():
                continue
            with torch.inference_mode(), torch.autocast(dev, torch.float16, enabled=dev == "cuda"):
                e = titan.encode_slide_from_patch_features(torch.from_numpy(f[m])[None].to(dev),
                                                           torch.from_numpy(c[m])[None].to(dev), step)
            have[r.roi_id] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().ravel()
        np.savez(cache, roi_id=np.array(list(have)), emb=np.stack(list(have.values())))
    return have


def main():
    prev_path = latest_rois()
    prev = pd.read_csv(prev_path)
    version = int(prev_path.stem.split("_v")[1]) + 1
    tag = f"lp-v{version:03d}"
    z = np.load(ANN / "roi_embeddings.npz", allow_pickle=True)
    emb = dict(zip(z["roi_id"], z["emb"]))

    con = connect()
    mr = manual_rois(con)
    emb.update(embed_manual(mr))
    rev = latest_reviews(con)
    con.close()
    print(len(rev), "reviewed ROIs,", len(mr), "drawn by pathologists")

    # all ROIs: model ROIs from the previous table plus manual ones
    R = pd.concat([prev[prev.source == "model"],
                   mr.assign(source="manual", n_patches=np.nan)[["roi_id", "slide_id", "source", "x", "y", "w", "h"]]],
                  ignore_index=True)
    R = R[R.roi_id.isin(emb)].reset_index(drop=True)
    X = np.stack([emb[r] for r in R.roi_id])
    man = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
    pkey = man.Unified_ID.fillna(man.patient)

    lab = rev.set_index("roi_id")
    tasks = {
        "tumor": lab.tumor[lab.artifact != 1],
        "pattern": lab.pattern[(lab.tumor == "tumor") & (lab.artifact != 1)],
        "necrosis": lab.necrosis[lab.artifact != 1],
        "artifact": lab.artifact.map({1: "yes", 0: "no"}),
    }
    report = []
    for task, y in tasks.items():
        y = y.dropna()
        y = y[y.index.isin(R.roi_id)]
        counts = y.value_counts()
        ok = counts[counts >= MIN_PER_CLASS]
        y = y[y.isin(ok.index)]
        groups = R.set_index("roi_id").loc[y.index, "slide_id"].map(pkey).values
        row = dict(task=task, n=len(y), classes=counts.to_dict(), trained=False)
        if len(ok) < 2 or len(set(groups)) < 3:
            report.append(row)
            print(task, "not enough labels:", counts.to_dict())
            continue
        Xy = np.stack([emb[r] for r in y.index])
        oof = np.empty((len(y), len(ok)))
        cv = GroupKFold(n_splits=min(5, len(set(groups))))
        for tr, te in cv.split(Xy, y.values, groups):
            if y.iloc[tr].nunique() < len(ok):
                oof[te] = np.nan
                continue
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xy[tr], y.values[tr])
            oof[te] = m.predict_proba(Xy[te])
        good = ~np.isnan(oof).any(1)
        cls = np.array(sorted(ok.index))
        row["cv_balanced_acc"] = balanced_accuracy_score(y.values[good], cls[oof[good].argmax(1)])
        if len(cls) == 2:
            row["cv_auc"] = roc_auc_score((y.values[good] == cls[1]).astype(int), oof[good, 1])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xy, y.values)
        P = m.predict_proba(X)
        R[[f"p_{task}:{c}" for c in m.classes_]] = P
        R[f"pred_{task}"] = m.classes_[P.argmax(1)]
        R[f"conf_{task}"] = P.max(1)
        row["trained"] = True
        report.append(row)
        print(task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    # tasks that were not retrained keep their previous columns
    keep = [c for c in prev.columns if c not in R.columns]
    R = R.merge(prev[["roi_id"] + keep], on="roi_id", how="left")
    if "pred_artifact" in R and R.pred_artifact.dtype == object:
        R["pred_artifact"] = R.pred_artifact.eq("yes") | R.pred_artifact.eq(True)
    trained = {r["task"] for r in report if r["trained"]}
    conf = R.conf_tumor.where(R.pred_tumor != "tumor", np.fmin(R.conf_tumor, R.conf_pattern)).fillna(0).values
    lo, hi = (LO, HI) if "tumor" in trained else (1 / 3, 2 / 3)
    S = X @ X.T
    S[R.slide_id.values[:, None] == R.slide_id.values[None, :]] = -1
    R["nn_sim"] = np.sort(S, 1)[:, -5:].mean(1)
    R["ood"] = R.nn_sim < np.percentile(R.nn_sim, 5)
    blocked = R.get("frac_qc_fail", pd.Series(0, index=R.index)).fillna(0).gt(0.5) | R.pred_artifact.fillna(False).astype(bool)
    R["confidence"], R["uncertainty"] = conf, 1 - conf
    R["state"] = triage(conf, R.ood.values, blocked.values, lo, hi)
    R["model_version"] = tag
    R.to_csv(ANN / f"rois_v{version:03d}.csv", index=False)
    pd.DataFrame(report).to_csv(ANN / f"refine_report_v{version:03d}.csv", index=False)
    print("wrote", f"rois_v{version:03d}.csv", R.state.value_counts().to_dict())


if __name__ == "__main__":
    main()
