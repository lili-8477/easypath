"""Slide review: the segments of one slide, a model-predicted label for each (SS pattern or stroma type from
CONCH zero-shot patch scores averaged over the segment), reviewer labels per segment and the slide-level
synovial sarcoma subtype. Segment labels given here override the class-level label on that slide only."""
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage as ndi

from ann_common import ANN, MAP_DIR, SEG_SLIDE_LABELS, SLIDE_PATTERNS, SLIDE_SUBTYPES
from ann_db import connect, latest_slide_reviews
from clusters import (LABEL_COLORS, SEG_MODE, _area, _hex, _load, cl, class_map, effective_labels, group_pick, merges, meta,
                      read_summary, slide_labels, subtype_call)

TISSUE_TO_LABEL = {"stroma": "stroma", "necrosis": "necrosis", "inflammation": "inflammation",
                   "benign_epithelium": "normal_tissue", "fat": "fat", "muscle": "muscle", "blood": "hemorrhage"}
PATTERN_TO_LABEL = {"spindle": "ss_spindle", "glandular": "ss_glandular", "corded": "ss_corded", "round_cell": "ss_round_cell"}
STROMA_TO_LABEL = {"fibrous": "stroma", "hyalinized": "stroma_hyalinized", "myxoid": "stroma_myxoid",
                   "calcified": "calcification"}
LIFT_MIN = 1.25   # a non-default pattern / stroma type needs 1.25 x the default's lift over the cohort baseline
BASE_MIN = 0.02    # floor on a cohort baseline, so a class the model almost never scores (hyalinized) is not inflated
SUB = 4           # label pixels per side pooled when mapping segments onto the patch grid (16 um)
router = APIRouter()
_stats = {}


def _prior():
    p = ANN / "tissue_map_ss_prior.json"
    return json.load(open(p)) if p.exists() else None


def _lifted(p, names, base, default):
    """Class furthest above its cohort baseline; the default unless another is clearly higher."""
    rel = np.array([p[i] / max(base[n], BASE_MIN) for i, n in enumerate(names)])
    if not np.isfinite(rel).all() or rel.sum() <= 0:   # no tumour / stroma weight under this segment
        return default, None
    j, d = int(rel.argmax()), names.index(default)
    if rel[j] < LIFT_MIN * rel[d]:
        j = d
    return names[j], {n: round(float(v), 3) for n, v in zip(names, rel / rel.sum())}


def segment_stats(sid):
    """Per segment on this slide: area, mean CONCH tissue probs, SS pattern and stroma-type scores, prediction."""
    p = cl() / "labels" / f"{sid}.png"
    if not SEG_MODE or not p.exists() or not (MAP_DIR / f"{sid}_conch.npz").exists():
        raise HTTPException(404, "no segmentation for this slide")
    cmap = class_map(sid)
    key = (p.stat().st_mtime, (ANN / "tissue_map_ss_prior.json").exists(), tuple(sorted(cmap.items())))
    if (p, sid) in _stats and _stats[(p, sid)][0] == key:
        return _stats[(p, sid)][1]
    lab = np.asarray(Image.open(p))
    m = json.load(open(cl() / "labels" / f"{sid}.json"))
    members = {g["segment"]: g["members"] for g in m.get("groups", [])}
    K = meta()["k"]
    S = _load("summary.csv", read_summary).set_index("cluster")
    area = np.bincount(lab[lab != 255].ravel(), minlength=K)[:K] * m["out_um"] ** 2 / 1e6

    z = np.load(MAP_DIR / f"{sid}_conch.npz")
    tcls = [str(c) for c in z["classes"]]
    step, N = int(z["step"]), len(z["probs"])
    sub = lab[::SUB, ::SUB]
    gx = ((m["x0"] + (np.arange(sub.shape[1]) * SUB + SUB / 2) * m["px_l0"]) // step).astype(int)
    gy = ((m["y0"] + (np.arange(sub.shape[0]) * SUB + SUB / 2) * m["px_l0"]) // step).astype(int)
    G = np.full((max(gy.max(), z["gy"].max()) + 1, max(gx.max(), z["gx"].max()) + 1), -1, np.int64)
    G[z["gy"], z["gx"]] = np.arange(N)
    pi = G[gy[:, None], gx[None, :]]
    ok = (sub != 255) & (pi >= 0)
    W = np.bincount(sub[ok].astype(np.int64) * N + pi[ok], minlength=K * N).reshape(K, N).astype(float)
    # one row per segment: a class, or a reviewer-merged group of classes under its representative
    reps = sorted({cmap.get(k, k) for k in range(K) if area[k] > 0})
    cls = [[k for k in range(K) if cmap.get(k, k) == r and area[k] > 0] for r in reps]
    W = np.stack([W[c].sum(0) for c in cls])
    area = np.array([area[c].sum() for c in cls])
    tissue = W @ z["probs"] / np.maximum(W.sum(1, keepdims=True), 1)

    prior, ss = _prior(), MAP_DIR / f"{sid}_ss.npz"
    if prior and ss.exists():
        zs = np.load(ss)
        pnames, snames = [str(c) for c in zs["pattern_classes"]], [str(c) for c in zs["stroma_classes"]]
        PP, PS = zs["pattern_probs"], zs["stroma_probs"]
    else:   # a2c not run yet: a2b patterns (no corded), no stroma types
        pnames, snames = [str(c) for c in z["pattern_classes"]], []
        PP, PS = z["pattern_probs"], None
        prior = {"pattern": {n: 1.0 for n in pnames}}
    wt, ws = W * z["probs"][:, tcls.index("tumor")], W * z["probs"][:, tcls.index("stroma")]
    pat = wt @ PP / np.maximum(wt.sum(1, keepdims=True), 1e-9)
    stro = ws @ PS / np.maximum(ws.sum(1, keepdims=True), 1e-9) if PS is not None else None

    rows, tot = [], float(area.sum())
    for k, (rep, c) in enumerate(zip(reps, cls)):
        top = tcls[int(tissue[k].argmax())]
        pattern_rel = stroma_rel = None
        if top == "tumor":
            pk, pattern_rel = _lifted(pat[k], pnames, prior["pattern"], "spindle")
            pred = PATTERN_TO_LABEL[pk]
        elif top == "stroma" and stro is not None:
            sk, stroma_rel = _lifted(stro[k], snames, prior["stroma"], "fibrous")
            pred = STROMA_TO_LABEL[sk]
        else:
            pred = TISSUE_TO_LABEL.get(top, top)
        cc, n = ndi.label(np.isin(sub, c))
        if n:   # largest connected piece, for "zoom to segment"
            big = int(np.argmax(ndi.sum(np.ones_like(cc), cc, range(1, n + 1)))) + 1
            ys, xs = ndi.find_objects((cc == big).astype(np.int32))[0]
            bbox = dict(x=m["x0"] + xs.start * SUB * m["px_l0"], y=m["y0"] + ys.start * SUB * m["px_l0"],
                        w=(xs.stop - xs.start) * SUB * m["px_l0"], h=(ys.stop - ys.start) * SUB * m["px_l0"])
        else:
            bbox = None
        rows.append(dict(segment=rep, classes=c, merged=[x for x in c if x != rep],
                         color=S.loc[rep, "color"], area_mm2=round(float(area[k]), 4),
                         frac=round(float(area[k] / tot), 4), pred_label=pred,
                         pred_conf=round(float(tissue[k].max()), 3),
                         tissue={c: round(float(v), 3) for c, v in zip(tcls, tissue[k])},
                         pattern=pattern_rel, stroma=stroma_rel, bbox=bbox,
                         members=sorted({x for y in c for x in members.get(y, [y])})))
    _stats[(p, sid)] = (key, rows)
    return rows


def _area_by(rows, field, labs=None):
    a = {}
    for r in rows:
        v = labs.get(r["segment"]) if labs is not None else r[field]
        if v:
            a[v] = a.get(v, 0) + r["area_mm2"]
    return a


@router.get("/api/slides/{sid}/segments")
def segments(sid: str):
    rows = segment_stats(sid)
    glob, over = effective_labels()
    sl, area = over.get(sid, {}), _area(sid)
    labs = slide_labels(sid, glob, over)
    g_all, g_slide = merges(sid)
    in_slide = {k for g in g_slide for k in g}
    out = [dict(r, class_label=group_pick(glob, r["segment"], r["classes"], area),
                slide_label=group_pick(sl, r["segment"], r["classes"], area), label=labs.get(r["segment"]),
                merge_scope=None if not r["merged"] else "slide" if in_slide & set(r["classes"]) else "all")
           for r in rows]
    tot = sum(r["area_mm2"] for r in rows)
    pa = _area_by(rows, "pred_label")
    con = connect()
    hist = [dict(h) for h in con.execute("SELECT * FROM slide_reviews WHERE slide_id=? ORDER BY id DESC", (sid,))]
    con.close()
    return dict(clustering=meta()["id"], segments=out, review=hist[0] if hist else None, history=hist,
                predicted=dict(subtype_call(pa), stroma_frac=round(sum(v for k, v in pa.items()
                                                                       if k.startswith("stroma") or k == "calcification") / max(tot, 1e-9), 3)),
                derived=subtype_call(_area_by(rows, None, labs)), merges=dict(all_slides=g_all, slide=g_slide),
                labels=SEG_SLIDE_LABELS, subtypes=SLIDE_SUBTYPES, patterns=SLIDE_PATTERNS,
                label_colors={k: _hex(v) for k, v in LABEL_COLORS.items()},
                patterns_ready=_prior() is not None)


class SlideReview(BaseModel):
    reviewer: str
    subtype: str
    patterns: List[str] = []
    segments: Dict[int, str] = {}
    comment: Optional[str] = ""


@router.post("/api/slides/{sid}/slide_review")
def slide_review(sid: str, body: SlideReview):
    if not body.reviewer.strip():
        raise HTTPException(400, "reviewer required")
    if body.subtype not in SLIDE_SUBTYPES or any(p not in SLIDE_PATTERNS for p in body.patterns):
        raise HTTPException(400, "bad subtype or pattern")
    if any(v not in SEG_SLIDE_LABELS for v in body.segments.values()):
        raise HTTPException(400, "bad segment label")
    rows = {r["segment"]: r for r in segment_stats(sid)}
    cur = slide_labels(sid)
    cid, ts, who = meta()["id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), body.reviewer.strip()
    con = connect()
    for k, v in body.segments.items():
        if k in rows and cur.get(k) != v:
            for c in rows[k]["classes"]:
                con.execute("""INSERT INTO slide_seg_reviews (slide_id, segment, reviewer, ts, label, predicted, clustering)
                               VALUES (?,?,?,?,?,?,?)""", (sid, c, who, ts, v, rows[k]["pred_label"], cid))
    pred = subtype_call(_area_by(list(rows.values()), "pred_label"))["call"]
    con.execute("""INSERT INTO slide_reviews (slide_id, reviewer, ts, subtype, patterns, comment, predicted_subtype, clustering)
                   VALUES (?,?,?,?,?,?,?,?)""", (sid, who, ts, body.subtype, ",".join(body.patterns), body.comment or "", pred, cid))
    con.commit()
    con.close()
    return {"ok": True}


@router.get("/api/slide_reviews")
def slide_reviews():
    con = connect()
    df = latest_slide_reviews(con)
    con.close()
    return {r.slide_id: dict(subtype=r.subtype, patterns=r.patterns, reviewer=r.reviewer, ts=r.ts) for r in df.itertuples()}


@router.get("/api/export/slide_reviews.csv")
def export_slide_reviews():
    con = connect()
    a = pd.read_sql_query("SELECT * FROM slide_reviews ORDER BY id", con)
    b = pd.read_sql_query("SELECT * FROM slide_seg_reviews ORDER BY id", con)
    c = pd.read_sql_query("SELECT * FROM merge_states ORDER BY id", con).rename(columns={"scope": "slide_id"})
    con.close()
    a.insert(0, "record", "slide")
    b.insert(0, "record", "segment")
    c.insert(0, "record", "merge")
    return Response(pd.concat([a, b, c], ignore_index=True).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=hss_slide_reviews_audit.csv"})
