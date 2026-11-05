"""Slide review: the segments of one slide, a model-predicted label for each (SS pattern or stroma type from
CONCH zero-shot patch scores averaged over the segment), reviewer labels and comments per segment, reviewer edits
of the segmentation, and the slide-level synovial sarcoma subtype. Segment labels given here override the
class-level label on that slide only."""
import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from scipy import ndimage as ndi

import segedit
from ann_common import ANN, CLUSTER_COLORS, MAP_DIR, SEG_SLIDE_LABELS, SLIDE_PATTERNS, SLIDE_SUBTYPES
from ann_db import connect, latest_slide_reviews
from clusters import (LABEL_COLORS, SEG_MODE, _area, _hex, _load, cl, class_map, edit_state, effective_labels, group_pick,
                      label_map, merges, meta, read_summary, seg_meta, slide_labels, subtype_call)

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
    active = edit_state(sid)[0]
    key = (p.stat().st_mtime, (ANN / "tissue_map_ss_prior.json").exists(), tuple(sorted(cmap.items())), tuple(e["id"] for e in active))
    if (p, sid) in _stats and _stats[(p, sid)][0] == key:
        return _stats[(p, sid)][1]
    lab = label_map(sid, active)
    m = json.load(open(cl() / "labels" / f"{sid}.json"))
    members = {g["segment"]: g["members"] for g in m.get("groups", [])}
    K = max(meta()["k"], int(lab[lab != segedit.BACKGROUND].max(initial=0)) + 1)   # reviewer-added segments come after the model's
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
                         color=S.loc[rep, "color"] if rep in S.index else CLUSTER_COLORS[rep % len(CLUSTER_COLORS)],
                         added=rep >= meta()["k"], area_mm2=round(float(area[k]), 4),
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
    notes = [dict(c) for c in con.execute("SELECT segment, reviewer, ts, comment FROM segment_comments WHERE clustering=? AND slide_id=? ORDER BY id",
                                          (meta()["id"], sid))]
    con.close()
    for r in out:
        r["comments"] = [c for c in notes if c["segment"] in r["classes"]]
    active, undone = edit_state(sid)
    return dict(clustering=meta()["id"], k=meta()["k"], segments=out,
                edits=dict(can_undo=bool(active), can_redo=bool(undone),
                           log=[dict(action=e["action"], segment=e["segment"], reviewer=e["reviewer"], ts=e["ts"]) for e in reversed(active)]), review=hist[0] if hist else None, history=hist,
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
    cur = effective_labels()[1].get(sid, {})   # a label accepted here is recorded even if it matches a class label
    cid, ts, who = meta()["id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), body.reviewer.strip()
    con = connect()
    for k, v in body.segments.items():
        if k in rows and any(cur.get(c) != v for c in rows[k]["classes"]):
            for c in rows[k]["classes"]:
                con.execute("""INSERT INTO slide_seg_reviews (slide_id, segment, reviewer, ts, label, predicted, clustering)
                               VALUES (?,?,?,?,?,?,?)""", (sid, c, who, ts, v, rows[k]["pred_label"], cid))
    pred = subtype_call(_area_by(list(rows.values()), "pred_label"))["call"]
    con.execute("""INSERT INTO slide_reviews (slide_id, reviewer, ts, subtype, patterns, comment, predicted_subtype, clustering)
                   VALUES (?,?,?,?,?,?,?,?)""", (sid, who, ts, body.subtype, ",".join(body.patterns), body.comment or "", pred, cid))
    con.commit()
    con.close()
    return {"ok": True}


# ---------------------------------------------------------------- segmentation edits and segment comments
class SegEdit(BaseModel):
    reviewer: str
    action: Literal["paint", "stroke", "reshape", "undo", "redo"]
    segment: Optional[int] = None    # class to paint; 255 erases; None with paint / stroke starts a new segment
    geometry: Optional[dict] = None


MAX_POINTS = 20000


def _points(pts, min_n):
    ok = isinstance(pts, list) and min_n <= len(pts) <= MAX_POINTS and all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in pts)
    if not ok:
        raise HTTPException(400, "bad geometry")
    return pts


def _new_segment(sid, active):
    """A free id for a segment added on this slide: after the model's classes and earlier additions."""
    used = [e["segment"] for e in active if e["segment"] != segedit.BACKGROUND] + [meta()["k"] - 1]
    k = max(used) + 1
    if k >= segedit.BACKGROUND:
        raise HTTPException(400, "too many segments on this slide")
    return k


@router.post("/api/slides/{sid}/edits")
def add_edit(sid: str, body: SegEdit):
    if seg_meta(sid) is None:
        raise HTTPException(404, "no segmentation for this slide")
    who = body.reviewer.strip()
    if not who:
        raise HTTPException(400, "reviewer required")
    active, undone = edit_state(sid)
    target, seg, geom = None, body.segment, None
    if body.action in ("undo", "redo"):
        pool = active if body.action == "undo" else undone
        if not pool:
            raise HTTPException(409, f"nothing to {body.action}")
        target, seg = pool[-1]["id"], pool[-1]["segment"]
    else:
        g = body.geometry or {}
        if body.action == "paint":
            polys = g.get("polygons")
            if not isinstance(polys, list) or not polys:
                raise HTTPException(400, "bad geometry")
            geom = dict(polygons=[_points(r, 3) for r in polys])
        elif body.action == "stroke":
            w = g.get("width")
            if not isinstance(w, (int, float)) or not 0 < w < 1e6:
                raise HTTPException(400, "bad brush width")
            geom = dict(points=_points(g.get("points"), 1), width=w)
        else:
            cls = g.get("classes")
            if seg is None or not isinstance(cls, list) or not cls or not all(isinstance(c, int) and 0 <= c < segedit.BACKGROUND for c in cls):
                raise HTTPException(400, "bad geometry")
            geom = dict(old=_points(g.get("old"), 3), new=_points(g.get("new"), 3), classes=cls)
        if seg is None:
            seg = _new_segment(sid, active + undone)
        if not 0 <= seg <= segedit.BACKGROUND:
            raise HTTPException(400, "bad segment")
    con = connect()
    cur = con.execute("""INSERT INTO seg_edits (clustering, slide_id, reviewer, ts, action, segment, target, geometry)
                         VALUES (?,?,?,?,?,?,?,?)""", (meta()["id"], sid, who, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                       body.action, seg, target, json.dumps(geom) if geom else None))
    con.commit()
    eid = cur.lastrowid
    con.close()
    return {"ok": True, "id": eid, "segment": seg}


@router.get("/api/slides/{sid}/outlines")
def segment_outlines(sid: str):
    if seg_meta(sid) is None:
        raise HTTPException(404, "no segmentation for this slide")
    return segedit.outlines(label_map(sid), seg_meta(sid), class_map(sid))


class SegComment(BaseModel):
    reviewer: str
    segment: int
    comment: str


@router.post("/api/slides/{sid}/segment_comments")
def add_segment_comment(sid: str, body: SegComment):
    if seg_meta(sid) is None:
        raise HTTPException(404, "no segmentation for this slide")
    who, text = body.reviewer.strip(), body.comment.strip()
    if not who or not text:
        raise HTTPException(400, "reviewer and comment required")
    con = connect()
    con.execute("INSERT INTO segment_comments (clustering, slide_id, segment, reviewer, ts, comment) VALUES (?,?,?,?,?,?)",
                (meta()["id"], sid, body.segment, who, datetime.now(timezone.utc).isoformat(timespec="seconds"), text))
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
    d = pd.read_sql_query("SELECT * FROM seg_edits ORDER BY id", con)
    e = pd.read_sql_query("SELECT * FROM segment_comments ORDER BY id", con)
    con.close()
    a.insert(0, "record", "slide")
    b.insert(0, "record", "segment")
    c.insert(0, "record", "merge")
    d.insert(0, "record", "segmentation_edit")
    e.insert(0, "record", "segment_comment")
    return Response(pd.concat([a, b, c, d, e], ignore_index=True).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=hss_slide_reviews_audit.csv"})
