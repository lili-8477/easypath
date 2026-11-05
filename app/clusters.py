"""Cluster-level review: token clusters from c1_token_clusters.py, exemplar galleries, reviewer labels
(append-only, keyed by clustering id) and the token cluster / label overlay for the slide viewer."""
import io
import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import openslide
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageDraw
from pydantic import BaseModel

from ann_common import (ANN, BIPHASIC_MIN_EPI, CLUSTER_COLORS, CLUSTER_LABELS, CLUSTER_PURITY, MAP_DIR, PATTERN_LABELS,
                        PD_MIN_ROUND, RES,
                        SS_COMPONENTS, slide_path)
from ann_db import connect, latest_cluster_reviews, latest_merges, latest_slide_seg_reviews, seg_edit_log
import segedit



def runs():
    """Smooth segmentation runs (s2 output folders results/annot/seg*), newest first: {folder: clustering.json}."""
    ps = sorted(ANN.glob("seg*/clustering.json"), key=lambda p: -p.stat().st_mtime)
    return {p.parent.name: json.load(open(p)) for p in ps}


# the active segmentation run (SEG_DIR, else the newest) is reviewed; without one, the token clusters (c1)
_active = {"dir": os.environ.get("SEG_DIR") or next(iter(runs()), None)}
SEG_MODE = _active["dir"] is not None


def cl():
    return ANN / _active["dir"] if SEG_MODE else ANN / "clusters"

# no pink, purple or blue (they disappear into H&E); SS components are the four strongest colours
LABEL_COLORS = {"ss_spindle": (225, 0, 0), "ss_glandular": (255, 212, 0), "ss_corded": (255, 106, 0),
                "ss_round_cell": (17, 17, 17), "stroma": (0, 176, 80), "necrosis": (139, 74, 0),
                "inflammation": (140, 140, 0), "hemorrhage": (110, 0, 0), "fat": (255, 245, 170),
                "muscle": (92, 46, 0), "vessel": (200, 160, 0), "normal_tissue": (0, 90, 30),
                "stroma_hyalinized": (166, 255, 0), "stroma_myxoid": (0, 255, 176), "calcification": (255, 255, 255),
                "artifact": (127, 127, 127), "mixed": (200, 200, 200)}
MAN = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
router = APIRouter()
_cache = {}


def _load(name, loader):
    p = cl() / name
    if not p.exists():
        return None
    m = p.stat().st_mtime
    if str(p) not in _cache or _cache[str(p)][0] != m:
        _cache[str(p)] = (m, loader(p))
    return _cache[str(p)][1]


def read_summary(p):
    """summary.csv with the current palette (runs made before a palette change keep their stored colours otherwise)."""
    S = pd.read_csv(p)
    S["color"] = [CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)] for k in S.cluster]
    return S


def meta():
    m = _load("clustering.json", lambda p: json.load(open(p)))
    if m is None:
        raise HTTPException(404, "no clustering yet")
    return m


def tokens():
    return _load("tokens.parquet", pd.read_parquet)


def labels_now():
    con = connect()
    lv = latest_cluster_reviews(con, meta()["id"]).set_index("cluster")
    con.close()
    return lv


def _hex(c):
    return "#%02x%02x%02x" % c


@router.get("/api/clusters")
def clusters():
    m = meta()
    S = _load("summary.csv", read_summary)
    E = _load("exemplars.csv", pd.read_csv)
    lv = labels_now()
    cmap = class_map()
    grp = groups_of(cmap)
    out = []
    for r in S.to_dict("records"):
        k = int(r["cluster"])
        r = {**r, "merged_into": cmap[k] if cmap.get(k, k) != k else None, "group": grp.get(k) if cmap.get(k) == k else None}
        ex = E[E.cluster == k][["name", "kind", "slide_id", "x0", "y0", "size"]].to_dict("records")
        rev = lv.loc[k].to_dict() if k in lv.index else None
        out.append({**{a: (None if isinstance(b, float) and math.isnan(b) else b) for a, b in r.items()},
                    "exemplars": ex, "review": rev})
    return {"clustering": {**m, "kind": m.get("kind", "tokens")}, "clusters": out, "labels": CLUSTER_LABELS,
            "patterns": [] if SEG_MODE else PATTERN_LABELS + ["mixed"],
            "purity": CLUSTER_PURITY, "label_colors": {k: _hex(v) for k, v in LABEL_COLORS.items()}}


class RunChoice(BaseModel):
    dir: str


@router.get("/api/clusterings")
def clusterings():
    return {"active": _active["dir"], "runs": [dict(dir=d, id=m["id"], k=m["k"], encoder=m.get("encoder", "conch15"),
                                                   adaptive=m.get("adaptive", False), k_slide_median=m.get("k_slide_median"),
                                                   method=m.get("method", "")) for d, m in runs().items()]}


@router.post("/api/clusterings/active")
def set_clustering(body: RunChoice):
    """Switch the segmentation run the whole app reviews (labels are stored per run id)."""
    if body.dir not in runs():
        raise HTTPException(404, "no such run")
    _active["dir"] = body.dir
    return {"ok": True}


class MergeBody(BaseModel):
    reviewer: str
    scope: str              # '*' = all slides, else a slide id
    classes: List[int]
    action: str = "merge"   # merge | unmerge


@router.get("/api/merges")
def get_merges(slide: Optional[str] = None):
    g, s = merges(slide)
    return {"all_slides": g, "slide": s, "map": {str(k): r for k, r in class_map(slide).items()}}


@router.post("/api/merges")
def set_merges(body: MergeBody):
    """Merge segment classes into one segment (or undo it), on one slide or on all slides. Append-only:
    every change stores the scope's full merge state, so the table is the audit trail."""
    m = meta()
    if not SEG_MODE:
        raise HTTPException(400, "merging needs a segmentation run")
    if not body.reviewer.strip():
        raise HTTPException(400, "reviewer required")
    if body.scope != "*" and body.scope not in MAN.index:
        raise HTTPException(404, "no such slide")
    sel = set(body.classes)
    # segments a reviewer added exist on one slide only (ids from k up), so they can merge there but not on all slides
    top = m["k"] if body.scope == "*" else segedit.BACKGROUND
    if body.action not in ("merge", "unmerge") or not sel or not all(0 <= k < top for k in sel):
        raise HTTPException(400, "bad merge request")
    con = connect()
    cur = latest_merges(con, m["id"], body.scope)
    touch = [g for g in cur if sel & set(g)]
    if body.action == "merge":
        if len(sel) < 2 and not touch:
            con.close()
            raise HTTPException(400, "select at least two segments")
        groups = [g for g in cur if g not in touch] + [sorted(sel.union(*map(set, touch)))]
    else:
        if not touch:
            other = latest_merges(con, m["id"], "*") if body.scope != "*" else []
            con.close()
            raise HTTPException(409, "merged on all slides; unmerge them there" if any(sel & set(g) for g in other)
                                else "these segments are not merged")
        groups = [g for g in cur if g not in touch]
    con.execute("""INSERT INTO merge_states (clustering, scope, groups, action, classes, reviewer, ts)
                   VALUES (?,?,?,?,?,?,?)""", (m["id"], body.scope, json.dumps(groups), body.action,
                                               ",".join(map(str, sorted(sel))), body.reviewer.strip(),
                                               datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return {"ok": True, "groups": groups}


@router.get("/api/clusters/ex/{name}.jpg")
def exemplar(name: str):
    p = (cl() / "ex" / f"{name}.jpg").resolve()
    if p.parent != (cl() / "ex").resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@router.get("/api/clusters/{k}/sample")
def sample(k: int, n: int = 24, seed: int = 1):
    if SEG_MODE:   # pre-rendered extra exemplars, served in pages
        E = _load("exemplars.csv", pd.read_csv)
        g = E[(E.cluster == k) & (E.kind == "more")]
        return g.iloc[(seed - 1) * n: seed * n][["name", "slide_id", "x0", "y0", "size"]].to_dict("records")
    T = tokens()
    g = T[T.cluster == k]
    g = g.sample(min(n, len(g)), random_state=seed)
    return g[["slide_id", "x0", "y0", "size", "dist"]].to_dict("records")


_slides = {}


def _slide(sid):
    if sid not in _slides:
        _slides[sid] = openslide.OpenSlide(slide_path(MAN.loc[sid, "file"]))
    return _slides[sid]


@router.get("/api/token_crop.jpg")
def token_crop(sid: str, x0: int, y0: int, size: float, context: int = 4):
    """Token box drawn inside its surrounding tissue (context x token size), 256 px."""
    if sid not in MAN.index or not 1 <= context <= 12:
        raise HTTPException(404)
    s = _slide(sid)
    ctx = int(round(size * context))
    cx, cy = int(x0 + size / 2 - ctx / 2), int(y0 + size / 2 - ctx / 2)
    im = s.read_region((cx, cy), 0, (ctx, ctx)).convert("RGB").resize((256, 256))
    a, b = 128 - 128 / context, 128 + 128 / context
    d = ImageDraw.Draw(im)
    d.rectangle([a - 1, a - 1, b + 1, b + 1], outline=(0, 0, 0), width=1)
    d.rectangle([a, a, b, b], outline=(255, 255, 255), width=2)
    buf = io.BytesIO()
    im.save(buf, "jpeg", quality=88)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


class ClusterReview(BaseModel):
    reviewer: str
    label: str
    pattern: Optional[str] = None
    purity: Optional[str] = "clean"
    batch_artifact: bool = False
    comment: Optional[str] = ""


@router.post("/api/clusters/{k}/review")
def review(k: int, body: ClusterReview):
    m = meta()
    if not 0 <= k < m["k"]:
        raise HTTPException(404)
    if not body.reviewer.strip():
        raise HTTPException(400, "reviewer required")
    if body.label not in CLUSTER_LABELS or body.purity not in CLUSTER_PURITY:
        raise HTTPException(400, "bad label")
    pattern = body.pattern if body.label == "tumor" and not SEG_MODE else None
    if pattern is not None and pattern not in PATTERN_LABELS + ["mixed"]:
        raise HTTPException(400, "bad pattern")
    con = connect()
    con.execute("""INSERT INTO cluster_reviews (cluster, reviewer, ts, label, pattern, purity, batch_artifact, comment, clustering)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (k, body.reviewer.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds"), body.label, pattern,
                 body.purity, int(body.batch_artifact), body.comment or "", m["id"]))
    con.commit()
    con.close()
    return {"ok": True}


@router.get("/api/clusters/{k}/history")
def history(k: int):
    con = connect()
    rows = [dict(r) for r in con.execute("SELECT * FROM cluster_reviews WHERE cluster=? AND clustering=? ORDER BY id DESC",
                                         (k, meta()["id"]))]
    con.close()
    return rows


@router.get("/api/slides/{sid}/clustermap/{mode}.png")
def clustermap(sid: str, mode: str):
    """mode: all (every cluster in its colour), labels (reviewer labels), or a cluster number (only that one)."""
    if SEG_MODE:
        return segmap(sid, mode)
    p = MAP_DIR / f"{sid}_clusters.npz"
    if not p.exists():
        raise HTTPException(404)
    z = np.load(p)
    step = float(z["tsize"]) * int(z["grid"])
    gw = math.ceil(MAN.loc[sid, "width"] / step) * int(z["grid"])
    gh = math.ceil(MAN.loc[sid, "height"] / step) * int(z["grid"])
    S = _load("summary.csv", read_summary)
    pal = {int(r.cluster): tuple(int(r.color[i:i + 2], 16) for i in (1, 3, 5)) for r in S.itertuples()}
    cl = z["cluster"].astype(int)
    rgba = np.zeros((gh, gw, 4), np.uint8)
    if mode == "all":
        col, keep = np.array([pal[c] for c in cl]), np.ones(len(cl), bool)
    elif mode == "labels":
        lv = labels_now()
        lab = np.array([lv.loc[c, "label"] if c in lv.index else "" for c in cl])
        keep = lab != ""
        col = np.array([LABEL_COLORS.get(v, (0, 0, 0)) for v in lab]).reshape(-1, 3)
    else:
        try:
            k = int(mode)
        except ValueError:
            raise HTTPException(404)
        keep = cl == k
        col = np.tile(pal.get(k, (255, 255, 255)), (len(cl), 1))
    rgba[z["ty"][keep], z["tx"][keep], :3] = col[keep]
    rgba[z["ty"][keep], z["tx"][keep], 3] = 190
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "png")
    return Response(buf.getvalue(), media_type="image/png")


@router.get("/api/export/cluster_labels.csv")
def export_labels():
    con = connect()
    df = pd.read_sql_query("SELECT * FROM cluster_reviews ORDER BY id", con)
    con.close()
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=hss_cluster_reviews_audit.csv"})


def seg_meta(sid):
    p = cl() / "labels" / f"{sid}.json"
    return json.load(open(p)) if SEG_MODE and p.exists() else None


_maps = OrderedDict()   # (run, slide) -> (mtime of the model's map, ids of the edits applied, label map)


def edit_state(sid):
    """(edits in force, undone edits) on this slide in the active run."""
    con = connect()
    st = seg_edit_log(con, meta()["id"], sid)
    con.close()
    return st


def label_map(sid, edits=None):
    """The slide's current label map: the model's, with the reviewer edits in force. Do not modify the result."""
    p = cl() / "labels" / f"{sid}.png"
    active = edit_state(sid)[0] if edits is None else edits
    ids, key, mtime = [e["id"] for e in active], (_active["dir"], sid), p.stat().st_mtime
    hit = _maps.get(key)
    if hit and hit[0] == mtime and hit[1] == ids:
        _maps.move_to_end(key)
        return hit[2]
    resume = hit is not None and hit[0] == mtime and ids[:len(hit[1])] == hit[1]   # usually one new edit on top
    if resume:
        lab = segedit.replay(None, seg_meta(sid), active[len(hit[1]):], start=hit[2])
    else:
        lab = segedit.replay(np.asarray(Image.open(p)), seg_meta(sid), active)
    _maps[key] = (mtime, ids, lab)
    _maps.move_to_end(key)
    while len(_maps) > 8:
        _maps.popitem(last=False)
    return lab


def edited_slides():
    con = connect()
    rows = con.execute("SELECT DISTINCT slide_id FROM seg_edits WHERE clustering = ?", (meta()["id"],)).fetchall()
    con.close()
    return {r["slide_id"] for r in rows}


def _area(sid=None):
    """Area (mm2) per segment class on one slide, else over the cohort."""
    if sid is not None:
        if edit_state(sid)[0]:
            return segedit.areas_mm2(label_map(sid), seg_meta(sid))
        A = _load("area_mm2.csv", lambda p: pd.read_csv(p, index_col=0))
        if A is not None and sid in A.index:
            return {int(k): float(v) for k, v in A.loc[sid].items()}
    S = _load("summary.csv", read_summary)
    return {int(r.cluster): float(r.area_mm2) for r in S.itertuples()} if "area_mm2" in S else {}


def merges(sid=None):
    """Reviewer merges in force for the active run: (all-slides groups, this slide's groups)."""
    if not SEG_MODE:
        return [], []
    con = connect()
    cid = meta()["id"]
    g, s = latest_merges(con, cid, "*"), (latest_merges(con, cid, sid) if sid else [])
    con.close()
    return g, s


def class_map(sid=None):
    """{class: representative} for merged classes (all-slides merges, plus this slide's when sid is given).
    The representative is the member with the largest area on the slide (else in the cohort)."""
    g, s = merges(sid)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for grp in g + s:
        for k in grp[1:]:
            a, b = find(grp[0]), find(k)
            if a != b:
                parent[b] = a
    comp = {}
    for k in {k for grp in g + s for k in grp}:
        comp.setdefault(find(k), []).append(k)
    area, out = _area(sid), {}
    for mem in comp.values():
        r = max(mem, key=lambda k: (area.get(k, 0), -k))
        out.update({k: r for k in mem})
    return out


def groups_of(cmap):
    """{representative: [members]} for merged classes only."""
    out = {}
    for k, r in cmap.items():
        out.setdefault(r, []).append(k)
    return {r: sorted(m) for r, m in out.items()}


def group_pick(d, r, mem, area):
    """A merged group's value from {class: value}: the representative's, else the largest member's that has one."""
    order = [r] + sorted((k for k in mem if k != r), key=lambda k: -area.get(k, 0))
    return next((d[k] for k in order if d.get(k)), None)


def effective_labels():
    """Class-level labels {segment: label} and per-slide overrides {slide_id: {segment: label}}."""
    lv = labels_now()
    glob = {int(k): lv.loc[k, "label"] for k in lv.index}
    con = connect()
    sl = latest_slide_seg_reviews(con, meta()["id"])
    con.close()
    over = {}
    for r in sl.itertuples():
        over.setdefault(r.slide_id, {})[int(r.segment)] = r.label
    return glob, over


def slide_labels(sid, glob=None, over=None):
    """Label per class on a slide: per-slide labels over class labels; a merged group carries one label for all
    its members (a per-slide label of any member first, then the class labels)."""
    if glob is None:
        glob, over = effective_labels()
    sl = over.get(sid, {})
    labs = {**glob, **sl}
    area = _area(sid)
    for r, mem in groups_of(class_map(sid)).items():
        v = group_pick(sl, r, mem, area) or group_pick(glob, r, mem, area)
        if v:
            labs.update({k: v for k in mem})
    return labs


def _mode_colors(mode, K, sid=None):
    """Colour per segment class for the overlay: all / labels (class level) / slide (labels on this slide,
    per-slide overrides first) / pred (model-predicted label per segment) / single class."""
    # the same colour rule as read_summary, for model classes and reviewer-added segments alike
    pal = {k: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
           for k, c in ((k, CLUSTER_COLORS[k % len(CLUSTER_COLORS)]) for k in range(segedit.BACKGROUND))}
    cmap = class_map(sid) if SEG_MODE else {}
    lut = np.zeros((256, 4), np.uint8)
    if mode == "all":   # a merged group in its representative's colour
        for k in pal:
            lut[k] = (*pal[cmap.get(k, k)], 150)
    elif mode == "labels":
        lv = labels_now()
        glob, area = {int(k): lv.loc[k, "label"] for k in lv.index}, _area(sid)
        for r, mem in groups_of(cmap).items():
            v = group_pick(glob, r, mem, area)
            if v:
                glob.update({k: v for k in mem})
        for k, v in glob.items():
            lut[int(k)] = (*LABEL_COLORS.get(v, (0, 0, 0)), 150)
    elif mode in ("slide", "pred"):
        if mode == "slide":
            labs = slide_labels(sid)
        else:
            from slidereview import segment_stats
            labs = {k: r["pred_label"] for r in segment_stats(sid) for k in r["classes"]}
        for k, v in labs.items():
            lut[int(k)] = (*LABEL_COLORS.get(v, (0, 0, 0)), 150)
    else:
        try:
            k = int(mode)
        except ValueError:
            raise HTTPException(404)
        k = cmap.get(k, k)
        for c in pal:   # the whole merged group of k
            if cmap.get(c, c) == k:
                lut[c] = (*pal[k], 170)
    return lut


def segmap(sid, mode):
    p = cl() / "labels" / f"{sid}.png"
    if not p.exists():
        raise HTTPException(404)
    lab = label_map(sid)
    rgba = _mode_colors(mode, meta()["k"], sid)[lab]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "png")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def subtype_call(area):
    """{label: mm2} -> monophasic / biphasic (epithelial >= BIPHASIC_MIN_EPI of SS area), PD flag (round cell share)."""
    t = float(sum(area.get(c, 0) for c in SS_COMPONENTS))
    if t <= 0:
        return dict(call="no SS segments", tumor_mm2=0.0)
    g, c, r = (float(area.get(x, 0)) / t for x in ("ss_glandular", "ss_corded", "ss_round_cell"))
    call = ("biphasic" if g + c >= BIPHASIC_MIN_EPI else "monophasic") + (" + poorly differentiated" if r >= PD_MIN_ROUND else "")
    return dict(call=call, tumor_mm2=round(t, 3), epithelial=round(g + c, 3), glandular=round(g, 3), corded=round(c, 3),
                round_cell=round(r, 3))


def derived_subtypes():
    """Per slide: SS component areas from reviewed segment labels (per-slide labels override class labels)."""
    if not SEG_MODE or not (cl() / "area_mm2.csv").exists():
        return {}
    A = pd.read_csv(cl() / "area_mm2.csv", index_col=0)
    glob, over = effective_labels()
    edited = edited_slides()
    out = {}
    for sid in A.index:
        labs = slide_labels(sid, glob, over)
        ka = _area(sid) if sid in edited else {int(k): float(v) for k, v in A.loc[sid].items()}
        area = {}
        for k, v in labs.items():
            area[v] = area.get(v, 0) + ka.get(int(k), 0)
        out[sid] = dict(**subtype_call(area), labelled_frac=round(
            sum(ka.get(int(k), 0) for k in labs) / max(sum(ka.values()), 1e-9), 3))
    return out


@router.get("/api/seg/subtypes")
def subtypes():
    return {"biphasic_min_epithelial": BIPHASIC_MIN_EPI, "pd_min_round_cell": PD_MIN_ROUND, "slides": derived_subtypes()}


@router.get("/api/export/segments.geojson")
def export_segments(slide: str):
    """Segment polygons of one slide with the label of each segment on that slide (QuPath-compatible)."""
    p = cl() / "geojson" / f"{slide}.geojson"
    if not SEG_MODE or not p.exists():
        raise HTTPException(404)
    if edit_state(slide)[0]:
        g = {"type": "FeatureCollection", "features": segedit.geojson_features(label_map(slide), seg_meta(slide), slide)}
    else:
        g = json.load(open(p))
    labs = slide_labels(slide)
    for f in g["features"]:
        k = f["properties"]["segment"]
        name = labs.get(k, f"segment {k}")
        f["properties"]["classification"] = {"name": name, "color": list(LABEL_COLORS.get(name, (200, 200, 200)))}
    return Response(json.dumps(g), media_type="application/geo+json",
                    headers={"Content-Disposition": f"attachment; filename={slide}_segments.geojson"})
