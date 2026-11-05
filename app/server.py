"""Pathologist review server: whole-slide viewer, tissue heatmaps, ROI predictions and an append-only
decision log. Run on a compute node with serve.slurm (ssh tunnel) or in the Docker image.

  uvicorn server:app --host 0.0.0.0 --port 8765      (from app/)

Access needs the token from EASYPATH_TOKEN, or else the file EASYPATH_TOKEN_FILE (default ~/.hss_annot_token,
created on first start). First visit with ?token=..., then a cookie.
"""
import io
import json
import math
import os
import secrets
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import openslide
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ann_common import (ANN, MAP_DIR, NECROSIS_LABELS, PATTERN_LABELS, RES, THUMB_DIR,  # noqa: E402
                        TUMOR_LABELS, latest_rois, slide_path)
from ann_db import connect, latest_reviews, manual_rois  # noqa: E402
from clusters import derived_subtypes, router as cluster_router, seg_meta  # noqa: E402
from expertroi import build_router as expert_roi_router  # noqa: E402
from slidereview import router as slide_router  # noqa: E402

STATIC = Path(__file__).parent / "static"
TOKEN_FILE = Path(os.environ.get("EASYPATH_TOKEN_FILE") or Path.home() / ".hss_annot_token")
TOKEN = os.environ.get("EASYPATH_TOKEN", "").strip()
if not TOKEN:
    if not TOKEN_FILE.exists():
        TOKEN_FILE.write_text(secrets.token_urlsafe(24))
        TOKEN_FILE.chmod(0o600)
    TOKEN = TOKEN_FILE.read_text().strip()

HEAT_COLORS = {"tumor": (230, 40, 110), "necrosis": (250, 170, 20), "inflammation": (40, 120, 255),
               "stroma": (60, 200, 170), "benign_epithelium": (150, 90, 230), "fat": (240, 230, 90),
               "muscle": (200, 80, 40), "blood": (180, 20, 20)}
STATE_ORDER = {"needs_review": 0, "blocked": 1, "review": 2, "prelabel": 3}

app = FastAPI(title="hSS ROI review")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(cluster_router)
app.include_router(slide_router)


@app.middleware("http")
async def require_token(request: Request, call_next):
    q = request.query_params.get("token")
    if request.cookies.get("annot_token") != TOKEN and q != TOKEN:
        return PlainTextResponse("Open the link that includes ?token=...", status_code=401)
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"   # browsers must revalidate, so a deploy reaches them at once
    if q == TOKEN:
        resp.set_cookie("annot_token", TOKEN, httponly=True, samesite="strict", max_age=30 * 86400,
                        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    return resp


# ---------------------------------------------------------------- data (reloaded when files change)
MAN = pd.read_csv(RES / "slide_manifest.csv").set_index("slide_id")
_cache = {}


def _cached(key, path, loader):
    m = path.stat().st_mtime if path and path.exists() else None
    if key not in _cache or _cache[key][0] != m:
        _cache[key] = (m, loader(path) if m else None)
    return _cache[key][1]


def roi_table():
    R = _cached("rois", latest_rois(), pd.read_csv)
    con = connect()
    mr = manual_rois(con)
    con.close()
    mr = mr[~mr.roi_id.isin(R.roi_id)].assign(source="manual", state="needs_review", uncertainty=1.0,
                                               model_version="none")
    return pd.concat([R, mr.drop(columns=["created_by", "created_at"])], ignore_index=True)


def embeddings():
    out = {}
    for name in ["roi_embeddings.npz", "manual_embeddings.npz"]:
        z = _cached(name, ANN / name, lambda p: dict(np.load(p, allow_pickle=True)))
        if z:
            out.update(dict(zip(z["roi_id"], z["emb"])))
    return out


def qc_slides():
    return _cached("qc", ANN / "qc_slides.csv", lambda p: pd.read_csv(p).set_index("slide_id"))


def reviews():
    con = connect()
    rv = latest_reviews(con).set_index("roi_id")
    con.close()
    return rv


def clean(d):
    """JSON-safe dict: NaN -> None, numpy scalars -> python."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            v = None if math.isnan(v) else float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def roi_record(r, rv):
    d = clean(r.to_dict())
    d["probs"] = {k: d.pop(k) for k in list(d) if k.startswith("p_") and ":" in k}
    d["conch_pattern"] = {k.split(":")[1]: d.pop(k) for k in list(d) if k.startswith("conch_pattern:")}
    d["review"] = clean(rv.loc[r.roi_id].to_dict()) if r.roi_id in rv.index else None
    return d


def patient_label(sid):
    m = MAN.loc[sid]
    return m.Unified_ID if isinstance(m.Unified_ID, str) else m.patient


# ---------------------------------------------------------------- slides and tiles
@lru_cache(maxsize=16)
def dz(sid):
    s = openslide.OpenSlide(slide_path(MAN.loc[sid, "file"]))
    return s, DeepZoomGenerator(s, tile_size=254, overlap=1, limit_bounds=False)


app.include_router(expert_roi_router(MAN, lambda sid: dz(sid)[0], patient_label))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/slides")
def slides():
    R, rv, qc = roi_table(), reviews(), qc_slides()
    out = []
    for sid, g in R.groupby("slide_id"):
        m = MAN.loc[sid]
        done = g.roi_id.isin(rv.index)
        q = qc.loc[sid] if sid in qc.index else None
        out.append(clean(dict(
            slide_id=sid, patient=patient_label(sid), cohort=m.cohort, histology=m.get("Histology"),
            fusion=m.get("Fusion"), n_rois=len(g), n_reviewed=int(done.sum()),
            n_needs_review=int(((g.state == "needs_review") & ~done).sum()),
            n_second=int(rv.loc[g.roi_id[done], "second_review"].fillna(0).astype(int).sum()) if done.any() else 0,
            qc_status=q.qc_status if q is not None else "unknown",
            qc_flags={k.replace("frac_flag_", ""): float(q[k]) for k in q.index if k.startswith("frac_flag_")} if q is not None else {})))
    return sorted(out, key=lambda d: (d["cohort"], str(d["patient"])))


@app.get("/api/slides/{sid}/info")
def slide_info(sid: str):
    if sid not in MAN.index:
        raise HTTPException(404)
    s, _ = dz(sid)
    W, H = s.dimensions
    mp = MAP_DIR / f"{sid}.npz"
    z = np.load(mp) if mp.exists() else None
    step = int(z["step"]) if z is not None else None
    m = MAN.loc[sid]
    return clean(dict(slide_id=sid, width=W, height=H, mpp=float(m.mpp), patient=patient_label(sid), cohort=m.cohort,
                      histology=m.get("Histology"), fusion=m.get("Fusion"), step=step,
                      grid_w=math.ceil(W / step) if step else None, grid_h=math.ceil(H / step) if step else None,
                      heat_classes=[str(c) for c in z["classes"]] if z is not None else [],
                      conch_map=(MAP_DIR / f"{sid}_conch.npz").exists(),
                      cluster_map=(MAP_DIR / f"{sid}_clusters.npz").exists() or seg_meta(sid) is not None,
                      seg=seg_meta(sid), subtype=derived_subtypes().get(sid)))


@app.get("/dzi/{sid}.dzi")
def dzi(sid: str):
    return Response(dz(sid)[1].get_dzi("jpeg"), media_type="application/xml")


@app.get("/dzi/{sid}_files/{level}/{col}_{row}.jpeg")
def tile(sid: str, level: int, col: int, row: int):
    try:
        im = dz(sid)[1].get_tile(level, (col, row))
    except ValueError:
        raise HTTPException(404)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "jpeg", quality=82)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/slides/{sid}/heatmap/{cls}.png")
def heatmap(sid: str, cls: str, src: str = "titan"):
    """src=titan: TITAN 5 x 5 neighbourhood map; src=conch: CONCH v1 per-patch map."""
    sfx = {"titan": "", "conch": "_conch"}.get(src)
    if sfx is None or not (MAP_DIR / f"{sid}{sfx}.npz").exists():
        raise HTTPException(404)
    z = np.load(MAP_DIR / f"{sid}{sfx}.npz")
    classes = [str(c) for c in z["classes"]]
    scale = json.load(open(ANN / f"tissue_map_scale{sfx}.json"))
    s, _ = dz(sid)
    step = int(z["step"])
    gw, gh = math.ceil(s.dimensions[0] / step), math.ceil(s.dimensions[1] / step)
    rgba = np.zeros((gh, gw, 4), np.uint8)
    P = z["probs"]
    if cls == "top":
        # each class scaled to its cohort range, then the strongest relative class wins
        rel = np.stack([(P[:, k] - scale[c][0]) / (scale[c][1] - scale[c][0] + 1e-9) for k, c in enumerate(classes)], 1)
        top = rel.argmax(1)
        col = np.array([HEAT_COLORS.get(c, (200, 200, 200)) for c in classes])[top]
        alpha = np.full(len(top), 170)
    else:
        if cls not in classes:
            raise HTTPException(404)
        lo, hi = scale[cls]
        v = np.clip((P[:, classes.index(cls)] - lo) / (hi - lo + 1e-9), 0, 1)
        col = np.tile(HEAT_COLORS.get(cls, (230, 40, 110)), (len(v), 1))
        alpha = (v * 230).astype(int)
    rgba[z["gy"], z["gx"], :3] = col
    rgba[z["gy"], z["gx"], 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "png")
    return Response(buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------- ROIs
@app.get("/api/slides/{sid}/rois")
def slide_rois(sid: str):
    R, rv = roi_table(), reviews()
    return [roi_record(r, rv) for _, r in R[R.slide_id == sid].iterrows()]


@app.get("/api/queue")
def queue(slide: Optional[str] = None):
    """Review order: unreviewed first, then needs_review > blocked > review > prelabel, most uncertain first."""
    R, rv = roi_table(), reviews()
    if slide:
        R = R[R.slide_id == slide]
    R = R.assign(reviewed=R.roi_id.isin(rv.index), so=R.state.map(STATE_ORDER).fillna(0))
    R = R.sort_values(["reviewed", "so", "uncertainty"], ascending=[True, True, False])
    return [clean(dict(roi_id=r.roi_id, slide_id=r.slide_id, state=r.state, uncertainty=r.uncertainty,
                       reviewed=bool(r.reviewed), decision=rv.loc[r.roi_id, "decision"] if r.reviewed else None))
            for r in R.itertuples()]


@app.get("/api/rois/{rid}/thumb.jpg")
def roi_thumb(rid: str):
    p = THUMB_DIR / f"{rid}.jpg"
    if not p.exists():
        R = roi_table().set_index("roi_id")
        if rid not in R.index:
            raise HTTPException(404)
        r = R.loc[rid]
        s, _ = dz(r.slide_id)
        s.read_region((int(r.x), int(r.y)), 0, (int(r.w), int(r.h))).convert("RGB").resize((256, 256)).save(p, quality=88)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/rois/{rid}/neighbors")
def neighbors(rid: str, k: int = 6):
    """Nearest ROIs in TITAN space: reviewed ones as precedent, or any ROI from other slides if none reviewed yet."""
    E = embeddings()
    if rid not in E:
        return {"reference": False, "items": []}
    R, rv = roi_table().set_index("roi_id"), reviews()
    ids = [i for i in E if i != rid and i in R.index and R.loc[i, "slide_id"] != R.loc[rid, "slide_id"]]
    ref = [i for i in ids if i in rv.index and rv.loc[i, "decision"] != "second_review"]
    use = ref if len(ref) >= 3 else ids
    sims = np.stack([E[i] for i in use]) @ E[rid]
    out = []
    for j in np.argsort(-sims)[:k]:
        i = use[j]
        rec = dict(roi_id=i, slide_id=R.loc[i, "slide_id"], patient=patient_label(R.loc[i, "slide_id"]), sim=float(sims[j]))
        if i in rv.index:
            rec.update(tumor=rv.loc[i, "tumor"], pattern=rv.loc[i, "pattern"], necrosis=rv.loc[i, "necrosis"],
                       artifact=rv.loc[i, "artifact"], reviewer=rv.loc[i, "reviewer"])
        else:
            rec.update(pred_tumor=R.loc[i, "pred_tumor"], pred_pattern=R.loc[i, "pred_pattern"])
        out.append(clean(rec))
    return {"reference": use is ref, "items": out}


@app.get("/api/rois/{rid}/history")
def history(rid: str):
    con = connect()
    rows = [dict(r) for r in con.execute("SELECT * FROM reviews WHERE roi_id=? ORDER BY id DESC", (rid,))]
    con.close()
    return rows


class Review(BaseModel):
    reviewer: str
    decision: str
    tumor: Optional[str] = None
    pattern: Optional[str] = None
    necrosis: Optional[str] = None
    comment: Optional[str] = ""


@app.post("/api/rois/{rid}/review")
def review(rid: str, body: Review):
    R = roi_table().set_index("roi_id")
    if rid not in R.index:
        raise HTTPException(404)
    if not body.reviewer.strip():
        raise HTTPException(400, "reviewer required")
    r = R.loc[rid]
    pred = lambda c: r.get(c) if isinstance(r.get(c), str) else None  # noqa: E731
    rv = reviews()
    cur = rv.loc[rid] if rid in rv.index else None
    tumor, pattern, necrosis, artifact, second = body.tumor, body.pattern, body.necrosis, 0, 0
    if body.decision == "accept":
        tumor, pattern, necrosis = pred("pred_tumor"), pred("pred_pattern"), pred("pred_necrosis")
    elif body.decision == "not_tumor":
        tumor, pattern = "non_tumor", None
        necrosis = necrosis or pred("pred_necrosis")
    elif body.decision == "artifact":
        tumor = pattern = necrosis = None
        artifact = 1
    elif body.decision == "second_review":
        second = 1
        if cur is not None and not any([tumor, pattern, necrosis]):
            tumor, pattern, necrosis, artifact = cur.tumor, cur.pattern, cur.necrosis, int(cur.artifact or 0)
    elif body.decision != "change":
        raise HTTPException(400, "unknown decision")
    if tumor != "tumor":
        pattern = None
    for v, vocab in [(tumor, TUMOR_LABELS), (pattern, PATTERN_LABELS), (necrosis, NECROSIS_LABELS)]:
        if v is not None and v not in vocab:
            raise HTTPException(400, f"bad label {v}")
    snap = {k: v for k, v in clean(r.to_dict()).items() if k.startswith(("pred_", "conf_", "p_")) or k in ("state", "confidence")}
    con = connect()
    con.execute("""INSERT INTO reviews (roi_id, slide_id, reviewer, ts, decision, tumor, pattern, necrosis, artifact,
                   second_review, comment, model_version, prediction, geometry) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, r.slide_id, body.reviewer.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 body.decision, tumor, pattern, necrosis, artifact, second, body.comment or "",
                 r.get("model_version"), json.dumps(snap),
                 json.dumps(dict(x=int(r.x), y=int(r.y), w=int(r.w), h=int(r.h)))))
    con.commit()
    con.close()
    return {"ok": True}


class NewRoi(BaseModel):
    reviewer: str
    x: int
    y: int
    w: int
    h: int


@app.post("/api/slides/{sid}/rois")
def add_roi(sid: str, body: NewRoi):
    if sid not in MAN.index or body.w < 64 or body.h < 64:
        raise HTTPException(400)
    rid = f"{sid}__m{uuid.uuid4().hex[:8]}"
    con = connect()
    con.execute("INSERT INTO manual_rois VALUES (?,?,?,?,?,?,?,?)",
                (rid, sid, body.x, body.y, body.w, body.h, body.reviewer.strip(),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return {"roi_id": rid}


# ---------------------------------------------------------------- exports
@app.get("/api/export/reviews.csv")
def export_reviews():
    con = connect()
    df = pd.read_sql_query("SELECT * FROM reviews ORDER BY id", con)
    con.close()
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=hss_roi_reviews_audit.csv"})


@app.get("/api/export/annotations.geojson")
def export_geojson(slide: Optional[str] = None):
    """Current decisions as QuPath-compatible GeoJSON (level-0 pixel coordinates)."""
    R, rv = roi_table(), reviews()
    R = R[R.roi_id.isin(rv.index)]
    if slide:
        R = R[R.slide_id == slide]
    feats = []
    for r in R.itertuples():
        d = rv.loc[r.roi_id]
        name = "Artifact" if d.artifact == 1 else ("Tumor" if d.tumor == "tumor" else "Non-tumor")
        x, y, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
        feats.append({"type": "Feature", "id": r.roi_id,
                      "geometry": {"type": "Polygon", "coordinates": [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]]},
                      "properties": {"objectType": "annotation", "classification": {"name": name},
                                     "slide_id": r.slide_id, **clean(dict(pattern=d.pattern, necrosis=d.necrosis,
                                     second_review=d.second_review, reviewer=d.reviewer, ts=d.ts, decision=d.decision))}})
    return JSONResponse({"type": "FeatureCollection", "features": feats},
                        headers={"Content-Disposition": f"attachment; filename={slide or 'all_slides'}_annotations.geojson"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8765)))
