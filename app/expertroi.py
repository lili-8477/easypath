"""ROIs picked by experts: an outline drawn on a slide and what it shows. Independent of the segmentation run.

Each reviewer has a workspace of their own: the app lists and changes only the ROIs drawn under the reviewer's initials, so
experts annotate independently. Initials are the only identity here, so this keeps people apart, it does not secure anything.

Append-only, like the other review tables: every change adds a row to expert_roi_events holding the full label set and
comment, and the latest row of a ROI is its current state. Deleting is one more row, so it can be restored."""
import io
import json
import math
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from ann_db import connect

NORMAL = "Normal tissue"
ROI_LABELS = [
    dict(key="glandular", name="Glandular", color="#ffd166"),
    dict(key="spindle", name="Spindle cells", color="#4cc9f0"),
    dict(key="poorly_differentiated", name="Poorly differentiated (round cells)", short="Poorly differentiated", color="#ff6b9d"),
    dict(key="myxoid", name="Myxoid", color="#b79cff"),
    dict(key="osteoid", name="Osteoid (bone forming)", short="Osteoid", color="#ff9f45"),
    dict(key="collagenous", name="Collagenous", color="#7ee081"),
    dict(key="normal_vessel", name="Vessels", short="Normal: vessels", group=NORMAL, color="#e8edf2"),
    dict(key="normal_muscle", name="Muscle", short="Normal: muscle", group=NORMAL, color="#e8edf2"),
    dict(key="normal_nerve", name="Nerve", short="Normal: nerve", group=NORMAL, color="#e8edf2"),
    dict(key="normal_fat", name="Fat", short="Normal: fat", group=NORMAL, color="#e8edf2"),
]
LABEL_KEYS = [d["key"] for d in ROI_LABELS]
LABEL_TEXT = {d["key"]: d.get("short", d["name"]) for d in ROI_LABELS}
MIN_SIDE = 32        # level-0 pixels; anything smaller is a slip of the mouse
MAX_POINTS = 20000
MAX_COMMENT = 4000
THUMB = 320
CSV_COLUMNS = ["roi_id", "slide_id", "patient", "cohort", "roi_number", "shape", "labels", "label_keys", "comment", "x", "y", "width",
               "height", "width_um", "height_um", "area_mm2", "drawn_by", "drawn_at", "last_ts", "outline"]


class NewRoi(BaseModel):
    reviewer: str
    slide_id: str
    shape: Literal["polygon", "square", "oval"]
    ring: List[List[float]]
    labels: List[str]
    comment: Optional[str] = Field("", max_length=MAX_COMMENT)


class RoiChange(BaseModel):
    reviewer: str
    action: Literal["annotate", "delete", "restore"]
    labels: List[str] = []
    comment: Optional[str] = Field("", max_length=MAX_COMMENT)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _labels(keys):
    """In the order of the list, whatever order they were clicked in."""
    if not keys or any(k not in LABEL_KEYS for k in keys):
        raise HTTPException(400, "choose at least one label from the list")
    return [k for k in LABEL_KEYS if k in keys]


def _who(name):
    who = name.strip()
    if not who:
        raise HTTPException(400, "reviewer required")
    return who


def _owner(name):
    """Workspaces do not depend on how the initials were typed: LL and ll are one person."""
    return name.strip().casefold()


def _ring_area(ring):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1]))) / 2


def _bbox(ring):
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return dict(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


def _fold(rows):
    """Current state per ROI from its events, oldest first. Deleted ones are kept here, to be restored."""
    out, seen = {}, {}
    for r in rows:
        if r["action"] == "create":
            key = (r["slide_id"], _owner(r["reviewer"]))
            seen[key] = seen.get(key, 0) + 1   # per slide and workspace; counts deleted ROIs too, so a number is never reused
            out[r["roi_id"]] = dict(id=r["roi_id"], slide_id=r["slide_id"], n=seen[key], shape=r["shape"],
                                    ring=json.loads(r["ring"]), created_by=r["reviewer"], created_at=r["ts"], deleted=False)
        d = out.get(r["roi_id"])
        if d is None:
            continue
        d.update(reviewer=r["reviewer"], ts=r["ts"], deleted=r["action"] == "delete")
        if r["action"] != "delete":
            d.update(labels=json.loads(r["labels"]), comment=r["comment"] or "")
    return out


def _load(con):
    return _fold(con.execute("SELECT * FROM expert_roi_events ORDER BY id").fetchall())


def build_router(man, open_slide, patient_label):
    """man: the slide manifest; open_slide(sid): an OpenSlide; patient_label(sid): the name shown for its patient."""
    router = APIRouter()

    def record(d):
        m = man.loc[d["slide_id"]] if d["slide_id"] in man.index else None
        b, mpp = _bbox(d["ring"]), float(m.mpp) if m is not None else float("nan")
        known = math.isfinite(mpp)
        return dict(d, bbox=b, patient=str(patient_label(d["slide_id"])) if m is not None else d["slide_id"],
                    cohort=m.cohort if m is not None else "",
                    width_um=round(b["w"] * mpp, 1) if known else None, height_um=round(b["h"] * mpp, 1) if known else None,
                    area_mm2=round(_ring_area(d["ring"]) * mpp ** 2 / 1e6, 4) if known else None)

    def current(reviewer=None, slide=None):
        """reviewer: one workspace; None: everyone's, for the exports that compare reviewers."""
        con = connect()
        rois = _load(con)
        con.close()
        return [record(d) for d in rois.values() if not d["deleted"] and (not slide or d["slide_id"] == slide)
                and (reviewer is None or _owner(d["created_by"]) == _owner(reviewer))]

    @router.get("/api/expert_rois")
    def list_rois(reviewer: str = ""):
        """The ROIs of one workspace. Without initials there is no workspace, so none."""
        return dict(labels=ROI_LABELS, rois=current(reviewer) if reviewer.strip() else [])

    @router.post("/api/expert_rois")
    def create(body: NewRoi):
        who, labels = _who(body.reviewer), _labels(body.labels)
        if body.slide_id not in man.index:
            raise HTTPException(404, "unknown slide")
        n = len(body.ring)
        fine = 3 <= n <= MAX_POINTS and (body.shape != "square" or n == 4) and all(
            len(p) == 2 and all(math.isfinite(v) for v in p) for p in body.ring)
        if not fine:
            raise HTTPException(400, "bad outline")
        W, H = open_slide(body.slide_id).dimensions
        ring = [[int(round(min(max(x, 0), W))), int(round(min(max(y, 0), H)))] for x, y in body.ring]
        b = _bbox(ring)
        if b["w"] < MIN_SIDE or b["h"] < MIN_SIDE or _ring_area(ring) <= 0:
            raise HTTPException(400, "the outline is too small")
        rid = uuid.uuid4().hex[:12]
        con = connect()
        con.execute("""INSERT INTO expert_roi_events (roi_id, slide_id, reviewer, ts, action, shape, ring, labels, comment)
                       VALUES (?,?,?,?,?,?,?,?,?)""", (rid, body.slide_id, who, _now(), "create", body.shape, json.dumps(ring),
                                                       json.dumps(labels), (body.comment or "").strip()))
        con.commit()
        d = _load(con)[rid]
        con.close()
        return record(d)

    @router.post("/api/expert_rois/{rid}")
    def change(rid: str, body: RoiChange):
        who = _who(body.reviewer)
        con = connect()
        try:
            d = _load(con).get(rid)
            if d is None:
                raise HTTPException(404, "unknown ROI")
            if _owner(d["created_by"]) != _owner(who):
                raise HTTPException(403, "this ROI is in another reviewer's workspace")
            if d["deleted"] != (body.action == "restore"):
                raise HTTPException(409, "this ROI has been deleted" if d["deleted"] else "this ROI is not deleted")
            labels, comment = d["labels"], d["comment"]
            if body.action == "annotate":
                labels, comment = _labels(body.labels), (body.comment or "").strip()
            con.execute("""INSERT INTO expert_roi_events (roi_id, slide_id, reviewer, ts, action, labels, comment)
                           VALUES (?,?,?,?,?,?,?)""", (rid, d["slide_id"], who, _now(), body.action, json.dumps(labels), comment))
            con.commit()
        finally:
            con.close()
        return {"ok": True}

    @router.get("/api/expert_rois/{rid}/thumb.jpg")
    def thumb(rid: str):
        con = connect()
        d = _load(con).get(rid)
        con.close()
        if d is None:
            raise HTTPException(404, "unknown ROI")
        b, s = _bbox(d["ring"]), open_slide(d["slide_id"])
        level = s.get_best_level_for_downsample(max(max(b["w"], b["h"]) / THUMB, 1))
        f = s.level_downsamples[level]
        size = (max(1, math.ceil(b["w"] / f)), max(1, math.ceil(b["h"] / f)))
        im = s.read_region((b["x"], b["y"]), level, size).convert("RGB")
        im.thumbnail((THUMB, THUMB))
        if d["shape"] != "square":   # the crop is the bounding box, so show which part of it was outlined
            sx, sy = im.width / b["w"], im.height / b["h"]
            pts = [((x - b["x"]) * sx, (y - b["y"]) * sy) for x, y in d["ring"]]
            draw = ImageDraw.Draw(im)
            draw.line(pts + pts[:1], fill=(0, 0, 0), width=4)
            draw.line(pts + pts[:1], fill=(255, 255, 255), width=2)
        buf = io.BytesIO()
        im.save(buf, "jpeg", quality=85)
        # an outline never changes, so the image of a ROI id does not either
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    # ------------------------------------------------------------ exports
    @router.get("/api/export/expert_rois.csv")
    def export_csv(reviewer: Optional[str] = None):
        """One workspace with ?reviewer=, otherwise every reviewer's ROIs, to compare them."""
        rows = [dict(roi_id=r["id"], slide_id=r["slide_id"], patient=r["patient"], cohort=r["cohort"], roi_number=r["n"], shape=r["shape"],
                     labels="; ".join(LABEL_TEXT[k] for k in r["labels"]), label_keys=";".join(r["labels"]), comment=r["comment"],
                     x=r["bbox"]["x"], y=r["bbox"]["y"], width=r["bbox"]["w"], height=r["bbox"]["h"],
                     width_um=r["width_um"], height_um=r["height_um"], area_mm2=r["area_mm2"], drawn_by=r["created_by"], drawn_at=r["created_at"],
                     last_ts=r["ts"], outline=json.dumps(r["ring"])) for r in current(reviewer)]
        return Response(pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=hss_expert_rois.csv"})

    @router.get("/api/export/expert_rois.geojson")
    def export_geojson(slide: Optional[str] = None, reviewer: Optional[str] = None):
        """Current ROIs as QuPath-compatible GeoJSON (level-0 pixel coordinates)."""
        feats = [{"type": "Feature", "id": r["id"],
                  "geometry": {"type": "Polygon", "coordinates": [r["ring"] + r["ring"][:1]]},
                  "properties": {"objectType": "annotation", "classification": {"name": " + ".join(LABEL_TEXT[k] for k in r["labels"])},
                                 "slide_id": r["slide_id"], "labels": r["labels"], "comment": r["comment"], "roi_number": r["n"],
                                 "drawn_by": r["created_by"], "ts": r["ts"]}} for r in current(reviewer, slide)]
        return JSONResponse({"type": "FeatureCollection", "features": feats},
                            headers={"Content-Disposition": f"attachment; filename={slide or 'all_slides'}_expert_rois.geojson"})

    @router.get("/api/export/expert_rois_audit.csv")
    def export_audit():
        con = connect()
        df = pd.read_sql_query("SELECT * FROM expert_roi_events ORDER BY id", con)
        con.close()
        return Response(df.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=hss_expert_rois_audit.csv"})

    return router
