"""Patch annotation for model training: a fixed grid over each slide, and every pixel of a patch annotated by hand.

Grid: patches of 512 x 512 pixels at 20x. The slides are scanned at 40x, so that is PATCH_SIZE = 1024 level-0 pixels on the
slide, counted from its top left corner, with masks of MASK_SIZE = 512. Only whole patches are in the grid, so a strip narrower
than a patch at the right and bottom edges is left out. A patch is named by the level-0 pixel of its corner.

A patch holds regions, each a mask with one or more labels. Regions may overlap, so one pixel can have several labels. A patch
is complete when every one of its pixels is in a region. The server works this out from the masks itself.

Each reviewer has a workspace of their own, as with the expert ROIs: initials are the only identity, so this keeps people
apart, it does not secure anything.

Append-only, like the other review tables: every save adds a row to patch_events holding the whole patch, and the latest row
of a patch in a workspace is its current state. Removing every region is one more row (status "empty").

A reviewer can also type a class of their own. It is stored as "custom:<text>" beside the keys of the fixed list, and a workspace's
classes are the ones in its patches now, so nothing else has to be kept for them. Free text never becomes a path: its mask file gets
a safe name of its own, and labels.json says which class that is.

Masks are run-length encoded in row order: counts[0] pixels of 0, then counts[1] pixels of 1, and so on, as in COCO.

Experts choose which patches to annotate, so how they work is part of the data: patch_activity logs every operation in the
order it happened, and each stay on a patch with its time. Time is kept twice: elapsed is the clock, active leaves out time with
the browser tab hidden or with no input for 2 minutes."""
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from ann_db import connect
from expertroi import ROI_LABELS

PATCH_SIZE = 1024    # on the slide, in level-0 pixels (0.2525 um/pixel, 40x)
MASK_SIZE = 512      # the side of a patch at training magnification
MAGNIFICATION = "20×"
DOWNSAMPLE = PATCH_SIZE // MASK_SIZE
MAX_REGIONS = 64
MAX_RUNS = 400000    # in one patch; far beyond anything drawn by hand
MAX_COMMENT = 4000

# The ROI labels, plus what a patch needs to be covered completely: not every pixel is one of the tissues on the list.
# The normal tissues have one colour in the ROI list. Here they fill a mask, so each needs its own.
COLORS = dict(normal_vessel="#ff4d4d", normal_muscle="#c8744a", normal_nerve="#2dd4bf", normal_fat="#f5f0a0")
PATCH_LABELS = [dict(d, color=COLORS.get(d["key"], d["color"])) for d in ROI_LABELS] + [
    dict(key="background", name="Background (no tissue)", short="Background", group="Everything else", color="#8d99ae"),
    dict(key="other", name="Other / can't tell", short="Other", group="Everything else", color="#f4f4f6"),
]
LABEL_KEYS = [d["key"] for d in PATCH_LABELS]
CUSTOM = "custom:"
MAX_CUSTOM = 40          # characters in a class typed by a reviewer
MAX_REGION_LABELS = 16
FIELDS = ["reviewer", "slide_id", "patient", "cohort", "x", "y", "size_l0", "mask_size", "downsample", "mpp_l0", "mpp_patch", "status",
          "coverage", "labels", "comment", "ts"]
MAX_EVENTS = 1000
MAX_DETAIL = 2000
OPERATIONS = ["brush", "erase", "wand", "polygon", "freehand", "label", "label_rest", "select_rest", "region_open", "region_close", "region_delete", "discard",
              "undo", "redo", "comment", "hide", "tool", "zoom_to_patch", "view"]
ANNOTATING = {"brush", "erase", "wand", "polygon", "freehand", "label", "label_rest", "select_rest", "region_delete", "discard", "undo", "redo"}
LOG_FIELDS = ["reviewer", "session", "step", "time", "operation", "slide_id", "x", "y", "size_l0", "pick_order", "visit_order", "visit_of_patch",
              "annotation_order", "label", "coverage_after", "via", "stay_active_s", "stay_elapsed_s", "patch_active_s", "patch_elapsed_s", "detail"]
VISIT_FIELDS = ["reviewer", "session", "visit_order", "pick_order", "visit_of_patch", "slide_id", "x", "y", "size_l0", "via", "started_at",
                "ended_at", "ended_by", "elapsed_s", "active_s", "coverage_start", "coverage_end", "status_end"] + [f"n_{k}" for k in OPERATIONS]


class Region(BaseModel):
    labels: List[str]
    counts: List[int]


class PatchSave(BaseModel):
    reviewer: str
    regions: List[Region] = []
    comment: Optional[str] = Field("", max_length=MAX_COMMENT)


class ActivityEvent(BaseModel):
    seq: int
    t: int                      # ms since 1970, by the reviewer's clock
    reviewer: str
    type: str = Field(pattern=r"^[a-z_]{1,24}$")
    visit: Optional[str] = Field(None, max_length=64)
    sid: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    detail: Dict = {}


class ActivityBatch(BaseModel):
    session: str = Field(min_length=1, max_length=64)
    events: List[ActivityEvent] = Field(max_length=MAX_EVENTS)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds")


def _attachment(base, ext, reviewer=None):
    who = re.sub(r"[^0-9A-Za-z._-]+", "_", reviewer.strip()) if reviewer is not None else "all"
    return {"Content-Disposition": f"attachment; filename=hss_{base}_{who}.{ext}"}


def _owner(name):
    """Workspaces do not depend on how the initials were typed: LL and ll are one person."""
    return name.strip().casefold()


def decode(counts, size=MASK_SIZE):
    """A boolean size x size mask. Raises ValueError if the runs do not describe exactly that many pixels."""
    a = np.asarray(counts, dtype=np.int64)
    if a.size == 0 or (a < 0).any() or (a > size * size).any() or a.sum() != size * size:
        raise ValueError("bad mask")
    return np.repeat(np.arange(a.size) % 2 == 1, a).reshape(size, size)


def encode(mask):
    flat = mask.ravel()
    runs = np.diff(np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]).tolist()
    return [0] + runs if flat[0] else runs


def custom_key(label):
    """A typed class in its one stored form, or None if nothing is left of it. ; separates labels in the CSVs, so it cannot be in one."""
    text = unicodedata.normalize("NFC", label[len(CUSTOM):])
    text = "".join(c for c in " ".join(text.split()) if unicodedata.category(c)[0] != "C").replace(";", ",")
    text = text[:MAX_CUSTOM].strip()
    return CUSTOM + text if text else None


def _label(label):
    return label if label in LABEL_KEYS else custom_key(label) if label.startswith(CUSTOM) else None


def _ordered(labels):
    return [k for k in LABEL_KEYS if k in labels] + sorted((k for k in labels if k not in LABEL_KEYS), key=str.casefold)


def mask_name(key):
    """The file of a label's mask. Spelled differently only by case, a typed class is one file."""
    if key in LABEL_KEYS:
        return key
    text = key[len(CUSTOM):].casefold()
    slug = re.sub(r"[^0-9a-z]+", "_", text).strip("_")[:24]
    return "_".join(filter(None, ["custom", slug, hashlib.sha1(text.encode()).hexdigest()[:8]]))


def _clean(regions):
    """Checked and in one form: labels in list order, empty regions gone, and regions with the same labels joined.
    Returns (regions as stored, covered pixels, {label: fraction of the patch}), all worked out from the masks."""
    if len(regions) > MAX_REGIONS or sum(len(r.counts) for r in regions) > MAX_RUNS:
        raise HTTPException(400, "too many regions in this patch")
    joined = {}
    for r in regions:
        labels = {_label(k) for k in r.labels}
        if not labels or None in labels or len(labels) > MAX_REGION_LABELS:
            raise HTTPException(400, "every region needs a label from the list, or one typed in")
        try:
            m = decode(r.counts)
        except (ValueError, OverflowError):
            raise HTTPException(400, "bad mask") from None
        if m.any():
            key = tuple(_ordered(labels))
            joined[key] = joined[key] | m if key in joined else m
    union = np.zeros((MASK_SIZE, MASK_SIZE), bool)
    per = {}
    for keys, m in joined.items():
        union |= m
        for k in keys:
            per[k] = per[k] | m if k in per else m
    fractions = {k: round(float(per[k].mean()), 4) for k in _ordered(per)}
    stored = [dict(labels=list(keys), counts=encode(m)) for keys, m in joined.items()]
    return stored, int(union.sum()), fractions


def _status(covered, stored):
    return "complete" if covered == MASK_SIZE * MASK_SIZE else "partial" if stored else "empty"


LATEST = """SELECT {cols} FROM patch_events WHERE size = ? AND id IN
            (SELECT MAX(id) FROM patch_events GROUP BY owner, slide_id, x, y, size) {where} ORDER BY slide_id, y, x"""
SUMMARY = "slide_id, reviewer, owner, ts, x, y, size, mask_size, status, covered, fractions, comment"


def _summary(r):
    return dict(slide_id=r["slide_id"], x=r["x"], y=r["y"], status=r["status"], coverage=r["covered"] / r["mask_size"] ** 2,
                fractions=json.loads(r["fractions"]), comment=r["comment"] or "", ts=r["ts"])


def build_router(man, open_slide, patient_label):
    """man: the slide manifest; open_slide(sid): an OpenSlide; patient_label(sid): the name shown for its patient."""
    router = APIRouter()

    def latest(cols, reviewer=None, status=None):
        where, args = "", [PATCH_SIZE]
        if reviewer is not None:
            where += " AND owner = ?"
            args.append(_owner(reviewer))
        where += " AND status IN (%s)" % ",".join("?" * len(status))
        args += status
        con = connect()
        rows = con.execute(LATEST.format(cols=cols, where=where), args).fetchall()
        con.close()
        return rows

    def check_cell(sid, x, y):
        if sid not in man.index:
            raise HTTPException(404, "unknown slide")
        W, H = open_slide(sid).dimensions
        if x < 0 or y < 0 or x % PATCH_SIZE or y % PATCH_SIZE or x + PATCH_SIZE > W or y + PATCH_SIZE > H:
            raise HTTPException(400, "not a patch of the grid")

    @router.get("/api/patches")
    def list_patches(reviewer: str = ""):
        """The patches of one workspace. Without initials there is no workspace, so none."""
        rows = latest(SUMMARY, reviewer, ["complete", "partial"]) if reviewer.strip() else []
        patches = [_summary(r) for r in rows]
        custom = _ordered({k for p in patches for k in p["fractions"] if k not in LABEL_KEYS})
        return dict(size=PATCH_SIZE, mask_size=MASK_SIZE, magnification=MAGNIFICATION, labels=PATCH_LABELS, custom=custom, patches=patches)

    @router.get("/api/patches/{sid}/{x}/{y}")
    def get_patch(sid: str, x: int, y: int, reviewer: str = ""):
        check_cell(sid, x, y)
        con = connect()
        r = con.execute("""SELECT regions, comment FROM patch_events WHERE slide_id = ? AND x = ? AND y = ? AND size = ? AND owner = ?
                           ORDER BY id DESC LIMIT 1""", (sid, x, y, PATCH_SIZE, _owner(reviewer))).fetchone() if reviewer.strip() else None
        con.close()
        return dict(regions=json.loads(r["regions"]) if r else [], comment=(r["comment"] or "") if r else "")

    @router.get("/api/patches/{sid}/{x}/{y}/image")
    def patch_image(sid: str, x: int, y: int):
        """The patch at training size, on the grid of its masks: what the wand tool reads. Lossless, so it sees the slide's own colours."""
        check_cell(sid, x, y)
        tile = open_slide(sid).read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
        flat = Image.new("RGB", tile.size, "white")   # outside the scanned area a slide file is transparent
        flat.paste(tile, mask=tile.getchannel("A"))
        buf = io.BytesIO()
        flat.resize((MASK_SIZE, MASK_SIZE), Image.BOX).save(buf, "png", compress_level=1)
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @router.post("/api/patches/{sid}/{x}/{y}")
    def save_patch(sid: str, x: int, y: int, body: PatchSave):
        who = body.reviewer.strip()
        if not who:
            raise HTTPException(400, "reviewer required")
        check_cell(sid, x, y)
        stored, covered, fractions = _clean(body.regions)
        comment, status, ts = (body.comment or "").strip(), _status(covered, stored), _now()
        con = connect()
        con.execute("""INSERT INTO patch_events (slide_id, owner, reviewer, ts, x, y, size, mask_size, status, covered, fractions, regions, comment)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sid, _owner(who), who, ts, x, y, PATCH_SIZE, MASK_SIZE, status, covered, json.dumps(fractions),
                     json.dumps(stored, separators=(",", ":")), comment))
        con.commit()
        con.close()
        return dict(slide_id=sid, x=x, y=y, status=status, coverage=covered / MASK_SIZE ** 2, fractions=fractions, comment=comment, ts=ts)

    # ------------------------------------------------------------ how the experts worked
    @router.post("/api/patch_activity")
    def log_activity(body: ActivityBatch):
        """Sent in batches by the browser. A batch that is sent again adds nothing. Sent with sendBeacon when a page closes, so
        nobody is waiting for the answer."""
        ts, rows = _now(), []
        for e in body.events:
            who = e.reviewer.strip()
            detail = json.dumps(e.detail, separators=(",", ":"))
            if who and len(detail) <= MAX_DETAIL:
                rows.append((body.session, e.seq, _owner(who), who, e.t, ts, e.visit, e.type, e.sid if e.sid in man.index else None, e.x, e.y, detail))
        con = connect()
        con.executemany("""INSERT OR IGNORE INTO patch_activity (session, seq, owner, reviewer, client_ms, received_ts, visit, type, slide_id, x, y, detail)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        con.commit()
        con.close()
        return dict(ok=True, stored=len(rows))

    def activity(reviewer=None):
        con = connect()
        where, args = ("WHERE owner = ?", [_owner(reviewer)]) if reviewer is not None else ("", [])
        rows = con.execute(f"SELECT * FROM patch_activity {where} ORDER BY owner, client_ms, session, seq", args).fetchall()
        con.close()
        return rows

    def visits(reviewer=None):
        """One row per stay on a patch, in the order they happened within each workspace. pick_order counts the patches a
        reviewer went to for the first time, so it is empty when they came back to one."""
        grouped, order = {}, []
        for r in activity(reviewer):
            if r["visit"]:
                if r["visit"] not in grouped:
                    grouped[r["visit"]] = []
                    order.append(r["visit"])
                grouped[r["visit"]].append(r)
        out, seen_visits, seen_cells, picks = [], Counter(), Counter(), Counter()
        for vid in order:
            ev = grouped[vid]
            first, details = ev[0], [json.loads(e["detail"]) for e in ev]
            owner, cell = first["owner"], (first["owner"], first["slide_id"], first["x"], first["y"])
            seen_visits[owner] += 1
            seen_cells[cell] += 1
            if seen_cells[cell] == 1:
                picks[owner] += 1
            kinds = Counter(e["type"] for e in ev)
            timed = [d for d in details if "active_ms" in d]
            closed = [d for e, d in zip(ev, details) if e["type"] == "close"]
            opened = next((d for e, d in zip(ev, details) if e["type"] == "open"), {})
            ended = closed[-1] if closed else {}
            covers = [d["covered"] for d in details if "covered" in d]
            out.append(dict(visit_id=vid, owner=owner, reviewer=first["reviewer"], session=first["session"], visit_order=seen_visits[owner],
                            pick_order=picks[owner] if seen_cells[cell] == 1 else "", visit_of_patch=seen_cells[cell], slide_id=first["slide_id"],
                            x=first["x"], y=first["y"], size_l0=PATCH_SIZE, via=opened.get("via", ""), started_at=_iso(first["client_ms"]),
                            ended_at=_iso(ev[-1]["client_ms"]), ended_by=ended.get("reason", "no close recorded"),
                            elapsed_s=round(timed[-1]["elapsed_ms"] / 1000, 1) if timed else 0, active_s=round(timed[-1]["active_ms"] / 1000, 1) if timed else 0,
                            coverage_start=opened.get("covered", ""), coverage_end=covers[-1] if covers else "", status_end=ended.get("status", ""),
                            **{f"n_{k}": kinds[k] for k in OPERATIONS}))
        return out

    @router.get("/api/export/patch_visits.csv")
    def export_visits(reviewer: Optional[str] = None):
        """One row per stay on a patch: pick order, how it was reached, stay time and operations. One workspace with ?reviewer=."""
        return Response(pd.DataFrame(visits(reviewer), columns=VISIT_FIELDS).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=hss_patch_visits.csv"})

    @router.get("/api/export/patch_activity.csv")
    def export_activity(reviewer: Optional[str] = None):
        """The operation log: every step of a reviewer in the order it happened, one row each, in one workspace with ?reviewer= or
        all of them without. On each row: step, that reviewer's running count; pick_order, this patch's place in the order they first
        went to patches, kept on return visits; annotation_order, the place of this step among the steps that changed the annotation
        of this patch, carried on across visits; stay_*, the whole stay this step belongs to; patch_*, all stays on this patch."""
        stays, picks, totals = {}, {}, {}
        for v in visits(reviewer):
            cell = (v["owner"], v["slide_id"], v["x"], v["y"])
            stays[v["visit_id"]] = v
            if v["pick_order"] != "":
                picks[cell] = v["pick_order"]
            t = totals.setdefault(cell, [0.0, 0.0])
            t[0] += v["active_s"]
            t[1] += v["elapsed_s"]
        rows, steps, done = [], Counter(), Counter()
        for r in activity(reviewer):
            if r["type"] == "tick":   # only the clock: the time it carries is in the stay columns
                continue
            d, v = json.loads(r["detail"]), stays.get(r["visit"])
            more = dict.fromkeys(("x", "y", "size_l0", "pick_order", "visit_order", "visit_of_patch", "annotation_order", "via", "stay_active_s", "stay_elapsed_s",
                                  "patch_active_s", "patch_elapsed_s"), "")   # blank outside a patch; missing values would turn the whole column into floats
            steps[r["owner"]] += 1
            if v:
                cell = (r["owner"], r["slide_id"], r["x"], r["y"])
                if r["type"] in ANNOTATING:
                    done[cell] += 1
                more = dict(x=r["x"], y=r["y"], size_l0=PATCH_SIZE, pick_order=picks.get(cell, ""), visit_order=v["visit_order"], visit_of_patch=v["visit_of_patch"],
                            annotation_order=done[cell] if r["type"] in ANNOTATING else "", via=v["via"], stay_active_s=v["active_s"], stay_elapsed_s=v["elapsed_s"],
                            patch_active_s=round(totals[cell][0], 1), patch_elapsed_s=round(totals[cell][1], 1))
            rows.append(dict(reviewer=r["reviewer"], session=r["session"], step=steps[r["owner"]], time=_iso(r["client_ms"]), operation=r["type"], slide_id=r["slide_id"],
                             label=d.get("key") or ";".join(d.get("labels", [])), coverage_after=d.get("covered", ""), detail=r["detail"], **more))
        return Response(pd.DataFrame(rows, columns=LOG_FIELDS).to_csv(index=False), media_type="text/csv", headers=_attachment("operation_log", "csv", reviewer))

    # ------------------------------------------------------------ exports
    def row(r, **more):
        known = r["slide_id"] in man.index
        mpp = float(man.loc[r["slide_id"], "mpp"]) if known else None
        return dict(reviewer=r["reviewer"], slide_id=r["slide_id"], patient=patient_label(r["slide_id"]) if known else "",
                    cohort=man.loc[r["slide_id"], "cohort"] if known else "", x=r["x"], y=r["y"], size_l0=r["size"], mask_size=r["mask_size"],
                    downsample=r["size"] / r["mask_size"], mpp_l0=mpp if known else "", mpp_patch=round(mpp * r["size"] / r["mask_size"], 4) if known else "",
                    status=r["status"], coverage=round(r["covered"] / r["mask_size"] ** 2, 6), labels=";".join(json.loads(r["fractions"])),
                    comment=r["comment"] or "", ts=r["ts"], **more)

    @router.get("/api/export/patches.csv")
    def export_csv(reviewer: Optional[str] = None):
        """One workspace with ?reviewer=, otherwise every reviewer's patches, to compare them. One column per label: its share of the
        patch. Also its place in the order the reviewer picked patches, and the time they stayed on it over all visits."""
        stays = {}
        for v in visits(reviewer):
            s = stays.setdefault((v["reviewer"].casefold(), v["slide_id"], v["x"], v["y"]), dict(pick_order="", visits=0, active_s=0.0, elapsed_s=0.0))
            s["pick_order"] = v["pick_order"] if v["pick_order"] != "" else s["pick_order"]
            s["visits"] += 1
            s["active_s"] = round(s["active_s"] + v["active_s"], 1)
            s["elapsed_s"] = round(s["elapsed_s"] + v["elapsed_s"], 1)
        rows, found = [], latest(SUMMARY, reviewer, ["complete", "partial"])
        custom = _ordered({k for r in found for k in json.loads(r["fractions"]) if k not in LABEL_KEYS})
        for r in found:
            f = json.loads(r["fractions"])
            stay = stays.get((r["owner"], r["slide_id"], r["x"], r["y"]), dict(pick_order="", visits=0, active_s="", elapsed_s=""))
            rows.append(row(r, **stay, **{f"frac_{k}": f.get(k, 0) for k in LABEL_KEYS + custom}))
        cols = FIELDS + ["pick_order", "visits", "active_s", "elapsed_s"] + [f"frac_{k}" for k in LABEL_KEYS + custom]
        return Response(pd.DataFrame(rows, columns=cols).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=hss_patches.csv"})

    @router.get("/api/export/patch_masks.zip")
    def export_masks(reviewer: Optional[str] = None, include_partial: bool = False):
        """Training masks. One 1-bit PNG of 512 x 512 per label per patch, 255 where the label applies. Regions may overlap, so these
        are multi-label: one pixel can be set in several. labels.json has the classes reviewers typed, with the file of each. Only
        complete patches unless include_partial=1. The image of a patch is
        read_region((x, y), 0, (size_l0, size_l0)) on the slide, resized to mask_size, with the numbers in manifest.csv. The slide
        files have no level at 20x, so it has to be resized from level 0."""
        buf, rows = io.BytesIO(), latest(SUMMARY + ", regions", reviewer, ["complete", "partial"] if include_partial else ["complete"])
        table = io.StringIO()
        out = csv.DictWriter(table, FIELDS + ["mask_dir"], lineterminator="\n")
        out.writeheader()
        typed = {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:   # PNGs are compressed already
            for r in rows:
                folder = f"masks/{re.sub(r'[^0-9A-Za-z._-]+', '_', r['owner'])}/{r['slide_id']}/x{r['x']}_y{r['y']}"
                per = {}
                for reg in json.loads(r["regions"]):
                    m = decode(reg["counts"], r["mask_size"])
                    for k in reg["labels"]:
                        name = mask_name(k)
                        per[name] = per[name] | m if name in per else m
                        if k not in LABEL_KEYS:
                            typed.setdefault(name, k)
                for k, m in per.items():
                    png = io.BytesIO()
                    Image.fromarray(m.astype(np.uint8) * 255).convert("1", dither=Image.NONE).save(png, "png")
                    z.writestr(f"{folder}/{k}.png", png.getvalue())
                out.writerow(row(r, mask_dir=folder))
            z.writestr("manifest.csv", table.getvalue())
            z.writestr("labels.json", json.dumps(dict(size_l0=PATCH_SIZE, mask_size=MASK_SIZE, magnification=MAGNIFICATION, labels=PATCH_LABELS,
                                                      custom_labels=[dict(key=typed[f], name=typed[f][len(CUSTOM):], file=f) for f in sorted(typed)]),
                                                 ensure_ascii=False, indent=1))
        return Response(buf.getvalue(), media_type="application/zip", headers=_attachment("complete_patches", "zip", reviewer))

    @router.get("/api/export/patches_audit.csv")
    def export_audit():
        """Every save, without the masks themselves."""
        con = connect()
        df = pd.read_sql_query("SELECT id, slide_id, reviewer, ts, x, y, size, mask_size, status, covered, fractions, comment FROM patch_events ORDER BY id", con)
        con.close()
        return Response(df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=hss_patches_audit.csv"})

    return router
