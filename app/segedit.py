"""Reviewer edits of a slide's segmentation. The model's label map is never changed: each edit is a logged
operation in level-0 slide coordinates, and the current map is the model's map with the active edits replayed.

  paint    fill polygons with a segment (segment 255 = background, i.e. erase)
  stroke   the same along a brush path of a given width
  reshape  move the outline of one piece of a segment: area gained is taken from whatever was there, area given up
           goes to the nearest other segment or background, so a shared boundary moves without gaps or overlaps
"""
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

BACKGROUND = 255
OUTLINE_EPS = 2.0      # label pixels of outline simplification for the editable lines
GEOJSON_EPS = 1.0      # as in s2_segment.py
MIN_PIECE_PX = 16      # pieces smaller than this get no editable outline
NEIGHBOUR_PAD = 32     # label pixels around a reshape searched for the segment that takes over vacated area
RESHAPE_MARGIN = int(OUTLINE_EPS) + 1


def _px(points, m):
    return (np.asarray(points, float) - (m["x0"], m["y0"])) / m["px_l0"] - 0.5


def _box(rings, shape, pad=2):
    """Crop (y0, y1, x0, x1) around pixel-coordinate rings, or None if they miss the map."""
    p = np.concatenate(rings)
    x0, y0 = np.floor(p.min(0)).astype(int) - pad
    x1, y1 = np.ceil(p.max(0)).astype(int) + pad + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
    return (y0, y1, x0, x1) if x0 < x1 and y0 < y1 else None


def _blank(box):
    im = Image.new("1", (box[3] - box[2], box[1] - box[0]))
    return im, ImageDraw.Draw(im), np.array([box[2], box[0]], float)


def _polygons_mask(rings, box):
    im, d, o = _blank(box)
    for r in rings:
        d.polygon([tuple(xy) for xy in r - o], fill=1)
    return np.asarray(im, bool)


def _stroke_mask(path, width, box):
    im, d, o = _blank(box)
    pts = [tuple(xy) for xy in path - o]
    if len(pts) > 1:
        d.line(pts, fill=1, width=max(1, round(width)))
    r = width / 2
    for x, y in pts:   # round joints and caps
        d.ellipse([x - r, y - r, x + r, y + r], fill=1)
    return np.asarray(im, bool)


def apply_edit(lab, m, action, segment, g):
    """Apply one edit to the label map in place."""
    if action == "paint":
        rings = [_px(r, m) for r in g["polygons"]]
        box = _box(rings, lab.shape)
        if box:
            lab[box[0]:box[1], box[2]:box[3]][_polygons_mask(rings, box)] = segment
    elif action == "stroke":
        path, w = _px(g["points"], m), g["width"] / m["px_l0"]
        box = _box([path], lab.shape, pad=int(w / 2) + 2)
        if box:
            lab[box[0]:box[1], box[2]:box[3]][_stroke_mask(path, w, box)] = segment
    elif action == "reshape":
        old, new = _px(g["old"], m), _px(g["new"], m)
        box = _box([old, new], lab.shape, pad=NEIGHBOUR_PAD)
        if box:
            _reshape(lab[box[0]:box[1], box[2]:box[3]], _polygons_mask([old], box), _polygons_mask([new], box),
                     segment, g["classes"])


def _reshape(sub, old, new, segment, classes):
    # The outlines are simplified, so a segment's real edge can lie just either side of them. Without this margin the
    # pixels between the two would stay behind as hairlines where the line used to run.
    near = np.ones((2 * RESHAPE_MARGIN + 1,) * 2, bool)
    vacated = ndi.binary_dilation(old & ~new, near) & ~new & np.isin(sub, classes)
    sub[ndi.binary_dilation(new & ~old, near) & new] = segment
    if not vacated.any():
        return
    other = ~np.isin(sub, [*classes, segment])
    if not other.any():
        sub[vacated] = BACKGROUND
        return
    iy, ix = ndi.distance_transform_edt(~other, return_distances=False, return_indices=True)
    sub[vacated] = sub[iy[vacated], ix[vacated]]


def replay(base, m, edits, start=None):
    """Label map after the edits, continuing from `start` (a map with an earlier part of them applied) if given."""
    lab = (start if start is not None else base).copy()
    for e in edits:
        apply_edit(lab, m, e["action"], e["segment"], json.loads(e["geometry"]))
    return lab


def areas_mm2(lab, m):
    n = np.bincount(lab[lab != BACKGROUND].ravel(), minlength=1)
    return {int(k): float(v) * m["out_um"] ** 2 / 1e6 for k, v in enumerate(n) if v}


def _ring(c, m):
    return [[float(m["x0"] + (u + 0.5) * m["px_l0"]), float(m["y0"] + (v + 0.5) * m["px_l0"])] for u, v in c]


def outlines(lab, m, cmap):
    """Editable outline of each piece of each segment, merged classes as one: [{segment, ring}] in level-0 px."""
    lut = np.arange(256, dtype=np.uint8)
    for k, r in cmap.items():
        lut[k] = r
    merged = lut[lab]
    out = []
    for seg in np.unique(merged):
        if seg == BACKGROUND:
            continue
        cs, _ = cv2.findContours((merged == seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < MIN_PIECE_PX:
                continue
            c = cv2.approxPolyDP(c, OUTLINE_EPS, True)[:, 0, :]
            if len(c) >= 3:
                out.append(dict(segment=int(seg), ring=[[round(x), round(y)] for x, y in _ring(c, m)]))
    return out


def geojson_features(lab, m, sid):
    """Polygons with holes per class, in the format s2_segment.py writes."""
    feats = []
    for k in np.unique(lab):
        if k == BACKGROUND:
            continue
        cs, hier = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polys = {}
        for i, c in enumerate(cs):
            c = cv2.approxPolyDP(c, GEOJSON_EPS, True)[:, 0, :]
            if len(c) < 3:
                continue
            ring = _ring(c, m)
            ring.append(ring[0])
            parent = hier[0][i][3]
            polys.setdefault(i if parent < 0 else parent, []).append(ring)
        for rings in polys.values():
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings},
                          "properties": {"objectType": "annotation", "segment": int(k), "slide_id": sid}})
    return feats
