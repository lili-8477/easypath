"""Shared paths, label sets and helpers for the annotation pipeline and the review app.

Data locations come from the environment, so the same code runs on the cluster and on a deployed host:
  EASYPATH_RESULTS   results folder (slide_manifest.csv, annot/, titan/patch_raw/)
  EASYPATH_SLIDE_MAP optional "old_prefix=new_prefix" rewrite for the slide paths in slide_manifest.csv
Without EASYPATH_RESULTS the project's scripts/common.py (one folder up) is used.
"""
import os
import sys
from pathlib import Path

if os.environ.get("EASYPATH_RESULTS"):
    RES = Path(os.environ["EASYPATH_RESULTS"])
    ROOT = RES.parent
    SEED = 0
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from common import RES, ROOT, SEED  # noqa: E402,F401

_SLIDE_MAP = os.environ.get("EASYPATH_SLIDE_MAP", "")


def slide_path(file):
    """Slide file from the manifest, with the EASYPATH_SLIDE_MAP prefix rewrite applied."""
    if "=" in _SLIDE_MAP:
        old, new = _SLIDE_MAP.split("=", 1)
        if file.startswith(old):
            return new + file[len(old):]
    return file


ANN = RES / "annot"
PATCH_DIR = RES / "titan/patch_raw"      # CONCH v1.5 features + level-0 coords from step 07 (raw tiles)
THUMB_DIR = ANN / "roi_thumbs"
MAP_DIR = ANN / "tissue_map"
DB = ANN / "annotations.sqlite"
for d in (ANN, THUMB_DIR, MAP_DIR):
    d.mkdir(parents=True, exist_ok=True)

TITAN_REV = "dac6773d9961cfc75503440676ff157a2c6e8d2e"
ROI_PATCHES = 4      # ROI = 4 x 4 patches of 512 px at 20x = 2048 px = 1.02 mm per side
MIN_ROI_PATCHES = 4  # tissue patches needed for a block to become an ROI
WIN_RADIUS = 2       # tissue map: each patch is scored from its 5 x 5 patch neighbourhood

# tissue classes for the probability map (zero-shot prompts, TITAN text space)
TISSUE = {
    "tumor": ["synovial sarcoma", "malignant spindle cell tumor", "densely cellular sarcoma"],
    "stroma": ["fibrous stroma", "collagenous connective tissue"],
    "necrosis": ["necrosis", "necrotic tumor tissue"],
    "inflammation": ["lymphocytic infiltrate", "inflammatory cells", "lymphoid aggregate"],
    "benign_epithelium": ["benign epithelium", "normal skin epidermis"],
    "fat": ["adipose tissue"],
    "muscle": ["skeletal muscle"],
    "blood": ["hemorrhage", "blood vessels with red blood cells"],
}
# tumour morphology at ROI level (maps onto monophasic / biphasic / poorly differentiated SS)
PATTERN = {
    "spindle": ["monophasic synovial sarcoma with spindle cells", "fascicles of uniform spindle cells"],
    "glandular": ["biphasic synovial sarcoma with epithelial glands", "epithelial glandular structures in sarcoma"],
    "round_cell": ["poorly differentiated synovial sarcoma with round cells", "sheets of small round blue cells"],
}
# a2c: synovial sarcoma patterns (corded added) and stroma types, scored on CONCH v1 patch embeddings
SS_PATTERN = {
    "spindle": PATTERN["spindle"],
    "glandular": PATTERN["glandular"],
    "corded": ["epithelial cells arranged in cords in synovial sarcoma", "cords and nests of epithelioid tumor cells",
               "solid epithelial nests in synovial sarcoma"],
    "round_cell": PATTERN["round_cell"],
}
STROMA_TYPE = {
    "fibrous": ["fibrous stroma", "collagenous connective tissue"],
    "hyalinized": ["hyalinized stroma", "dense hyalinized collagen"],
    "myxoid": ["myxoid stroma", "loose myxoid matrix"],
    "calcified": ["stromal calcification", "calcified deposits in stroma"],
}
TEMPLATES = ["CLASSNAME.", "an image of CLASSNAME.", "H&E stained image of CLASSNAME.",
             "a histopathology image showing CLASSNAME.", "a region of a slide showing CLASSNAME."]

# token clusters: distinct colours for the viewer overlay and figures
# high contrast on H&E: no pink, purple or blue; neighbouring classes (s2 orders similar ones together) differ most
CLUSTER_COLORS = ["#ffd400", "#00b050", "#ff6a00", "#111111", "#ffffff", "#e00000", "#8b4a00", "#a6ff00",
                  "#005a1e", "#7f7f7f", "#8c8c00", "#00ffb0", "#5c2e00", "#c8a000", "#4d4d4d", "#ffe9a0"]
# segment / cluster review vocabulary: synovial sarcoma components first, then non-tumour tissue.
# Monophasic / biphasic / poorly differentiated are slide-level calls derived from component areas.
CLUSTER_LABELS = ["ss_spindle", "ss_glandular", "ss_corded", "ss_round_cell", "stroma", "necrosis", "inflammation",
                  "hemorrhage", "fat", "muscle", "vessel", "normal_tissue", "artifact", "mixed"]
SS_COMPONENTS = ["ss_spindle", "ss_glandular", "ss_corded", "ss_round_cell"]
# slide review: segment labels refine stroma; the slide subtype and its patterns are called by the reviewer
SEG_SLIDE_LABELS = ["ss_spindle", "ss_glandular", "ss_corded", "ss_round_cell", "stroma", "stroma_hyalinized",
                    "stroma_myxoid", "calcification", "necrosis", "inflammation", "hemorrhage", "fat", "muscle", "vessel",
                    "normal_tissue", "artifact", "mixed"]
SLIDE_SUBTYPES = ["monophasic", "biphasic", "poorly_differentiated", "not_ss", "cannot_assess"]
SLIDE_PATTERNS = ["glandular", "corded"]
BIPHASIC_MIN_EPI = 0.05   # epithelial (glandular + corded) share of tumour area to call biphasic
PD_MIN_ROUND = 0.20       # round-cell share of tumour area to flag poorly differentiated
CLUSTER_PURITY = ["clean", "mostly", "mixed"]

# review vocabulary shared by pipeline, refinement and UI
TUMOR_LABELS = ["tumor", "non_tumor"]
PATTERN_LABELS = list(PATTERN)
NECROSIS_LABELS = ["present", "absent"]


def load_patches(slide_id):
    import h5py
    with h5py.File(PATCH_DIR / f"{slide_id}.h5") as f:
        return f["features"][:], f["coords"][:], int(f["coords"].attrs["patch_size_level0"])


def latest_rois():
    """Path of the newest ROI prediction table (rois_v000.csv is zero-shot, later ones linear probes)."""
    files = sorted(ANN.glob("rois_v*.csv"))
    return files[-1] if files else None


def triage(conf, ood, blocked, lo, hi):
    """Review policy: blocked (poor QC) > needs_review (low confidence or OOD) > review > prelabel."""
    import numpy as np
    return np.select([blocked, ood | (conf < lo), conf < hi], ["blocked", "needs_review", "review"], "prelabel")
