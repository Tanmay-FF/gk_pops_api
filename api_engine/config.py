"""Slim config for the headless tracking engine.

Keeps only what the JSON-only code path actually reads:
  - Detection + classifier weight paths
  - YOLO + BoTSORT settings
  - Classification preprocessing transform and bag class labels
  - Linker / motion / co-movement / direction thresholds
  - Processing cadence (JSON_EVERY_N_FRAMES, CLASSIFY_EVERY_N_FRAMES)

Everything related to rendering, BEV, VLM, pose, analytics, case reports, and
zone congestion has been stripped — `pops_api` does not run those paths.

Paths default to a layout where this engine lives at
    <repo>/pops_api_v1/api_engine/config.py
and the model weights at
    <repo>/pops_api_v1/weights/...
Override with the POPS_API_MODEL_ROOT env var if your layout differs.
"""
from __future__ import annotations

import os
from pathlib import Path

from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths — resolved relative to the pops_api_v1 package root (one parent up).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../pops_api_v1/api_engine
_PKG_ROOT = _THIS_DIR.parent                         # .../pops_api_v1

# Allow override so the same image can be deployed with weights mounted at
# a different path (e.g. /opt/models on Cloud Run).
_MODEL_ROOT_ENV = os.environ.get("POPS_API_MODEL_ROOT")
_MODEL_ROOT = Path(_MODEL_ROOT_ENV) if _MODEL_ROOT_ENV else (_PKG_ROOT / "weights")

MODEL_PATH = str(_MODEL_ROOT / "detection" / "weights" / "best.pt")
TRACKER_CONFIG = str(_THIS_DIR / "botsort_retail.yaml")
QUALITY_WEIGHT_PATH = str(_MODEL_ROOT / "cart_quality" / "weights" / "best.pt")
FILL_WEIGHT_PATH = str(_MODEL_ROOT / "fill_and_bag_classifier" / "weights" / "best.pt")

# ---------------------------------------------------------------------------
# Cart classifier preprocessing
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

BAG_CLASSES = ("bagged", "unbagged", "not_applicable")
BAG_NA_IDX = BAG_CLASSES.index("not_applicable")
EMPTY_OVERRIDE_THRESH = 0.5

# ---------------------------------------------------------------------------
# Drawing colours (BGR for OpenCV). Mirror the Gradio demo so the classifier
# crops the same pixels — the fill/bag head conditions on the centroid dot
# and bbox edge that the renderer paints, so the trail/box must be on the
# frame BEFORE the classifier sees it. Annotated-video output reuses them.
# ---------------------------------------------------------------------------
COLOR_PERSON = (0, 230, 118)
COLOR_CART   = (0, 165, 255)
COLOR_LINK   = (255, 50, 255)

COLOR_PUSHOUT    = (0, 0, 255)
COLOR_SUSPICIOUS = (0, 140, 255)
COLOR_MONITORING = (0, 220, 220)
COLOR_CLEAR      = (0, 200, 0)

CLR_VALID   = (0, 200, 0)
CLR_UNCLEAR = (0, 0, 220)
CLR_EMPTY   = (153, 211, 52)
CLR_PARTIAL = (36, 191, 251)
CLR_FULL    = (68, 68, 239)
CLR_NA      = (184, 163, 148)

FILL_COLOR_MAP = {"EMPTY": CLR_EMPTY, "PARTIAL": CLR_PARTIAL, "FULL": CLR_FULL}

# Trail length used by draw_centroid_trail. Matches the Gradio demo so the
# pixels the classifier sees are bit-identical between the two code paths.
TRAIL_MAX_LEN = 50

# ---------------------------------------------------------------------------
# Detection + processing cadence
# ---------------------------------------------------------------------------
YOLO_IMGSZ = 640
# Mirror the Gradio demo's CLASSIFY_EVERY_N_FRAMES=8 exactly. Anything else
# changes which frames feed _cart_cls_history (the reconciliation vote)
# and so changes the final fill/bag decision.
CLASSIFY_EVERY_N_FRAMES = 1
# pcenf=1 means POPS scoring runs every frame, reading whatever the
# classifier last cached. Gradio has no separate POPS cadence — it
# reads _cart_cls_cache every frame — so pcenf=1 is the only value that
# matches Gradio behaviour. settings.py mirrors this default.
POPS_CLASSIFY_EVERY_N_FRAMES = 8
JSON_EVERY_N_FRAMES = 1
QUALITY_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Linking hyper-parameters (PersonCartLinker)
# ---------------------------------------------------------------------------
LINK_CONFIRM_FRAMES = 6
LINK_CONTESTED_FRAMES = 20
LINK_GRACE_FRAMES = 15
LINK_DRIFT_FRAMES = 6
STALE_CART_FRAMES = 30
ABANDON_FRAMES = 30
WALKAWAY_DIST_THRESH = 200

# Cart re-identification
REID_DIST_THRESH = 200
REID_MAX_GONE_FRAMES = 15

# ---------------------------------------------------------------------------
# Motion thresholds
# ---------------------------------------------------------------------------
SPEED_STATIC = 10
SPEED_SLOW = 100
SPEED_MEDIUM = 240

# Co-movement (linker uses this to reject coincidental overlap)
COMOVEMENT_MIN_POSITIONS = 4
COMOVEMENT_WINDOW = 6
COMOVEMENT_STATIC_PX = 5
COMOVEMENT_COS_THRESH = 0.3

# Direction labelling
DIRECTION_MIN_POSITIONS = 10
DIRECTION_MIN_DY = 20

# ---------------------------------------------------------------------------
# Camera placement strings — the single source of truth. schemas.CameraPlacement
# builds its enum from these, and motion.compute_direction_label matches
# against them. Drift here breaks both ends loudly.
# ---------------------------------------------------------------------------
PLACEMENT_OUTSIDE_FACING_ENTRANCE = "outside_facing_entrance"
PLACEMENT_INSIDE_FACING_EXIT      = "inside_facing_exit"
PLACEMENT_INSIDE_EXIT_ON_RIGHT    = "inside_exit_on_right"
PLACEMENT_INSIDE_EXIT_ON_LEFT     = "inside_exit_on_left"
PLACEMENT_INSIDE_EXIT_ON_BOTH     = "inside_exit_on_both"

# ---------------------------------------------------------------------------
# Per-track history retention. compute_motion / compute_direction_label only
# look at the head and tail of the position/timestamp/speed lists; capping
# them keeps memory bounded on long videos (multi-hour streams used to
# accumulate one tuple per frame per track forever).
# ---------------------------------------------------------------------------
TRACK_HISTORY_MAXLEN = 600  # ~20s at 30fps
