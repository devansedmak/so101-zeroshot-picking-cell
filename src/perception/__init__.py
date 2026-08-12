"""Perception: zero-shot open-vocab detection (hosted VLM) -> pixel -> table homography,
plus the closed-loop placement verification that gates fulfillment (D9 thesis).

Primary path is the Cyberwave-hosted VLM (Gemini Robotics-ER) via ``detect`` (D10/D13);
BERT-NER + GroundingDINO remain an offline fallback (reference/old-pickplace.md).
``verify_placement`` reuses the same ``detect_points`` call to answer "is it in the bin?".
``locate_item`` is the composition the loop consumes: capture -> detect -> homography ->
table (X, Y) mm, i.e. the pick target the IK path needs (see locate.py for why).
``orient`` adds the *how it is turned* to that *where*, so the wrist can roll and the
jaws close across a long object instead of end-on.
"""

from .detect import (
    TASK_BOXES,
    TASK_POINTS,
    Detection,
    detect,
    parse_detections,
    pick_target,
)
from .homography import Homography, ReprojectionError
from .locate import (
    DEFAULT_HOMOGRAPHY_PATH,
    FrameSizeUnknown,
    HomographyMissing,
    ItemNotFound,
    LocatedItem,
    LocateError,
    load_homography,
    locate_item,
)
from .orient import MIN_ELONGATION, axis_angle_table, long_axis_pixels
from .verify import (
    DEFAULT_BIN_REGIONS_PATH,
    BinRegion,
    BinRegionsMissing,
    UnknownBinRegion,
    VerifyResult,
    bin_region_for,
    evaluate_placement,
    load_bin_regions,
    save_bin_regions,
    verify_placement,
)

__all__ = [
    "DEFAULT_BIN_REGIONS_PATH",
    "DEFAULT_HOMOGRAPHY_PATH",
    "MIN_ELONGATION",
    "TASK_BOXES",
    "TASK_POINTS",
    "BinRegion",
    "BinRegionsMissing",
    "Detection",
    "FrameSizeUnknown",
    "Homography",
    "HomographyMissing",
    "ItemNotFound",
    "LocateError",
    "LocatedItem",
    "ReprojectionError",
    "UnknownBinRegion",
    "VerifyResult",
    "axis_angle_table",
    "bin_region_for",
    "detect",
    "evaluate_placement",
    "load_bin_regions",
    "load_homography",
    "locate_item",
    "long_axis_pixels",
    "parse_detections",
    "pick_target",
    "save_bin_regions",
    "verify_placement",
]
