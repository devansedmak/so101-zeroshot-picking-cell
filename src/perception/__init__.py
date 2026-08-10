"""Perception: zero-shot open-vocab detection (hosted VLM) → pixel → table homography,
plus the closed-loop placement verification that gates fulfillment (D9 thesis).

Primary path is the Cyberwave-hosted VLM (Gemini Robotics-ER) via ``detect`` (D10/D13);
BERT-NER + GroundingDINO remain an offline fallback (reference/old-pickplace.md).
``verify_placement`` reuses the same ``detect_points`` call to answer "is it in the bin?".
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
    "TASK_BOXES",
    "TASK_POINTS",
    "BinRegion",
    "BinRegionsMissing",
    "Detection",
    "Homography",
    "ReprojectionError",
    "UnknownBinRegion",
    "VerifyResult",
    "bin_region_for",
    "detect",
    "evaluate_placement",
    "load_bin_regions",
    "parse_detections",
    "pick_target",
    "save_bin_regions",
    "verify_placement",
]
