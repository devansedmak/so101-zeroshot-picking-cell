"""Perception: zero-shot open-vocab detection (hosted VLM) → pixel → table homography.

Primary path is the Cyberwave-hosted VLM (Gemini Robotics-ER) via ``detect`` (D10/D13);
BERT-NER + GroundingDINO remain an offline fallback (reference/old-pickplace.md).
"""

from .detect import (
    TASK_BOXES,
    TASK_POINTS,
    Detection,
    detect,
    parse_detections,
    pick_target,
)

__all__ = [
    "TASK_BOXES",
    "TASK_POINTS",
    "Detection",
    "detect",
    "parse_detections",
    "pick_target",
]
