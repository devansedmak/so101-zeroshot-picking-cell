"""Locate the ordered item on the table: capture -> detect -> homography -> (X, Y) mm.

WHY THIS MODULE EXISTS. Every piece of the zero-shot claim already worked in isolation
(``capture`` -> ``detect`` -> ``homography`` -> ``control.ik``), but the runnable loop still
picked from the throwaway hardcoded pose table (``agent_service.poses.PICK_POSES``), so
the headline (*the arm picks what the camera sees*) was never proven end-to-end. This
is the one composition step that was missing. It adds **no new geometry maths**: it only
chains the four proven parts and hands the loop a table coordinate for
``poses.pick_plan_from_table`` -> IK.

Deliberately thin and pure-ish:
- ``client`` only needs ``mlmodels.run(...)`` (duck-typed, so a fake replays canned VLM
  output offline, no credits; same trick as detect.py / verify.py).
- camera code is imported **lazily inside the function**, so importing this module (and
  therefore the whole agent_service loop) never needs a camera or the SDK.
- the ``[y, x]`` / 0-1000 VLM convention is **not** re-derived here;
  it stays owned by ``detect.parse_detections``, and target selection reuses
  ``detect.pick_target`` so picking and verification agree on which blob "is" the item.

TODO(calibration): the homography's **world frame is assumed to be the IK base frame**:
same origin, same axes (see hardware/config/link-lengths.md §"Frame convention" and the
TODO(calibration) in src/control/ik.py). Nothing here detects or corrects an offset: if
the calibration sheet's origin is not at the arm base, every pick is off by that offset.
Cheapest fix at calibration time: put the sheet's origin corner **at the arm base**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import TASK_POINTS, Detection, detect, pick_target
from .homography import Homography

__all__ = [
    "DEFAULT_HOMOGRAPHY_PATH",
    "FrameSizeUnknown",
    "HomographyMissing",
    "ItemNotFound",
    "LocateError",
    "LocatedItem",
    "load_homography",
    "locate_item",
]

#: Where tools/calibrate_homography.py writes the pixel -> table calibration.
DEFAULT_HOMOGRAPHY_PATH = Path("hardware/config/homography.json")


class LocateError(RuntimeError):
    """Could not turn "the item is somewhere on the table" into a table coordinate."""


class ItemNotFound(LocateError):
    """The VLM returned no usable point for the ordered item in this frame."""


class FrameSizeUnknown(LocateError):
    """Could not determine the frame's pixel size (needed to de-normalize VLM coords)."""


class HomographyMissing(FileNotFoundError):
    """No pixel -> table calibration file yet (needs the mounted overhead camera).

    Mirrors ``verify.BinRegionsMissing``: callers degrade to the hardcoded pose table
    with a warning rather than crashing a demo run.
    """


@dataclass(frozen=True)
class LocatedItem:
    """Where the ordered item is, in both frames that matter, plus the evidence.

    ``pixel`` is the grasp point in the source frame's pixels, ``table_mm`` the same
    point on the table plane in mm (IK input), ``frame`` the JPEG it was decided on
    (keep it: it is the demo's narration and the post-mortem artifact for a bad pick).

    ``axis_deg`` is the object's long axis on the table (degrees mod 180, from
    ``orient``). That is what lets IK roll the wrist so the jaws close across the object.
    ``None`` means "no measurable axis": the pick still happens, at the default roll.
    """

    item: str
    pixel: tuple[float, float]
    table_mm: tuple[float, float]
    label: str = ""
    frame: Path | None = None
    detection: Detection | None = None
    axis_deg: float | None = None

    @property
    def x_mm(self) -> float:
        return self.table_mm[0]

    @property
    def y_mm(self) -> float:
        return self.table_mm[1]

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe form for session logs and alert metadata."""
        return {
            "item": self.item,
            "label": self.label,
            "pixel": [self.pixel[0], self.pixel[1]],
            "table_mm": [self.table_mm[0], self.table_mm[1]],
            "frame": str(self.frame) if self.frame else None,
            "axis_deg": self.axis_deg,
        }

    def describe(self) -> str:
        """One terse line for the run log / demo narration."""
        axis = f", axis {self.axis_deg:.0f}°" if self.axis_deg is not None else ""
        return (
            f"{self.item!r} at px=({self.pixel[0]:.0f}, {self.pixel[1]:.0f}) "
            f"→ table ({self.table_mm[0]:.0f}, {self.table_mm[1]:.0f}) mm{axis}"
        )


def load_homography(path: str | Path = DEFAULT_HOMOGRAPHY_PATH) -> Homography:
    """Load the pixel -> table calibration, or raise :class:`HomographyMissing`.

    The missing-file message is the instruction for fixing it: print it, don't swallow it.
    """
    p = Path(path)
    if not p.exists():
        raise HomographyMissing(
            f"no pixel→table calibration at {p} — mount the overhead camera and run "
            f"`.venv/bin/python tools/calibrate_homography.py` (hardware/config/cameras.md)"
        )
    return Homography.load(p)


def frame_size(image: str | Path) -> tuple[int, int]:
    """(width, height) in px of a saved frame, via OpenCV then Pillow (both lazy).

    Needed because the VLM answers on its own 0-1000 grid while the homography was
    fitted in real image pixels. Raises :class:`FrameSizeUnknown` with the fix.
    """
    path = Path(image)
    try:
        import cv2  # noqa: PLC0415, optional/heavy: probed at call time
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    try:
        from PIL import Image  # noqa: PLC0415, optional backend
    except ImportError:
        pass
    else:
        try:
            with Image.open(path) as im:
                return int(im.size[0]), int(im.size[1])
        except OSError:
            pass  # unreadable/not an image; fall through to the actionable error
    raise FrameSizeUnknown(
        f"could not read the pixel size of {path} (missing, empty, or not an image) — "
        f"pass image_size=(w, h) explicitly, or install a reader "
        f"(.venv/bin/pip install opencv-python)"
    )


def locate_item(
    item: str,
    *,
    client: Any,
    homography: Homography,
    image: str | Path | None = None,
    camera: str | int | None = None,
    model: Any = None,
    image_size: tuple[int, int] | None = None,
    task: str = TASK_POINTS,
) -> LocatedItem:
    """Where is ``item`` on the table? -> :class:`LocatedItem` (pixel + table mm).

    Args:
        item: the ordered item name, used verbatim as the open-vocab VLM prompt.
        client: anything exposing ``mlmodels.run(...)`` (the hosted VLM; a fake offline).
        homography: a fitted pixel -> table map (see :func:`load_homography`).
        image: a **saved** frame to re-use; makes the whole path re-runnable and
            testable and bypasses capture entirely. ``None`` means grab a fresh still.
        camera: ``/dev/videoN`` or index for the fresh capture (``None`` auto-selects).
        model: VLM model ref; ``None`` means ``$CYBERWAVE_VLM_MODEL`` (the default slug).
        image_size: (w, h) of ``image`` when it should not be probed from the file.
        task: VLM structured task (``detect_points`` by default).

    Raises:
        ItemNotFound: the VLM pointed at nothing usable for ``item``.
        FrameSizeUnknown: ``image_size`` unknown and the frame could not be read.
        capture.CaptureError: no camera / no backend (only when ``image`` is ``None``).
    """
    if image is None:
        from .capture import capture_still  # lazy: camera-gated module

        frame = capture_still(device=camera)
    else:
        frame = Path(image)

    size = image_size if image_size is not None else frame_size(frame)

    if model is None:
        import os  # lazy: keeps this module import-cheap and env-free for pure tests

        model = os.getenv("CYBERWAVE_VLM_MODEL", "gemini-robotics-er")

    detections = detect(client, str(frame), item, model=model, image_size=size, task=task)
    target = pick_target(detections, item)
    if target is None:
        raise ItemNotFound(
            f"the VLM saw no {item!r} in {frame} — is it on the table, in view, and "
            f"named the way the order names it?"
        )

    # The only geometry here: pixel -> table plane. Frame assumption: see module TODO.
    x_mm, y_mm = homography.project((target.x, target.y))

    # Best-effort: how is the item TURNED? (lets IK roll the wrist so the jaws close
    # across a long thin object instead of end-on.) Everything about this is optional:
    # OpenCV may be absent, the frame may be a stub, the blob may be round, and a
    # missing angle must degrade to the old fixed-roll grasp, never fail an order.
    axis_deg: float | None = None
    try:
        from . import orient  # noqa: PLC0415, lazy: OpenCV-gated, same rule as capture

        ends = orient.long_axis_pixels(frame, (target.x, target.y))
        if ends is not None:
            axis_deg = orient.axis_angle_table(homography, *ends)
    except Exception:  # noqa: BLE001, orientation is a bonus, never a failure mode
        axis_deg = None

    return LocatedItem(
        item=item,
        pixel=(target.x, target.y),
        table_mm=(x_mm, y_mm),
        label=target.label,
        frame=frame,
        detection=target,
        axis_deg=axis_deg,
    )
