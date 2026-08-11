"""Grab ONE still overhead frame — robustly, three ways.

Everything downstream (detect.py → homography.py → IK → verify.py) starts from a
single still frame, so this is the one hardware dependency the whole loop shares.
Cohort office hours reported the dashboard *live stream* disconnecting every 4–5 s for
many builders, and that the reliable path was a one-shot "get frame" call. So we never
depend on a stream: three independent paths, in decreasing order of trust.

1. ``capture_still()`` — local V4L2 webcam, OpenCV if importable else **ffmpeg**
   (`/usr/bin/ffmpeg`, confirmed present). This is the tomorrow-morning path.
2. ``capture_twin_frame()`` — platform side, one REST ``latest-frame`` call.
3. neither → :class:`CaptureError` with the device listing, so failures are readable.

VERIFIED SDK SURFACE (introspected against the installed ``cyberwave`` 0.5.3 — not
guessed; decisions.md D15 is the scar tissue that made this rule):

    cw = Cyberwave(); twin = cw.twin(twin_id=...)      # -> CameraTwin
    twin.get_frame(format="bytes", *, path=None, source="cloud", sensor_id=None,
                   mock=False, idx=0, max_age_ms=None, zenoh_timeout_s=3.0,
                   edge_timeout_s=5.0) -> Any | None
        # defined in cyberwave/twin/mixins.py:273 -> cyberwave/twin/sensors/camera.py:75
        # format: "bytes" | "numpy" | "pil" | "path" (JPEG on disk, abs path returned)
        # source: "cloud"  = REST GET /api/v1/twins/{uuid}/latest-frame   <- what we use
        #         "local"  = active twin.stream() frame, else USB cam at idx
        #         "zenoh"  = cw.data frames subscribe
        #         "remote_edge" = MQTT take_photo (raises on timeout; others fail-SOFT
        #                         and return None — hence the None check below)
        # NOTE: in cw.affect("simulation") ONLY source="cloud" is allowed (ValueError).
    Deprecated aliases kept as fallbacks for older/newer SDK shuffles:
        twin.capture_frame(format="path")            (mixins.py:364, DeprecationWarning)
        twin.get_latest_frame(...) -> bytes          (twin/base.py:392, deprecated)
        cw.twins.get_latest_frame(twin_id, ...) -> bytes   (resources.py:1996)

Device auto-selection never hardcodes ``/dev/video4``: node numbers reshuffle on replug.
We read ``/sys/class/video4linux/*/name`` and **exclude the built-in Acer laptop camera**
by name (see hardware/config/cameras.md).

CLI (this is step 1 of the tomorrow-morning runbook):
    python -m src.perception.capture                    # list video devices
    python -m src.perception.capture --save frame.jpg   # grab one still
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "CaptureError",
    "capture_still",
    "capture_twin_frame",
    "list_video_devices",
    "select_device",
]


class CaptureError(RuntimeError):
    """Could not obtain a frame (no device, device busy, black frame, no backend)."""


# Where the kernel publishes human-readable names for every /dev/videoN node.
SYS_V4L_DIR = Path("/sys/class/video4linux")

# Built-in Acer laptop camera (USB 5986:215d, /dev/video0-3, "Integrated RGB Camera",
# vendor SunplusIT). NEVER the overhead cam — a laptop-lid camera cannot be fixed
# relative to the table, which planar homography requires (decisions D5/D11).
BUILTIN_NAME_HINTS: tuple[str, ...] = (
    "integrated rgb",
    "integrated camera",
    "sunplusit",
    "acer",
    "ir camera",
    "infrared",
)

# The WowRobo kit camera (USB 05a3:9230 "ARC International Camera"); the driver
# container sees it as "USB2.0_CAM1". Hints only *rank* devices — anything that is not
# the built-in is still eligible, so an unknown name never blocks the demo.
KIT_NAME_HINTS: tuple[str, ...] = (
    "usb2.0_cam",
    "usb2.0 cam",
    "arc international",
    "usb camera",
    "uvc camera",
    "webcam",
)

_DEFAULT_WARMUP = 5
_FFMPEG_TIMEOUT_S = 20.0

#: Frame size every capture in the pipeline must use. The kit camera (05a3:9230)
#: offers 1920x1080 over MJPEG only; raw YUYV maxes out at 640x480 on USB 2.0.
#: This is a CALIBRATION-CRITICAL constant: homography.json and bin-regions.json
#: store *pixel* coordinates, so changing it after calibrating invalidates both.
CAPTURE_SIZE = (1920, 1080)


def _warn_if_unexpected_size(actual: tuple[int, int]) -> None:
    """Shout if the driver gave us a size other than :data:`CAPTURE_SIZE`.

    A silent fallback would still produce a perfectly good-looking JPEG whose pixel
    coordinates no longer match the calibration — the worst possible failure mode.
    """
    if tuple(actual) != CAPTURE_SIZE:
        print(
            f"[capture] ⚠ WARNING: got {actual[0]}x{actual[1]}, expected "
            f"{CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}. Pixel coordinates will NOT match "
            "hardware/config/homography.json or bin-regions.json — recalibrate or fix "
            "the camera before trusting any pick.",
            file=sys.stderr,
        )


# --------------------------------------------------------------- enumeration
def _node_index(name: str) -> int:
    """Sort key: numeric part of ``videoN`` (so video10 sorts after video4)."""
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 1 << 30


def list_video_devices(sys_dir: str | Path = SYS_V4L_DIR) -> list[tuple[str, str]]:
    """Enumerate V4L2 nodes as ``(device_path, human_name)``, ordered by node number.

    Reads ``/sys/class/video4linux/videoN/name`` (no v4l-utils needed — the host lacks
    it, see cameras.md). Returns ``[]`` when nothing is plugged in / the path is absent.
    """
    root = Path(sys_dir)
    if not root.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for entry in sorted(root.iterdir(), key=lambda p: _node_index(p.name)):
        if not entry.name.startswith("video"):
            continue
        try:
            human = (entry / "name").read_text().strip()
        except OSError:
            human = "?"
        out.append((f"/dev/{entry.name}", human))
    return out


def _is_builtin(human: str) -> bool:
    low = human.lower()
    return any(hint in low for hint in BUILTIN_NAME_HINTS)


def _looks_like_kit(human: str) -> bool:
    low = human.lower()
    return any(hint in low for hint in KIT_NAME_HINTS)


def _sysfs_index(device: str, sys_dir: str | Path = SYS_V4L_DIR) -> int:
    """``/sys/.../videoN/index`` — 0 is the capture node, 1+ is usually metadata.

    UVC cameras expose a capture node *and* a metadata node (the kit cam showed up as
    /dev/video4 + /dev/video5). Opening the metadata node yields no images, so prefer
    index 0. Unknown ⇒ 0 (optimistic: try it rather than skip the only candidate).
    """
    node = Path(device).name
    try:
        return int((Path(sys_dir) / node / "index").read_text().strip())
    except (OSError, ValueError):
        return 0


def select_device(
    devices: Iterable[tuple[str, str]] | None = None,
    *,
    sys_dir: str | Path = SYS_V4L_DIR,
) -> str:
    """Pick the overhead (kit) camera node, skipping the built-in laptop camera.

    Ranking: named-like-the-kit first, then capture nodes (sysfs ``index`` 0) before
    metadata nodes, then the lowest node number. Raises :class:`CaptureError` listing
    everything it saw when only the built-in (or nothing) is present — the message is
    the troubleshooting output, so read it rather than guessing.
    """
    listing = list(devices) if devices is not None else list_video_devices(sys_dir)
    candidates = [(p, n) for (p, n) in listing if not _is_builtin(n)]
    if not candidates:
        seen = ", ".join(f"{p} ({n})" for p, n in listing) or "none"
        raise CaptureError(
            "no external camera found — only the built-in laptop camera is present.\n"
            f"  saw: {seen}\n"
            "  fix: plug the kit camera (USB 05a3:9230) into a USB port, wait 2 s, and\n"
            "       re-run `python -m src.perception.capture`. If it still does not show\n"
            "       up, try another port/cable (the kit cable is 3 m) and check `dmesg | tail`."
        )
    candidates.sort(
        key=lambda d: (
            0 if _looks_like_kit(d[1]) else 1,
            _sysfs_index(d[0], sys_dir),
            _node_index(Path(d[0]).name),
        )
    )
    return candidates[0][0]


# ------------------------------------------------------------------ backends
def _cv2_or_none() -> Any | None:
    try:
        import cv2  # noqa: PLC0415 — optional backend, probed at call time
    except ImportError:
        return None
    return cv2


def choose_backend(preferred: str | None = None) -> str:
    """Resolve the still-capture backend: ``"cv2"`` or ``"ffmpeg"``.

    ``preferred`` forces one (and errors if it is unavailable) — used by the CLI and by
    the offline tests. Otherwise OpenCV wins when importable (in-process, no subprocess),
    with ffmpeg as the always-there fallback.
    """
    if preferred:
        pref = preferred.strip().lower()
        if pref == "cv2":
            if _cv2_or_none() is None:
                raise CaptureError("backend 'cv2' requested but opencv-python is not installed")
            return "cv2"
        if pref == "ffmpeg":
            if shutil.which("ffmpeg") is None:
                raise CaptureError("backend 'ffmpeg' requested but the ffmpeg binary is not on PATH")
            return "ffmpeg"
        raise CaptureError(f"unknown backend {preferred!r} (use 'cv2' or 'ffmpeg')")
    if _cv2_or_none() is not None:
        return "cv2"
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    raise CaptureError(
        "no capture backend: neither opencv-python (import cv2) nor the ffmpeg binary.\n"
        "  fix: .venv/bin/pip install opencv-python   (or: sudo apt-get install ffmpeg)"
    )


def _default_out_path() -> Path:
    fd, name = tempfile.mkstemp(prefix="overhead_", suffix=".jpg")
    os.close(fd)
    return Path(name)


def _capture_cv2(device: str, out_path: Path, warmup_frames: int) -> None:
    cv2 = _cv2_or_none()
    if cv2 is None:  # pragma: no cover — guarded by choose_backend
        raise CaptureError("opencv-python is not installed")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise CaptureError(
                f"cv2 could not open {device} — is it the metadata node, or busy?\n"
                "  fix: `python -m src.perception.capture` to list nodes; stop any other\n"
                "       process holding the camera (`fuser -v /dev/video*`); try --backend ffmpeg."
            )
        # MJPEG *before* the size: raw YUYV at 1080p exceeds USB 2.0 bandwidth, so the
        # driver silently falls back to 640x480 if the size is requested on its own.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        frame = None
        # Auto-exposure/auto-white-balance need a few frames or the first one is black.
        for _ in range(max(0, warmup_frames) + 1):
            ok, buf = cap.read()
            if ok and buf is not None:
                frame = buf
        if frame is None:
            raise CaptureError(f"cv2 opened {device} but read no frame (all reads failed)")
        _warn_if_unexpected_size((frame.shape[1], frame.shape[0]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), frame):
            raise CaptureError(f"cv2 could not write {out_path} (bad extension or path?)")
    finally:
        cap.release()


def _capture_ffmpeg(device: str, out_path: Path, warmup_frames: int) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:  # pragma: no cover — guarded by choose_backend
        raise CaptureError("the ffmpeg binary is not on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "v4l2",
        # Match _capture_cv2 exactly: a backend switch must not change the frame size,
        # or every pixel coordinate in homography.json / bin-regions.json goes stale.
        "-input_format", "mjpeg", "-video_size", f"{CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}",
        "-i", device,
    ]
    if warmup_frames > 0:
        # Drop the first N frames *after* decoding, then take exactly one.
        cmd += ["-vf", f"select=gte(n\\,{int(warmup_frames)})", "-vsync", "0"]
    cmd += ["-frames:v", "1", str(out_path)]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise CaptureError(f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S:.0f}s on {device}") from e
    except OSError as e:
        raise CaptureError(f"could not run ffmpeg: {e}") from e
    if proc.returncode != 0:
        raise CaptureError(
            f"ffmpeg failed on {device} (exit {proc.returncode}): "
            f"{(proc.stderr or '').strip()[:400]}"
        )
    try:
        from PIL import Image

        with Image.open(out_path) as im:
            _warn_if_unexpected_size(im.size)
    except Exception:  # noqa: BLE001 — a size check must never fail the capture
        pass


def capture_still(
    device: str | int | None = None,
    out_path: str | Path | None = None,
    warmup_frames: int = _DEFAULT_WARMUP,
    *,
    backend: str | None = None,
) -> Path:
    """Grab one still from a local V4L2 webcam and return the JPEG path.

    Args:
        device: ``/dev/videoN``, an integer index (``4`` → ``/dev/video4``), or ``None``
            to auto-select via :func:`select_device` (built-in laptop cam excluded).
        out_path: destination; a temp ``overhead_*.jpg`` when ``None``.
        warmup_frames: frames to discard before keeping one — webcam auto-exposure
            needs a moment or the first frame comes out black.
        backend: force ``"cv2"`` / ``"ffmpeg"`` (default: cv2 if importable, else ffmpeg).

    Raises:
        CaptureError: no device, no backend, device unopenable, or no frame written.
    """
    dev = (
        select_device()
        if device is None
        else (f"/dev/video{device}" if isinstance(device, int) else str(device))
    )
    out = Path(out_path) if out_path is not None else _default_out_path()
    which = choose_backend(backend)
    if which == "cv2":
        _capture_cv2(dev, out, warmup_frames)
    else:
        _capture_ffmpeg(dev, out, warmup_frames)
    if not out.exists() or out.stat().st_size == 0:
        raise CaptureError(f"{which} reported success but {out} is missing/empty")
    return out


# ------------------------------------------------------------- platform path
def _write_bytes(data: bytes, out_path: Path) -> Path:
    if not data:
        raise CaptureError("the platform returned an empty frame")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def capture_twin_frame(
    client: Any,
    twin_id: str,
    out_path: str | Path | None = None,
    *,
    source: str = "cloud",
    sensor_id: str | None = None,
) -> Path:
    """Fetch the twin's latest frame through the Cyberwave SDK → JPEG path.

    Uses the verified surface documented in this module's docstring, newest API first:
    ``twin.get_frame("path", path=..., source="cloud")``, then the deprecated
    ``twin.capture_frame`` / ``twin.get_latest_frame``, then
    ``client.twins.get_latest_frame(twin_id)``. ``source="cloud"`` is the REST
    ``latest-frame`` endpoint — the one-shot call that does not depend on the flaky
    dashboard stream; it is also the only source allowed under ``cw.affect("simulation")``.

    Raises:
        CaptureError: the SDK returned no frame (sensor not streaming yet — mount the
            camera and start the streamer first, see hardware/config/cameras.md).
        NotImplementedError: the installed SDK exposes none of the four calls above;
            the message lists what it *does* expose so nobody has to guess an API name.
    """
    out = Path(out_path) if out_path is not None else _default_out_path()

    twin: Any = None
    getter = getattr(client, "twin", None)
    if callable(getter):
        try:
            twin = getter(twin_id=twin_id)
        except TypeError:
            twin = getter(twin_id)

    if twin is not None and hasattr(twin, "get_frame"):
        got = twin.get_frame("path", path=str(out), source=source, sensor_id=sensor_id)
        if got is None:
            raise CaptureError(
                f"twin {twin_id} returned no frame from source={source!r} — the camera "
                "sensor is probably not streaming yet (mount + start the streamer, "
                "cameras.md step 2), or the twin has no imaging sensor."
            )
        got_path = Path(str(got))
        if got_path != out and got_path.exists():
            shutil.copyfile(got_path, out)
        return out

    if twin is not None and hasattr(twin, "capture_frame"):  # deprecated alias
        got = twin.capture_frame(format="path")
        got_path = Path(str(got))
        if got_path != out and got_path.exists():
            shutil.copyfile(got_path, out)
        return out

    if twin is not None and hasattr(twin, "get_latest_frame"):  # deprecated alias
        return _write_bytes(twin.get_latest_frame(sensor_id=sensor_id), out)

    twins = getattr(client, "twins", None)
    if twins is not None and hasattr(twins, "get_latest_frame"):
        return _write_bytes(twins.get_latest_frame(twin_id, sensor_id=sensor_id), out)

    surface = sorted(n for n in dir(twin if twin is not None else client) if not n.startswith("_"))
    raise NotImplementedError(
        "no frame-capture call found on the installed Cyberwave SDK. Expected one of "
        "twin.get_frame / twin.capture_frame / twin.get_latest_frame / "
        "client.twins.get_latest_frame (all four verified present in cyberwave 0.5.3). "
        f"Available attributes instead: {surface}"
    )


# ------------------------------------------------------------------ CLI
def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m src.perception.capture",
        description="List V4L2 cameras and grab one still frame (no stream, no robot).",
        epilog=(
            "Tomorrow morning, run this FIRST: with no arguments it must list the kit "
            "camera (a node that is not 'Integrated RGB Camera'). Then --save frame.jpg "
            "and open the file — if it is black, raise --warmup."
        ),
    )
    p.add_argument("-d", "--device", help="/dev/videoN or an index (default: auto-select).")
    p.add_argument("-s", "--save", help="Grab one frame and write it here (e.g. frame.jpg).")
    p.add_argument(
        "-w", "--warmup", type=int, default=_DEFAULT_WARMUP, help="Frames to discard first."
    )
    p.add_argument("--backend", choices=("cv2", "ffmpeg"), help="Force a capture backend.")
    p.add_argument(
        "--twin",
        metavar="TWIN_ID",
        help="Instead of the local webcam, fetch the twin's latest frame via the SDK "
        "(one REST call; needs CYBERWAVE_API_KEY; requires --save).",
    )
    args = p.parse_args(argv)

    devices = list_video_devices()
    print(f"[capture] V4L2 devices ({len(devices)}):")
    for path, human in devices:
        tag = " <-- built-in laptop cam, skipped" if _is_builtin(human) else ""
        print(f"    {path:<14} {human}{tag}")
    if not devices:
        print("    (none — is the camera plugged in?)")
    try:
        print(f"[capture] auto-selected: {select_device(devices)}")
    except CaptureError as e:
        print(f"[capture] auto-select failed: {e}")

    if args.twin:
        if not args.save:
            print("ERROR: --twin needs --save PATH.", file=sys.stderr)
            return 2
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        from cyberwave import Cyberwave

        try:
            path = capture_twin_frame(Cyberwave(), args.twin, args.save)
        except (CaptureError, NotImplementedError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"[capture] twin frame saved → {path} ({path.stat().st_size} bytes)")
        return 0

    if not args.save:
        print("[capture] (no --save given; nothing captured)")
        return 0
    try:
        path = capture_still(args.device, args.save, args.warmup, backend=args.backend)
    except CaptureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[capture] backend={choose_backend(args.backend)} saved → {path} "
          f"({path.stat().st_size} bytes). Open it and check the whole table is visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
