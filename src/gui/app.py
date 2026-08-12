"""The dashboard application object: state + frames + platform link, behind one lock.

The HTTP layer (:mod:`src.gui.server`) is deliberately thin: every decision lives here so
it can be exercised in tests without a socket. Two producers feed the same door:

    MockDriver.play(beat)                     # canned demo
    POST /events  ->  app.ingest([...])       # a real run_order execution

and one consumer reads :meth:`snapshot`, which is the exact JSON the page renders,
including the **overlay geometry** (zone rectangles, the detected point, the verification
point) already projected into the served crop, so the browser does no coordinate maths.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import frames as F
from .events import DashboardState
from .modes import DEFAULT_MODE, ZONES, check_mode
from .platform import PlatformIdentity, PlatformLink


class DashboardApp:
    """Everything the dashboard knows. Thread-safe; never raises at its public edges."""

    def __init__(
        self,
        *,
        mode: str = DEFAULT_MODE,
        source: str = "mock",
        view: F.View | None = None,
        auto_view: bool = True,
        platform: PlatformLink | None = None,
        regions_path: str | Path = F.BIN_REGIONS_PATH,
    ) -> None:
        self.lock = threading.RLock()
        self.rects = F.load_zone_rects(regions_path)
        self.frame_size = F.load_frame_size(regions_path)
        self.regions_calibrated = Path(regions_path).exists()
        if view is None:
            # A mock run knows where its props lie, so it can frame the pick area too;
            # a real camera only gets the zones (its scene is whatever is on the table).
            extra = list(F.PROP_HOME.values()) if source == "mock" else ()
            view = (
                F.auto_view(self.rects, self.frame_size, extra=extra)
                if auto_view
                else F.full_view(self.frame_size)
            )
        self.view = view.clamp_to(self.frame_size)
        self.frames = F.FrameStore(view=self.view, size=self.frame_size)
        self.state = DashboardState(mode=check_mode(mode), source=source)
        self.state.frame_size = list(self.frame_size)
        self.platform = platform or PlatformLink(identity=PlatformIdentity.from_env())
        self.driver: Any = None  # set by tools/dashboard.py in mock mode

    # ------------------------------------------------------------------ input
    def ingest(self, events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply events from any producer. Alerts are forwarded to the platform link."""
        applied = 0
        with self.lock:
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                kind = str(event.get("kind") or "").lower()
                if kind == "frame":
                    self._frame_from_event(event)
                index = len(self.state.alerts)
                if self.state.apply(event):
                    applied += 1
                    if kind == "alert" and len(self.state.alerts) > index:
                        self._dispatch_alert(index)
        return {"applied": applied, "seq": self.state.seq, "dropped": self.state.dropped}

    def _frame_from_event(self, event: Mapping[str, Any]) -> None:
        """``{"kind":"frame","path":...}`` from a real run -> load the bytes now."""
        path = event.get("path")
        if not path:
            return
        try:
            self.frames.set_path(path, token=str(event.get("token") or ""))
        except Exception:  # noqa: BLE001, a vanished /tmp capture must not blank the page
            pass

    def _dispatch_alert(self, index: int) -> None:
        alert = self.state.alerts[index]
        origin = str(alert.get("origin") or "dashboard")
        if origin != "dashboard":
            # A real run raised this one itself (run_order calls robot.alerts.create).
            # Re-sending would duplicate it in the platform's alert view.
            alert["platform"] = {
                "dispatched": bool(alert.get("dispatched")),
                "mock": False,
                "detail": f"raised by {origin}" + ("" if alert.get("dispatched") else " (dry-run)"),
            }
            return
        alert["platform"] = {"dispatched": False, "mock": not self.platform.enabled,
                             "detail": "sending…" if self.platform.enabled else "MOCK — not sent"}

        def done(result: dict[str, Any]) -> None:
            with self.lock:
                if index < len(self.state.alerts):
                    self.state.alerts[index]["platform"] = result
                    self.state.alerts[index]["dispatched"] = bool(result.get("dispatched"))

        self.platform.send(alert, done)

    def play(self, beat: Any) -> None:
        """Execute one :class:`src.gui.mock.Beat`: render its frame, apply its events."""
        if getattr(beat, "frame", None) is not None:
            self.render(**{k: v for k, v in beat.frame.items() if v is not None})
        self.ingest(beat.events)

    def render(self, **scene: Any) -> None:
        """Synthesize an overhead frame for the mock run (cv2, no camera, no network)."""
        try:
            img = F.synth_scene(rects=self.rects, size=self.frame_size, **scene)
            self.frames.set_image(img)
        except Exception:  # noqa: BLE001, no cv2 / bad kwargs must not stop the demo
            pass

    def set_mode(self, mode: str) -> str:
        with self.lock:
            self.state.mode = check_mode(mode)
        if self.driver is not None:
            self.driver.configure(mode=self.state.mode)
        return self.state.mode

    def replay(self, outcome: str | None = None) -> bool:
        """Restart the canned run (UI button). No-op with a live producer."""
        if self.driver is None:
            with self.lock:
                self.state.reset()
            return False
        self.driver.configure(outcome=outcome) if outcome else self.driver.restart()
        return True

    # ------------------------------------------------------------------ output
    def snapshot(self) -> dict[str, Any]:
        """The whole document the page polls."""
        with self.lock:
            doc = self.state.to_dict()
            doc["overlay"] = self._overlay()
        doc["platform"] = self.platform.to_dict()
        doc["view"] = self.view.to_dict()
        doc["frame_token"] = self.frames.token
        doc["calibrated"] = self.regions_calibrated
        driver = self.driver
        doc["controls"] = {
            "mock": driver is not None,
            "outcome": getattr(driver, "outcome", None),
            "playing": getattr(driver, "playing", None),
            "loop": bool(getattr(driver, "loop", False)),
        }
        return doc

    def _overlay(self) -> dict[str, Any]:
        """Zone rectangles + points, as 0..1 fractions of the served crop."""
        st = self.state
        dest = (st.routing or {}).get("bin") or (st.order or {}).get("bin")
        verification = st.verification

        zones = []
        for label, rect in sorted(self.rects.items()):
            zone = ZONES.get(label)
            x0, y0 = F.pixel_to_view_fraction(rect[0], rect[1], self.view)
            x1, y1 = F.pixel_to_view_fraction(rect[2], rect[3], self.view)
            active = bool(dest) and label == dest
            verdict = None
            if active and verification is not None:
                verdict = "ok" if verification.get("ok") else "fail"
            zones.append(
                {
                    "label": label,
                    "color": zone.color if zone else "grey",
                    "css": zone.css if zone else "#94a3b8",
                    "meaning": zone.meaning if zone else "",
                    "x": round(x0, 5),
                    "y": round(y0, 5),
                    "w": round(x1 - x0, 5),
                    "h": round(y1 - y0, 5),
                    "active": active,
                    "verdict": verdict,
                }
            )

        def frac(point: Sequence[float] | None) -> dict[str, float] | None:
            if not point:
                return None
            fx, fy = F.pixel_to_view_fraction(point[0], point[1], self.view)
            return {"x": round(fx, 5), "y": round(fy, 5)}

        det = st.detection or {}
        return {
            "zones": zones,
            "point": frac(det.get("pixel")),
            "point_label": det.get("label") or "",
            "point_mm": det.get("table_mm"),
            "verify": frac((verification or {}).get("point")),
            "verify_ok": bool((verification or {}).get("ok")) if verification else None,
            "destination": dest,
        }

    def frame_jpeg(self) -> tuple[bytes, str]:
        self.frames.ensure()
        return self.frames.get()
