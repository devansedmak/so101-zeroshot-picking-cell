"""The dashboard's (fail-soft) link to the Cyberwave platform.

Two things the video has to prove, and one thing it must never fake:

* **Identity**: the header shows the environment id, the twin id and the runtime
  (MOCK / SIMULATION / LIVE) it is actually running as, read from ``.env``. No network
  needed, so it is always there.
* **Alerts**: every alert the dashboard raises is shown in the UI *and*, when the run is
  real, dispatched through the already-verified surface in
  :mod:`src.agent_service.alerts` (``robot.alerts.create(name, description=..., severity=...,
  source_type=...)``, note there is no ``message`` kwarg) so the same alert lands in the
  Cyberwave alerts view while the camera is rolling.
* **Never fake it**: in ``--mock`` nothing is sent, and every alert card is stamped
  "MOCK — not sent". A mock alert can never be mistaken for a real one.

Everything network here is **lazy, threaded and swallowed**: connecting happens on a
background thread, sending happens on a worker queue, and a DNS/MQTT failure (we had one
on ``mqtt.cyberwave.com`` tonight) degrades the badge to OFFLINE without blocking a single
pipeline stage. The dashboard is a viewer: it never moves the robot and never calls
``cw.affect("live")``.

SDK note (verified offline against cyberwave 0.5.3): the client exposes no ``workflows``
attribute, so the ``cw.workflows.trigger(...)`` surface documented for the platform could not
be verified here. This module therefore **does not trigger workflows**; it only displays
the configured identity. See the dashboard's report/README note.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

#: Where a viewer goes to see the same alerts (shown as a link in the header).
DASHBOARD_URL = "https://cyberwave.com/dashboard"

#: Runtime badges. LIVE is *display only* here: this module never drives hardware.
RUNTIMES: tuple[str, ...] = ("MOCK", "SIMULATION", "LIVE")

#: alerts.create's source_type per runtime (see agent_service.alerts docstring).
SOURCE_TYPE_FOR: dict[str, str] = {
    "MOCK": "simulation",
    "SIMULATION": "simulation",
    "LIVE": "edge",
}

STATUS_MOCK = "mock"
STATUS_OFFLINE = "offline"
STATUS_CONNECTING = "connecting"
STATUS_CONNECTED = "connected"


def load_env() -> None:
    """Best-effort ``.env`` load (same helper the sim entrypoints use)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover, python-dotenv is a project dependency
        pass


@dataclass
class PlatformIdentity:
    """What the header shows. Pure config: no network, always available."""

    environment_id: str | None = None
    twin_id: str | None = None
    twin_name: str = "SO-101 follower"
    api_key: bool = False
    dashboard_url: str = DASHBOARD_URL
    workflow: str | None = None

    @classmethod
    def from_env(cls) -> "PlatformIdentity":
        load_env()
        return cls(
            environment_id=os.getenv("CYBERWAVE_ENVIRONMENT_ID") or None,
            twin_id=os.getenv("CYBERWAVE_TWIN_ID") or None,
            api_key=bool(os.getenv("CYBERWAVE_API_KEY")),
            workflow=os.getenv("CYBERWAVE_WORKFLOW") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "twin_id": self.twin_id,
            "twin_name": self.twin_name,
            "api_key": self.api_key,
            "dashboard_url": self.dashboard_url,
            "workflow": self.workflow,
        }


@dataclass
class PlatformLink:
    """Sends alerts to the platform when (and only when) the run is real.

    ``enabled=False`` (the ``--mock`` default) means: never import the SDK, never open a
    socket, and stamp every alert ``mock=True`` so the UI can say "not sent".
    """

    runtime: str = "MOCK"
    enabled: bool = False
    identity: PlatformIdentity = field(default_factory=PlatformIdentity)
    source_type: str | None = None
    #: Connect and show the real identity, but PRINT alerts instead of creating them,
    #: for rehearsing against the platform without filling its alert view.
    dry_run: bool = False

    _status: str = field(default=STATUS_MOCK, init=False)
    _error: str = field(default="", init=False)
    _robot: Any = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _queue: "queue.Queue[tuple[dict[str, Any], Callable[[dict[str, Any]], None]] | None]" = field(
        default_factory=queue.Queue, init=False
    )
    _worker: threading.Thread | None = field(default=None, init=False)
    _sent: int = field(default=0, init=False)
    _failed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime.upper() if self.runtime.upper() in RUNTIMES else "MOCK"
        if self.source_type is None:
            self.source_type = SOURCE_TYPE_FOR[self.runtime]
        self._status = STATUS_MOCK if not self.enabled else STATUS_OFFLINE

    # ---------------------------------------------------------------- state
    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def _set(self, status: str, error: str = "") -> None:
        with self._lock:
            self._status = status
            self._error = error

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            status, error, sent, failed = self._status, self._error, self._sent, self._failed
        return {
            "runtime": self.runtime,
            "enabled": self.enabled,
            "status": status,
            "error": error,
            "dry_run": self.dry_run,
            "source_type": self.source_type,
            "sent": sent,
            "failed": failed,
            "identity": self.identity.to_dict(),
            # Honest about the one thing we could not verify (SDK 0.5.3 has no
            # client.workflows). The header links to the dashboard instead.
            "workflow_trigger": False,
        }

    # ---------------------------------------------------------------- connect
    def start(self) -> None:
        """Kick off connection + the send worker. Returns immediately, always."""
        if not self.enabled:
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._pump, name="platform-send", daemon=True)
            self._worker.start()
        threading.Thread(target=self._connect, name="platform-connect", daemon=True).start()

    def stop(self) -> None:
        self._queue.put(None)

    def _connect(self) -> None:
        twin_id = self.identity.twin_id
        if not twin_id:
            self._set(STATUS_OFFLINE, "CYBERWAVE_TWIN_ID is not set")
            return
        if not self.identity.api_key:
            self._set(STATUS_OFFLINE, "CYBERWAVE_API_KEY is not set")
            return
        self._set(STATUS_CONNECTING)
        try:
            from cyberwave import Cyberwave  # lazy: --mock never imports the SDK

            cw = Cyberwave()
            # SAFETY: the dashboard is a viewer. Pin SIMULATION and
            # never call affect("live"); nothing here commands a joint either way.
            cw.affect("simulation")
            robot = cw.twin(twin_id=twin_id)
            with self._lock:
                self._robot = robot
                name = getattr(robot, "name", None)
            if name:
                self.identity.twin_name = str(name)
            self._set(STATUS_CONNECTED)
        except Exception as e:  # noqa: BLE001, DNS/MQTT flakiness is expected, not fatal
            self._set(STATUS_OFFLINE, f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- alerts
    def send(self, payload: dict[str, Any], done: Callable[[dict[str, Any]], None]) -> None:
        """Queue an alert. ``done`` is called with the dispatch result (on the worker).

        Non-blocking by construction: the pipeline must never wait on the platform.
        """
        if not self.enabled:
            done({"dispatched": False, "mock": True, "detail": "MOCK — not sent"})
            return
        self._queue.put((payload, done))

    def _pump(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            payload, done = job
            done(self._send_now(payload))

    def _send_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            robot = self._robot
        if robot is None:
            with self._lock:
                self._failed += 1
                err = self._error or "not connected"
            return {"dispatched": False, "mock": False, "detail": f"offline: {err}"}
        try:
            from src.agent_service.alerts import send_alert  # the verified SDK surface

            if self.dry_run:
                print(preview_alert(payload, self.source_type or "simulation"), flush=True)
                return {
                    "dispatched": False,
                    "mock": False,
                    "dry_run": True,
                    "detail": "DRY RUN — not sent",
                }
            alert = {
                k: str(v)
                for k, v in payload.items()
                if k in ("name", "description", "alert_type", "severity", "category") and v
            }
            alert.setdefault("severity", "info")
            alert.setdefault("category", "business")
            result = send_alert(robot, alert, source_type=self.source_type or "simulation")
            dispatched = bool(result.get("dispatched"))
            with self._lock:
                if dispatched:
                    self._sent += 1
                else:
                    self._failed += 1
            return {
                "dispatched": dispatched,
                "mock": False,
                "detail": "sent to Cyberwave" if dispatched else "not dispatched",
                "source_type": self.source_type,
            }
        except Exception as e:  # noqa: BLE001, an alert failure must not break the demo
            with self._lock:
                self._failed += 1
            self._set(STATUS_OFFLINE, f"{type(e).__name__}: {e}")
            return {"dispatched": False, "mock": False, "detail": f"send failed: {type(e).__name__}"}


def preview_alert(payload: dict[str, Any], source_type: str = "simulation") -> str:
    """Render what *would* be sent: the --dry-run path, so verifying costs no real alert."""
    lines = [f"[platform] would create alert (source_type={source_type}):"]
    for key in ("name", "severity", "alert_type", "category", "description"):
        if payload.get(key):
            lines.append(f"           {key}: {payload[key]}")
    return "\n".join(lines)
