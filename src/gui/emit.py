"""Bridge a **real** ``run_order`` execution into the dashboard, without touching it.

The least invasive route that exists: ``src.agent_service.run_order`` is not imported,
not monkey-patched and not edited. It already narrates itself on stdout, in a small and
stable vocabulary (``▶ order:``, ``💬 picking``, ``✅ plan complete``, ``👁 verified:``,
``🔔 alert dispatched:``, ``✅ fulfilled:``). :class:`RunOrderTranslator` reads those
lines and writes the dashboard's events to a **JSONL file**; the dashboard tails that
file. Two processes, one append-only file, zero coupling::

    # terminal 1: the real run, piped through the translator
    .venv/bin/python -m src.agent_service.run_order --verify --verify-fail 1 2>&1 \
      | .venv/bin/python -m src.gui.emit --out run-events.jsonl

    # terminal 2: the dashboard, following that file
    .venv/bin/python tools/dashboard.py --follow run-events.jsonl

The translator echoes every line it reads straight back to stdout, so the operator still
watches the run exactly as before; the JSONL is a side effect.

**The honest part.** ``run_order`` has no encoder read-back: ``MotionExecutor.execute``
publishes to MQTT and prints "plan complete" whether or not a servo ever moved. So this
translator renders a real run's PICK/PLACE as **COMMANDED** and *never* as VERIFIED:
there is nothing in that output that could justify green. Anything that *does* have
telemetry (``control.live_session.verify_pose``) can append its own
``{"kind": "motion", "state": "verified"|"mismatch"|"unverified", ...}`` line to the same
file and the dashboard will upgrade the stage. The file format is the whole API.

The JSONL is also just a log: replaying it re-runs the demo exactly (each order starts
with a ``reset``), which makes a filmed run reproducible without the robot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .events import ev, motion_event, stage_event
from .modes import DEFAULT_MODE, check_mode

#: Default event file. Kept out of the repo (it is gitignored).
DEFAULT_EVENT_PATH = Path("run-events.jsonl")

#: What ``run_order`` says versus what the dashboard draws. Anchored on the emoji the
#: loop/executor print, which are the most stable tokens in that output.
RE_ORDER = re.compile(r"^▶ order: pick '(?P<item>.+?)' → bin '(?P<bin>.+?)'")
RE_LOCATED = re.compile(
    r"👁\s+'(?P<item>.+?)' at px=\((?P<px>[-\d.]+), (?P<py>[-\d.]+)\)"
    r"\s*→ table \((?P<x>[-\d.]+), (?P<y>[-\d.]+)\) mm(?P<rest>.*)$"
)
RE_PICKING = re.compile(r"💬\s+picking (?P<item>.+?)\s*$")
RE_PLACING = re.compile(r"💬\s+placing in bin (?P<bin>.+?)\s*$")
RE_PLAN_DONE = re.compile(r"✅\s+plan complete")
RE_STEP = re.compile(r"▶\s+\[(?P<i>\d+)/(?P<n>\d+)\]\s+(?P<what>.+?)\s*$")
RE_VERDICT = re.compile(r"👁\s+(?P<neg>NOT )?verified: (?P<reason>.+?)\s*$")
RE_RETRY = re.compile(r"↻\s+retry (?P<n>\d+)/(?P<max>\d+): (?P<reason>.*?)\s*$")
RE_ALERT_STUB = re.compile(r"🔔\s+\(stub\) (?P<name>.+?)\s*$")
RE_ALERT_SENT = re.compile(r"🔔\s+alert dispatched: (?P<name>.+?)\s*$")
RE_FULFILLED = re.compile(r"^✅ fulfilled: (?P<item>.+?) → bin (?P<bin>.+?)\s*$")
RE_FAILED = re.compile(r"^❌ failed: (?P<item>.+?) → bin (?P<bin>\S+)\s+\((?P<reason>.*)\)\s*$")

#: The one sentence that keeps a real run honest on screen.
COMMANDED_DETAIL = "published to MQTT · no encoder read-back in this run"
PLAN_COMPLETE_DETAIL = (
    "MotionExecutor reported 'plan complete' — that is the WIRE, not the servos"
)


def _f(value: str) -> float:
    return float(value)


class RunOrderTranslator:
    """``run_order`` stdout -> dashboard events. Pure, line at a time, never raises.

    Stateful only in the way the output is: it remembers which move is in flight (so
    ``plan complete`` closes the right stage), which attempt it is on, and whether a
    verifier ever spoke (so a loop left open is drawn ``skipped``, not ``done``).
    """

    def __init__(self, *, mode: str = DEFAULT_MODE, log_unmatched: bool = True) -> None:
        self.mode = check_mode(mode)
        self.log_unmatched = log_unmatched
        self.reset()

    def reset(self) -> None:
        self.item: str = ""
        self.bin: str = ""
        self.attempt: int = 1
        self.retried: bool = False
        self.current_move: str | None = None
        self.detected: bool = False
        self.verified_seen: bool = False
        self.alerted: bool = False

    # ------------------------------------------------------------------ feeding
    def feed_all(self, lines: Iterable[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in lines:
            out.extend(self.feed(line))
        return out

    def feed(self, line: str) -> list[dict[str, Any]]:
        """Translate one line. Returns zero or more events (never raises)."""
        try:
            return self._feed(line.rstrip("\r\n"))
        except Exception:  # noqa: BLE001, a weird line must never break the bridge
            return []

    def _feed(self, line: str) -> list[dict[str, Any]]:
        text = line.strip()
        if not text:
            return []

        m = RE_ORDER.match(line)
        if m:
            return self._order(m.group("item"), m.group("bin"))

        m = RE_LOCATED.search(line)
        if m:
            return self._located(m)

        m = RE_PICKING.search(line)
        if m:
            return self._move("PICK", f"MotionExecutor · picking {m.group('item')}")

        m = RE_PLACING.search(line)
        if m:
            return self._move("PLACE", f"MotionExecutor · placing in bin {m.group('bin')}")

        if RE_PLAN_DONE.search(line):
            return self._plan_complete()

        m = RE_VERDICT.search(line)
        if m:
            return self._verdict(not m.group("neg"), m.group("reason"))

        m = RE_RETRY.search(line)
        if m:
            self.attempt = int(m.group("n"))
            self.retried = True
            return [ev("log", text=f"retry {m.group('n')}/{m.group('max')}: {m.group('reason')}")]

        m = RE_ALERT_SENT.search(line)
        if m:
            return self._alert(m.group("name"), dispatched=True)

        m = RE_ALERT_STUB.search(line)
        if m:
            return self._alert(m.group("name"), dispatched=False)

        m = RE_FULFILLED.match(line)
        if m:
            return self._outcome(
                "retried" if self.retried else "fulfilled",
                f"{m.group('item')} verified in bin {m.group('bin')}"
                + (f" after {self.attempt} attempts" if self.retried else " on the first attempt"),
            )

        m = RE_FAILED.match(line)
        if m:
            return self._outcome("failed", m.group("reason"))

        m = RE_STEP.search(line)
        if m:
            # Motion steps are narration only: re-activating the stage here would clear
            # the COMMANDED badge the stage is carrying, which is the one thing we must
            # never do (see events.MOTION_STATES).
            return [ev("log", text=f"[{m.group('i')}/{m.group('n')}] {m.group('what')}")]

        return [ev("log", text=text)] if self.log_unmatched else []

    # ------------------------------------------------------------------ handlers
    def _order(self, item: str, bin_label: str) -> list[dict[str, Any]]:
        self.reset()
        self.item, self.bin = item, bin_label.strip().upper()
        return [
            ev("reset", mode=self.mode, source="live"),
            ev("order", order_id="run_order", item=item, bin=self.bin),
            stage_event("ORDER", "done", "order accepted from the WMS source"),
            stage_event("DETECT", "active", "hosted VLM · detect_points"),
        ]

    def _located(self, m: "re.Match[str]") -> list[dict[str, Any]]:
        self.detected = True
        px, py = _f(m.group("px")), _f(m.group("py"))
        x, y = _f(m.group("x")), _f(m.group("y"))
        return [
            ev(
                "detection",
                pixel=[px, py],
                table_mm=[x, y],
                label=m.group("item"),
            ),
            stage_event("DETECT", "done", f"1 point · px ({px:.0f}, {py:.0f})"),
            stage_event("LOCATE", "active", "homography px → table mm · IK"),
            stage_event("LOCATE", "done", f"table ({x:.0f}, {y:.0f}) mm{m.group('rest').strip()}"),
        ]

    def _move(self, stage: str, detail: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if stage == "PICK" and not self.detected:
            # No perceived point reached this pick: either --perceive is off, or it
            # degraded to the pose table. Say so rather than letting DETECT/LOCATE
            # quietly go green for work that never happened.
            events += [
                stage_event("DETECT", "skipped", "no perceived point · hardcoded pose table"),
                stage_event("LOCATE", "skipped", "no perceived point · hardcoded pose table"),
            ]
            self.detected = True  # only announce it once per order
        self.current_move = stage
        events += [
            stage_event(stage, "active", detail, attempt=self.attempt),
            motion_event(stage, "commanded", detail=COMMANDED_DETAIL, attempt=self.attempt),
        ]
        return events

    def _plan_complete(self) -> list[dict[str, Any]]:
        stage = self.current_move
        if stage is None:
            return [ev("log", text="plan complete (no move in flight)")]
        self.current_move = None
        # The stage is finished; the MOVE is not confirmed. events.py keeps the
        # "commanded" motion report on the stage, so the chip stays amber, not green.
        return [stage_event(stage, "done", PLAN_COMPLETE_DETAIL, attempt=self.attempt)]

    def _verdict(self, ok: bool, reason: str) -> list[dict[str, Any]]:
        self.verified_seen = True
        return [
            stage_event("VERIFY", "active", "re-capture → detect_points → point-in-zone"),
            ev("verification", ok=ok, reason=reason, bin=self.bin, attempt=self.attempt),
            stage_event(
                "VERIFY",
                "done" if ok else "failed",
                "verdict: IN BIN" if ok else "verdict: NOT IN BIN → retry",
                attempt=self.attempt,
            ),
        ]

    def _alert(self, name: str, *, dispatched: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.alerted:
            self.alerted = True
            events.append(stage_event("ALERT", "active", "platform alert · robot.alerts.create"))
        severity = "error" if name.upper().startswith("ORDER FAILED") else "info"
        events.append(
            ev(
                "alert",
                name=name,
                description=(
                    "Raised by the run itself (robot.alerts.create)."
                    if dispatched
                    else "Dry-run alert: the run printed it but did not dispatch it."
                ),
                severity=severity,
                dispatched=dispatched,
                # Tells the dashboard NOT to re-send this to the platform: the run owns it.
                origin="run_order",
            )
        )
        return events

    def _outcome(self, status: str, detail: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.verified_seen:
            events.append(
                stage_event("VERIFY", "skipped", "no --verify · closed loop not exercised")
            )
        events.append(
            stage_event("ALERT", "done" if status != "failed" else "failed", "alert raised")
        )
        events.append(ev("outcome", status=status, detail=detail, attempts=self.attempt))
        return events


# ---------------------------------------------------------------- writing
class EventLog:
    """Append-only JSONL writer: one event per line, flushed immediately.

    Flushing every line is the whole point: the dashboard is tailing this file live, and
    a buffered writer would make the demo lag a stage behind the arm.
    """

    def __init__(self, path: str | Path = DEFAULT_EVENT_PATH, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a" if append else "w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._handle.write(json.dumps(event, default=str) + "\n")
            self._handle.flush()

    def write_all(self, events: Iterable[dict[str, Any]]) -> int:
        count = 0
        for event in events:
            self.write(event)
            count += 1
        return count

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------- reading
def _sleep(stop: threading.Event | None, seconds: float) -> bool:
    """Wait, returning ``True`` if we were asked to stop."""
    if stop is None:
        time.sleep(seconds)
        return False
    return stop.wait(seconds)


def tail_lines(
    path: str | Path,
    *,
    stop: threading.Event | None = None,
    poll: float = 0.25,
    from_start: bool = True,
) -> Iterator[str]:
    """Follow a growing text file, yielding complete lines (``tail -f``).

    Tolerates the file not existing yet (the dashboard is usually started first) and
    being truncated or replaced between runs (it reopens from the top).
    """
    p = Path(path)
    handle = None
    buf = ""
    pos = 0
    try:
        while not (stop is not None and stop.is_set()):
            if handle is None:
                try:
                    handle = p.open("r", encoding="utf-8", errors="replace")
                except OSError:
                    if _sleep(stop, poll):
                        return
                    continue
                if not from_start:
                    handle.seek(0, 2)
                pos = handle.tell()
                buf = ""
            chunk = handle.read()
            if chunk:
                pos = handle.tell()
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            if size < pos:  # truncated or replaced -> start over
                handle.close()
                handle = None
                continue
            if _sleep(stop, poll):
                return
    finally:
        if handle is not None:
            handle.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read a finished JSONL file in one go (replay a recorded run)."""
    p = Path(path)
    if not p.exists():
        return []
    return parse_events(p.read_text(encoding="utf-8", errors="replace").splitlines())


def parse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """JSONL lines -> event dicts, silently skipping anything unparseable."""
    events: list[dict[str, Any]] = []
    for line in lines:
        event = parse_event(line)
        if event is not None:
            events.append(event)
    return events


def parse_event(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("kind") else None


def follow_into(
    app: Any,
    path: str | Path,
    *,
    stop: threading.Event | None = None,
    poll: float = 0.25,
    from_start: bool = True,
) -> None:
    """Tail ``path`` and feed every event into a :class:`~src.gui.app.DashboardApp`."""
    for line in tail_lines(path, stop=stop, poll=poll, from_start=from_start):
        event = parse_event(line)
        if event is not None:
            app.ingest([event])


def follow_in_thread(
    app: Any,
    path: str | Path,
    *,
    stop: threading.Event | None = None,
    from_start: bool = True,
) -> threading.Thread:
    thread = threading.Thread(
        target=follow_into,
        args=(app, path),
        kwargs={"stop": stop, "from_start": from_start},
        name="event-tail",
        daemon=True,
    )
    thread.start()
    return thread


# ---------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.gui.emit",
        description="Translate run_order's stdout into dashboard events (JSONL). "
        "Reads stdin, echoes it back, appends events to --out.",
        epilog="example: python -m src.agent_service.run_order --verify 2>&1 "
        "| python -m src.gui.emit --out run-events.jsonl",
    )
    parser.add_argument("--out", default=str(DEFAULT_EVENT_PATH), help="JSONL file to write.")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Dashboard mode to tag the run with.")
    parser.add_argument("--append", action="store_true", help="Append instead of truncating.")
    parser.add_argument("--quiet", action="store_true", help="Do not echo stdin to stdout.")
    args = parser.parse_args(argv)

    translator = RunOrderTranslator(mode=args.mode)
    written = 0
    with EventLog(args.out, append=args.append) as log:
        print(f"[emit] writing dashboard events → {args.out}", file=sys.stderr, flush=True)
        for line in sys.stdin:
            if not args.quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            written += log.write_all(translator.feed(line))
    print(f"[emit] {written} events written to {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
