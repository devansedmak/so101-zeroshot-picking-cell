#!/usr/bin/env python3
"""Demo dashboard for the order-driven picking cell — the thing the camera films.

    .venv/bin/python tools/dashboard.py                 # MOCK (default): no network, no hardware
    .venv/bin/python tools/dashboard.py --mode fusion   # start in a different mode
    .venv/bin/python tools/dashboard.py --follow run-events.jsonl --runtime simulation

One stdlib ``http.server`` process (same call as ``src.agent_service.order_api``, D16),
one self-contained HTML page, no CDN and no external asset — it renders with the network
cable pulled out. The bare command is the fallback demo: a scripted run drives every
stage, every mode and all four motion states with nothing plugged in.

**Safety (CLAUDE.md rule 2).** This is a *viewer*. It never imports
``control.live_session``, never opens a serial port and never calls ``cw.affect("live")``;
``--runtime live`` only changes an alert's ``source_type`` to "edge" and the header badge.
Nothing here can move the robot.

**What is real, what is not.** ``--mock`` synthesizes the camera frame and the run, but
the zone rectangles come from the surveyed ``hardware/config/bin-regions.json``, the
millimetres come from the calibrated ``hardware/config/homography.json``, the routing goes
through the same ``src.gui.modes.route`` a live run uses, and the verification verdict is
the same point-in-rectangle test as ``perception.verify``. Alerts in mock mode are shown
and **never sent** — every card is stamped "MOCK — not sent".

**Real runs.** ``--follow PATH`` tails a JSONL event file instead of scripting one. To get
that file from an actual ``run_order`` execution without editing it (see ``src.gui.emit``)::

    .venv/bin/python -m src.agent_service.run_order --verify --verify-fail 1 2>&1 \
      | .venv/bin/python -m src.gui.emit --out run-events.jsonl
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # so `python tools/dashboard.py` works from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from src.gui import emit, frames as F, mock  # noqa: E402
from src.gui.app import DashboardApp  # noqa: E402
from src.gui.modes import MODES  # noqa: E402
from src.gui.platform import RUNTIMES, PlatformIdentity, PlatformLink  # noqa: E402
from src.gui.server import make_server, serve_in_thread  # noqa: E402

DEFAULT_PORT = 8090


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tools/dashboard.py",
        description="Live demo dashboard for the order-driven zero-shot picking cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  order   the order names the destination bin\n"
            "  qc      no bin in the order — the VLM grades the item and the traffic\n"
            "          light routes it (green=pass / orange=review / red=reject)\n"
            "  fusion  the order names a bin AND a QC check runs; a defect diverts to\n"
            "          red and raises an alert explaining why\n"
        ),
    )
    src = p.add_argument_group("data source")
    src.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        default=None,
        help="Canned run, no network, no hardware (DEFAULT).",
    )
    src.add_argument(
        "--follow",
        metavar="PATH",
        help="Tail a JSONL event file from a real run instead of scripting one "
        "(see `python -m src.gui.emit --help`). Implies --no-mock.",
    )
    src.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Wait for events on POST /events instead of scripting a run.",
    )
    src.add_argument(
        "--from-end",
        action="store_true",
        help="--follow only: skip events already in the file, show only new ones.",
    )

    demo = p.add_argument_group("demo")
    demo.add_argument("--mode", choices=MODES, default="order", help="Starting mode.")
    demo.add_argument(
        "--outcome",
        choices=("auto", *mock.OUTCOMES),
        default="auto",
        help="Mock ending: auto rotates verified → mismatch+retry → ghost-arm (default).",
    )
    demo.add_argument("--speed", type=float, default=1.0, help="Mock playback speed (1.0 = real).")
    demo.add_argument("--once", action="store_true", help="Play the mock run once, then hold.")

    plat = p.add_argument_group("platform")
    plat.add_argument(
        "--runtime",
        choices=[r.lower() for r in RUNTIMES],
        default=None,
        help="Header badge + alert source_type: mock (nothing sent), simulation, live. "
        "Defaults to mock with --mock, simulation otherwise.",
    )
    plat.add_argument(
        "--dry-run-alerts",
        action="store_true",
        help="Connect and show the real identity, but PRINT alerts instead of creating "
        "them (rehearse without filling the platform's alert view).",
    )

    view = p.add_argument_group("view")
    view.add_argument("--view", metavar="X,Y,W,H", help="Crop of the frame to serve.")
    view.add_argument("--full-view", action="store_true", help="Serve the whole frame.")
    view.add_argument(
        "--frame",
        metavar="PATH",
        help="--follow only: a real capture to use as the backdrop until the run pushes one.",
    )

    net = p.add_argument_group("server")
    net.add_argument("--host", default="127.0.0.1")
    net.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Default {DEFAULT_PORT}.")
    net.add_argument("--open", action="store_true", help="Open a browser at the page.")
    net.add_argument("--verbose", action="store_true", help="Log every HTTP request.")
    net.add_argument(
        "--self-test",
        action="store_true",
        help="Start, exercise every route, print a report and exit (no browser needed).",
    )
    return p


def resolve_view(args: argparse.Namespace) -> tuple[F.View | None, bool]:
    """``(explicit view, auto_view)`` — an explicit --view always wins."""
    if args.view:
        return F.parse_view(args.view), False
    return None, not args.full_view


def build_app(args: argparse.Namespace) -> DashboardApp:
    is_mock = args.mock
    runtime = (args.runtime or ("mock" if is_mock else "simulation")).upper()
    if is_mock and runtime != "MOCK":
        # Nothing may imply a mock alert was real. A scripted run stays MOCK, full stop.
        print("[dashboard] --mock forces runtime MOCK (a canned run never sends alerts).")
        runtime = "MOCK"

    view, auto = resolve_view(args)
    link = PlatformLink(
        runtime=runtime,
        enabled=runtime != "MOCK",
        identity=PlatformIdentity.from_env(),
        dry_run=bool(args.dry_run_alerts),
    )
    app = DashboardApp(
        mode=args.mode,
        source="mock" if is_mock else "live",
        view=view,
        auto_view=auto,
        platform=link,
    )
    link.start()  # returns immediately; connects on a background thread
    return app


def attach_source(app: DashboardApp, args: argparse.Namespace) -> threading.Event:
    """Wire up whichever producer feeds the dashboard. Returns its stop event."""
    stop = threading.Event()
    if args.mock:
        driver = mock.MockDriver(
            app, mode=args.mode, outcome=args.outcome, loop=not args.once, speed=args.speed
        )
        app.driver = driver
        driver.start()
        return stop

    if args.frame:
        try:
            app.frames.set_path(args.frame, token="backdrop")
            print(f"[dashboard] backdrop: {args.frame}")
        except Exception as e:  # noqa: BLE001 — a bad backdrop must not stop the server
            print(f"[dashboard] ⚠ could not read --frame {args.frame}: {e}")
    elif args.follow:
        found = F.find_real_frame()
        if found is not None:
            try:
                app.frames.set_path(found, token="backdrop")
                print(f"[dashboard] backdrop (newest capture): {found}")
            except Exception:  # noqa: BLE001
                pass

    if args.follow:
        emit.follow_in_thread(app, args.follow, stop=stop, from_start=not args.from_end)
        print(f"[dashboard] following events: {args.follow}")
    return stop


def banner(app: DashboardApp, args: argparse.Namespace, url: str) -> None:
    plat = app.platform.to_dict()
    ident = plat["identity"]
    source = "MOCK SCRIPT (no network, no hardware)" if args.mock else (
        f"tailing {args.follow}" if args.follow else "waiting on POST /events"
    )
    alerts = (
        "shown only, NEVER sent (MOCK)"
        if not app.platform.enabled
        else (
            f"DRY RUN — printed, not created (source_type={plat['source_type']})"
            if app.platform.dry_run
            else f"sent via robot.alerts.create (source_type={plat['source_type']})"
        )
    )
    zones = (
        "bin-regions.json (surveyed)" if app.regions_calibrated else "FALLBACK — not surveyed"
    )
    print(
        "\n".join(
            [
                "",
                "  ┌─ Order-Driven Zero-Shot Picking Cell — demo dashboard",
                f"  │  page      {url}",
                f"  │  mode      {app.state.mode}   ({', '.join(MODES)} — switchable in the UI)",
                f"  │  source    {source}",
                f"  │  runtime   {plat['runtime']}",
                f"  │  env id    {ident['environment_id'] or '— (CYBERWAVE_ENVIRONMENT_ID unset)'}",
                f"  │  twin id   {ident['twin_id'] or '— (CYBERWAVE_TWIN_ID unset)'}",
                f"  │  alerts    {alerts}",
                f"  │  zones     {zones}",
                "  │  safety    viewer only — never moves the robot",
                "  └─ ctrl-c to stop",
                "",
            ]
        ),
        flush=True,
    )


def self_test(app: DashboardApp, args: argparse.Namespace) -> int:
    """Start the server, exercise every route, report, exit. Used to verify a build."""
    import json
    import os
    import time
    import urllib.request

    server = make_server(app, host=args.host, port=0, quiet=True)
    host, port = server.server_address[0], server.server_address[1]
    serve_in_thread(server)
    base = f"http://{host}:{port}"
    stop = attach_source(app, args)
    failures: list[str] = []

    def get(path: str) -> tuple[int, bytes, str]:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")

    def post(path: str, body: dict) -> tuple[int, bytes]:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"\n[self-test] {base}\n")
    status, html, ctype = get("/")
    check("GET /", status == 200 and b"<!doctype html>" in html[:40].lower(), f"{len(html)} bytes")
    check("page is self-contained", b"http://" not in html and b"https://" not in html,
          "no absolute URL anywhere in the document")
    status, blob, _ = get("/health")
    check("GET /health", status == 200 and json.loads(blob)["status"] == "ok")
    status, blob, ctype = get("/frame.jpg")
    check("GET /frame.jpg", status == 200 and ctype == "image/jpeg", f"{len(blob)} bytes")
    status, blob, _ = get("/state")
    doc = json.loads(blob)
    check("GET /state", status == 200 and doc["stages"][0]["name"] == "ORDER")
    check("motion panel has PICK + PLACE", [m["stage"] for m in doc["motion"]] == ["PICK", "PLACE"])
    check("four motion states in the legend", len(doc["motion_legend"]) == 4,
          ", ".join(m["label"] for m in doc["motion_legend"]))
    ident = doc["platform"]["identity"]
    check("platform identity in the header",
          {"environment_id", "twin_id"} <= set(ident),
          f"env={ident['environment_id']} twin={ident['twin_id']}")
    key = os.getenv("CYBERWAVE_API_KEY") or ""
    check("API key never rendered",
          ident["api_key"] in (True, False) and (not key or key not in blob.decode()),
          "only a present/absent boolean is exposed")

    # Assert on the POST's own response, NOT on a following GET /state: when --mock is
    # running, its script emits a `reset` carrying the mock's mode (src/gui/mock.py) on
    # every loop, so a reset landing between the POST and the GET reverts `state.mode`
    # and the check fails — observed roughly 1 run in 4. That is a race in the check,
    # not a defect in set_mode, and re-reading shared state that a live producer is
    # concurrently writing can never be made reliable. The POST response is the
    # authoritative result of the call under test.
    status, blob = post("/mode", {"mode": "fusion"})
    check("POST /mode", status == 200 and json.loads(blob)["mode"] == "fusion")
    post("/mode", {"mode": args.mode})
    status, blob = post("/events", {"kind": "log", "text": "self-test"})
    check("POST /events", status == 200 and json.loads(blob)["applied"] == 1)

    if args.mock:
        print("\n[self-test] watching the scripted run advance through every stage …")
        # Alerts and outcomes are cleared by the next run's reset, so accumulate
        # everything as it goes by rather than sampling the final state.
        seen: set[str] = set()
        motion_states: set[str] = set()
        outcomes: set[str] = set()
        alerts: dict[str, bool] = {}
        deadline = time.time() + 180
        while time.time() < deadline and not (
            len(seen) >= 7
            and {"commanded", "verified", "mismatch"} <= motion_states
            and alerts
            and outcomes
        ):
            doc = json.loads(get("/state")[1])
            for s in doc["stages"]:
                if s["status"] != "idle":
                    seen.add(s["name"])
            for m in doc["motion"]:
                if m["state"]:
                    motion_states.add(m["state"])
            for a in doc["alerts"]:
                alerts[a["name"]] = bool(a.get("platform", {}).get("mock"))
            if doc["outcome"] != "running":
                outcomes.add(doc["outcome"])
            time.sleep(0.2)
        check("every stage lit up", len(seen) == 7, " ".join(sorted(seen)))
        check("COMMANDED / VERIFIED / MISMATCH all rendered",
              {"commanded", "verified", "mismatch"} <= motion_states,
              " ".join(sorted(motion_states)))
        check("a run reached an outcome", bool(outcomes), " ".join(sorted(outcomes)) or "none")
        check("alerts were raised and every one is stamped MOCK — not sent",
              bool(alerts) and all(alerts.values()), f"{len(alerts)} alert(s)")
        check("nothing was sent to the platform",
              json.loads(get("/state")[1])["platform"]["sent"] == 0)

    stop.set()
    if app.driver is not None:
        app.driver.stop()
    server.shutdown()
    verdict = "ALL PASS" if not failures else f"{len(failures)} FAILED: {', '.join(failures)}"
    print(f"\n[self-test] {verdict}\n")
    return 0 if not failures else 1


def resolve_source(args: argparse.Namespace) -> argparse.Namespace:
    """``--mock`` is the default; ``--follow`` opts out of it. Mutates and returns args."""
    if args.mock is None:
        args.mock = not args.follow
    if args.follow and args.mock:
        args.mock = False  # a real event file always wins over the canned script
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_source(build_parser().parse_args(argv))
    app = build_app(args)
    if args.self_test:
        return self_test(app, args)

    try:
        server = make_server(app, host=args.host, port=args.port, quiet=not args.verbose)
    except OSError as e:
        print(f"[dashboard] cannot bind {args.host}:{args.port} — {e}", file=sys.stderr)
        print("[dashboard] another dashboard already running? try --port 8091", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{server.server_address[1]}/"
    stop = attach_source(app, args)
    banner(app, args, url)
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping …")
    finally:
        stop.set()
        if app.driver is not None:
            app.driver.stop()
        app.platform.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
