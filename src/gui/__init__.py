"""Demo dashboard — makes the picking cell's pipeline legible on camera.

A single self-contained HTML page served by a **stdlib** ``http.server`` (same call as
``agent_service.order_api``, D16: no FastAPI, no CDN — the demo laptop must work with
its network cable pulled out). Nothing in this package imports the SDK, the camera or
``control.live_session``; it is a *viewer*, it never moves the robot.

Modules:
    modes    — the three interchangeable routing policies (order / qc / fusion)
    events   — the event vocabulary + the state machine the page renders
    frames   — the overhead frame the overlays sit on (real jpg if one exists, else cv2)
    mock     — canned, convincing runs that drive the whole UI with no hardware
    platform — the fail-soft Cyberwave link (identity in the header, real alerts)
    app      — state + frames + platform behind one lock; ``snapshot()`` is the page
    page     — the single self-contained HTML document
    server   — the HTTP surface (page, /state, /frame.jpg, POST /events)
    emit     — the zero-touch bridge from a real ``run_order`` run (stdout → JSONL → here)

Entrypoint: ``tools/dashboard.py``.
"""

from __future__ import annotations

__all__ = ["app", "emit", "events", "frames", "mock", "modes", "page", "platform", "server"]
