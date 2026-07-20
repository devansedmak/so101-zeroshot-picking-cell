"""Shared SIMULATION connect + warm-up → a ready :class:`MotionExecutor`.

Factored out of hello_sim so every sim entrypoint (hello_sim, the agent loop's
run_order) connects the same, proven way: fetch the twin by ``CYBERWAVE_TWIN_ID``,
pin the runtime to **simulation**, discover the twin's real joint names (our SO-101
exposes ``_1``..``_6`` — see progress.md S4), warm up the controller, and hand back
an executor bound to that twin.

SAFETY (CLAUDE.md rule 2): SIMULATION-ONLY by construction — this hard-codes
``cw.affect("simulation")`` and offers no live path. Live hardware motion is a
separate, explicitly-confirmed step and is intentionally not wired here.
"""

from __future__ import annotations

import time

from .motion import JOINTS, MotionExecutor


def load_env() -> None:
    """Best-effort load of .env (python-dotenv is a project dependency)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # rely on the ambient environment


def open_sim_executor(twin_id: str, *, settle: float = 3.0) -> tuple[MotionExecutor, object]:
    """Connect to ``twin_id`` in SIMULATION and return ``(executor, robot)``.

    Warms up the controller (the first joint command auto-attaches it and can be
    dropped) and waits ``settle`` seconds so the first real plan lands cleanly.
    Imports the SDK lazily so callers' ``--dry-run`` paths need no credentials.
    """
    from cyberwave import Cyberwave

    cw = Cyberwave()
    cw.affect("simulation")  # pin runtime to SIM — do not remove without a confirmed live plan
    print(f"[sim] runtime = simulation; fetching twin {twin_id} …")

    robot = cw.twin(twin_id=twin_id)
    print(f"[sim] twin ready: {getattr(robot, 'name', twin_id)}")

    # Map canonical joint keys ("1".."6") → this twin's actual SDK joint names.
    name_map: dict[str, str] = {}
    try:
        from cyberwave.twin.capabilities import controllable_joint_names

        actual = controllable_joint_names(robot)
        if len(actual) == len(JOINTS):
            name_map = dict(zip(JOINTS, actual))
            print(f"[sim] joint map: {name_map}")
    except Exception as e:  # noqa: BLE001 — discovery is best-effort; fall back to identity
        print(f"[sim] joint-name discovery skipped ({e}); using canonical names")

    executor = MotionExecutor(robot, name_map=name_map)

    if settle > 0:
        print(f"[sim] warming up controller ({settle:.0f}s settle) …")
        try:
            executor._snap_to({j: 0.0 for j in JOINTS})
        except Exception as e:  # noqa: BLE001 — first command may be lost by design
            print(f"[sim] warm-up command note: {e}")
        time.sleep(settle)

    return executor, robot
