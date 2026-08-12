"""Shared LIVE connect + warm-up -> a ready :class:`MotionExecutor`. MOVES REAL HARDWARE.

The live twin of :mod:`sim_session`: same connect/joint-discovery/warm-up sequence,
but pinned to ``cw.affect("live")`` so commands reach the physical follower arm via
the edge driver (``cyberwave edge status`` must show the driver up).

SAFETY: importing this module moves nothing; only calling
:func:`open_live_executor` does, and callers must pass ``i_understand_this_moves_real_hardware=True``
so a live session can never be opened by accident or by a stray import. Every
entrypoint that uses it must ask the operator for explicit confirmation first and
must never batch live motions.

Why a separate module rather than a flag on ``sim_session``: a boolean parameter on
the sim path would make "did this move the real arm?" depend on a call site far from
the code being read. A distinct import is greppable: ``grep -rn live_session src/``
enumerates every place that can move hardware.
"""

from __future__ import annotations

import math
import time

from .motion import JOINTS, MotionExecutor


def open_live_executor(
    twin_id: str,
    *,
    settle: float = 3.0,
    i_understand_this_moves_real_hardware: bool = False,
) -> tuple[MotionExecutor, object]:
    """Connect to ``twin_id`` in **LIVE** mode and return ``(executor, robot)``.

    Mirrors :func:`sim_session.open_live_executor`'s contract. Raises ``RuntimeError``
    unless the caller explicitly acknowledges that this drives the physical arm.

    Unlike the sim path this does **not** snap the arm to all-zeros as a warm-up: on
    real hardware that would be a large, unannounced full-body move from whatever
    pose the arm is currently resting in. The controller is warmed up by re-commanding
    the arm's *current* pose instead, which is a no-op move.
    """
    if not i_understand_this_moves_real_hardware:
        raise RuntimeError(
            "refusing to open a LIVE session without an explicit acknowledgement; "
            "pass i_understand_this_moves_real_hardware=True"
        )

    from cyberwave import Cyberwave

    cw = Cyberwave()
    cw.affect("live")  # real hardware: the physical follower arm will move
    print(f"[live] ⚠ runtime = LIVE (real robot); fetching twin {twin_id} …")

    robot = cw.twin(twin_id=twin_id)
    print(f"[live] twin ready: {getattr(robot, 'name', twin_id)}")

    name_map: dict[str, str] = {}
    try:
        from cyberwave.twin.capabilities import controllable_joint_names

        actual = controllable_joint_names(robot)
        if len(actual) == len(JOINTS):
            name_map = dict(zip(JOINTS, actual))
            print(f"[live] joint map: {name_map}")
    except Exception as e:  # noqa: BLE001, discovery is best-effort; fall back to identity
        print(f"[live] joint-name discovery skipped ({e}); using canonical names")

    executor = MotionExecutor(robot, name_map=name_map)

    if settle > 0:
        print(f"[live] settling {settle:.0f}s (no warm-up move on real hardware) …")
        time.sleep(settle)

    return executor, robot


def canonical_joints(state: dict[str, float], name_map: dict[str, str]) -> dict[str, float]:
    """Translate SDK joint names ("_1".."_6") -> canonical keys ("1".."6"), RADIANS -> DEGREES.

    The driver reports encoder positions in radians (visible in its own status table,
    e.g. ``shoulder_lift pos=1.04rad``) while everything above the SDK boundary in this
    repo speaks degrees. Confirmed 2026-08-11 against a commanded pose: commanded
    (11.89, 59.24, -62.00, -87.25)° read back as (0.20, 1.04, -1.05, -1.52) rad, which
    is the same pose to within half a degree. Converting here keeps the unit switch at
    the one boundary where it belongs.
    """
    inverse = {sdk: canon for canon, sdk in name_map.items()}
    return {inverse.get(name, name): math.degrees(value) for name, value in state.items()}


def sync_executor_pose(executor: object, robot: object) -> dict[str, float]:
    """Seed the executor's idea of "where we are" from the arm's ACTUAL joint angles.

    :class:`MotionExecutor` tracks ``_current_pose`` in memory and starts it at all
    zeros, so a freshly-opened session believes the arm is home no matter where it
    physically is. Ramping from that false start makes the first move a step command
    instead of a ramp: the arm snaps rather than eases. Reading the real angles first
    turns the next move back into a genuine interpolation.

    Returns the canonical pose it read (empty dict if telemetry was unavailable, in
    which case the caller should refuse to ramp).
    """
    state = canonical_joints(read_joints(robot), getattr(executor, "name_map", {}) or {})
    if state:
        executor._current_pose.update({j: v for j, v in state.items() if j in executor._current_pose})
    return state


def verify_pose(
    robot: object,
    expected: dict[str, float],
    *,
    name_map: dict[str, str] | None = None,
    tolerance_deg: float = 5.0,
) -> dict[str, float]:
    """Read the arm back and return {joint: error_deg} for joints outside ``tolerance_deg``.

    Closes a real gap: :meth:`MotionExecutor.execute` publishes to MQTT and prints
    "plan complete" with no acknowledgement from the servos, so a disconnected driver
    (or a clamped/rejected angle) looks exactly like success. On 2026-08-11 that cost
    a live session: every command "succeeded" while the arm never moved, because the
    driver had no serial binding. Comparing commanded vs measured is the only honest
    completion signal we have.

    An empty return means every joint landed within tolerance. A missing telemetry
    read raises, because "cannot tell" must never be reported as "fine".
    """
    measured = canonical_joints(read_joints(robot), name_map or {})
    if not measured:
        raise RuntimeError(
            "no joint telemetry — cannot confirm the arm actually moved. "
            "Check `docker logs` for the driver's serial binding."
        )
    drift: dict[str, float] = {}
    for joint, want in expected.items():
        if joint not in measured:
            continue
        error = measured[joint] - want
        if abs(error) > tolerance_deg:
            drift[joint] = error
    return drift


def read_joints(robot: object) -> dict[str, float]:
    """Best-effort read of the twin's current joint angles (degrees). Moves nothing.

    Returns ``{}`` if the SDK/driver exposes no readable state. An empty result is
    informative in itself (usually: servos unpowered, or telemetry not yet flowing),
    so callers should report it rather than treat it as an error.
    """
    joints = getattr(robot, "joints", None)
    if joints is None:
        return {}
    for method in ("get", "state", "positions", "read", "all"):
        fn = getattr(joints, method, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except Exception:  # noqa: BLE001, try the next accessor
            continue
        if isinstance(value, dict) and value:
            return {str(k): float(v) for k, v in value.items()}
    return {}
