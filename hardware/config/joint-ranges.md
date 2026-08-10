# SO-101 joint ranges — measured, from OUR twin's calibration

Pulled 2026-08-10 from twin `aa1dd0ad-…` (`joint_calibration.follower`), i.e. the ranges recorded by
**our own guided calibration** on 15 Jul. Until now these numbers only existed server-side, and
`motion.py` used hand-guessed conservative limits instead. Recorded here so the repo is the source of
truth and a re-pair can be diffed against it.

## Follower (the arm that executes)

| Joint | Label | Calibrated range (rad) | = degrees | `DEFAULT_JOINT_LIMITS` now |
|---|---|---|---|---|
| 1 | shoulder_pan | ±2.0752 | **±118.9°** | ±113 |
| 2 | shoulder_lift | ±1.7937 | **±102.8°** | ±97 |
| 3 | elbow_flex | ±1.6863 | **±96.6°** | ±91 |
| 4 | wrist_flex | ±1.7729 | **±101.6°** | ±96 |
| 5 | wrist_roll | ±3.1332 | **±179.5°** | **±60 (deliberately narrow)** |
| 6 | gripper | 0 → 2.2494 | **0° → 128.9°** | ⚠ unchanged, see below |

Leader ranges are within ~1° of the follower except wrist_roll (±167.8°) and gripper (0→104.6°);
irrelevant to us, since only the follower is commanded.

The limits keep a **~5° margin inside** the calibrated range: the calibrated value is where the servo
was measured to stop, so commanding it exactly means driving into the mechanical end.
**wrist_roll stays clamped to ±60° on purpose** — its zero sits near the encoder 0↔4095 seam and it
wrapped during calibration (see [calibration-notes.md](calibration-notes.md)). We never need roll for
a top-down pick, so the cheap safety win is worth more than the range.

## Why this mattered

With the old guessed `±60°` on joints 2–4, a perceived pick in the **middle of the calibration sheet**
solved to `elbow = -83°`, got silently clamped to `-60°`, and the tip would have landed short of the
object. The arm can physically reach it (±96.6°) — only our guess said otherwise. Found offline on
10 Aug by running `run_order --perceive` against a synthetic calibration, i.e. before it could waste a
hardware session. The executor still clamps every command; it just no longer clamps away real reach.

## ⚠️ OPEN — gripper convention must be verified before the live demo

`src/agent_service/poses.py` commands the gripper **open at −40°** and closed at **+20°**, but the
follower's calibrated gripper range is **0° → 128.9°**. So `−40°` is **outside** the calibrated range
and will be clamped (to `0°`) or rejected on the real arm. One of these is true and we cannot tell
offline:

1. the driver maps our degrees straight onto that 0→128.9° span, in which case **open/closed must be
   re-expressed inside it** (e.g. open ≈ 110°, closed ≈ 10° — *direction unverified*); or
2. the driver applies its own offset/sign, in which case the current values may be fine.

`DEFAULT_JOINT_LIMITS["6"]` is therefore **left untouched** — changing it while the sign convention is
unknown could command a *closing* motion when we mean *opening*, with an object or fingers in the jaws.

**Tomorrow, before any full-loop live run** (~2 min, gripper only, nothing else moving):

```bash
# arms powered, workspace clear, NOTHING in the jaws
.venv/bin/python -m src.control.hello_sim --dry-run    # sanity: no connection
# then, live, one joint only — watch the physical gripper and note which way it goes:
#   command 6 → +20°, observe; command 6 → 0°, observe; command 6 → +110°, observe
```

Record what you see here, then set the open/closed constants in `poses.py` accordingly and re-run the
sim loop. Per [CLAUDE.md](../../CLAUDE.md) rule 2 this is live motion → explain + confirm first, and it
is the *only* live command that should run before the gripper direction is known.
