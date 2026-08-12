# SO-101 joint ranges: measured, from OUR twin's calibration

Pulled 2026-08-10 from twin `aa1dd0ad-...` (`joint_calibration.follower`), i.e. the ranges recorded by
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
| 5 | wrist_roll | ±3.1332 | **±179.5°** | **±90 (deliberately narrow)** |
| 6 | gripper | 0 → 2.2494 | **0° → 128.9°** | 0 → 124 (**resolved**, see below) |

Leader ranges are within ~1° of the follower except wrist_roll (±167.8°) and gripper (0→104.6°);
irrelevant to us, since only the follower is commanded.

The limits keep a **~5° margin inside** the calibrated range: the calibrated value is where the servo
was measured to stop, so commanding it exactly means driving into the mechanical end.

**wrist_roll stays clamped well inside its ±179.5°**: its zero sits near the encoder 0..4095 seam and
it wrapped during calibration (see [calibration-notes.md](calibration-notes.md)), so the cheap safety
win is worth more than the last 90° of range.

### wrist_roll: ±60° → ±90° (rotation-invariant grasping)

It was ±60° while every pick was top-down with a fixed jaw direction. Oriented grasping
(`ik.solve_ik(..., axis_deg=...)`) needs roll: the jaws must close **across** the object's long axis, or
a long thin item (marker, pen) is grasped end-on and squirts out.

±90° is exactly the amount required, not a compromise: a parallel jaw is **symmetric under a 180°
roll** (swapping which finger is which grasps identically), so every possible object orientation folds
into `[-90, +90)`. At ±60° a 60° wedge of orientations was simply unpickable; with ±90° that dead
wedge disappears entirely, and `GraspAngleUnreachable` becomes unreachable-by-construction rather than
a failure mode we have to live with. The ±5° safety-margin reasoning above is untouched: ±90° is still
~90° clear of both the mechanical stop and the ±180° encoder seam that motivated the narrowing.

## Why this mattered

With the old guessed `±60°` on joints 2-4, a perceived pick in the **middle of the calibration sheet**
solved to `elbow = -83°`, got silently clamped to `-60°`, and the tip would have landed short of the
object. The arm can physically reach it (±96.6°); only our guess said otherwise. Found offline on
10 Aug by running `run_order --perceive` against a synthetic calibration, i.e. before it could waste a
hardware session. The executor still clamps every command; it just no longer clamps away real reach.

## RESOLVED 2026-08-11: gripper convention measured on the arm

**This was the last open risk before the live demo. It is closed.** The gripper was commanded on its
own (`tools/live_check.py gripper`, nothing else moving) and observed directly. Two independent
readings, both pointing the same way:

1. **Jaws pushed shut by hand** → the encoder read **0.107 rad = 6.1°**. So the *bottom* of the
   calibrated span is the *closed* end, and the jaws touch a few degrees above 0.
2. **Commanded sweep 110° → 60° → 10° → 110°** → observed (operator's words) *fully open* → *roughly
   half closed* → *almost fully shut* → *fully open again*.

So **LOW = SHUT, HIGH = OPEN**, mapped straight onto the follower's calibrated `0° to 128.9°`. Option 2
of the old hypothesis (a hidden driver offset/sign) is ruled out: the commanded angles behaved exactly
as the calibration table says.

Set accordingly:

| Constant | Was | Now | Why |
|---|---|---|---|
| `poses.GRIPPER_CLOSE` | +20° | **10°** | just above the 6.1° hard-shut reading; squeezes, doesn't stall into the stop |
| `poses.GRIPPER_OPEN` (mirrored in `control.ik`) | **-40°** | **105°** | -40° was *outside the servo's range entirely*; 105° is wide open, clear of the 128.9° end |
| `DEFAULT_JOINT_LIMITS["6"]` | ±60° | **0 to 124** | same ~5° margin as every other joint; ±60 was doubly wrong: it admitted negative angles the servo does not have, and capped "open" at 60°, i.e. *half closed* |

Note how bad the old values were on real hardware: `open = -40°` would have been clamped to the shut
end, so *every* "open the gripper" command (including the release over the drop bin) would have
closed the jaws instead.

### ⚠ Still not modelled: grasp force and width

`GRIPPER_CLOSE` is **one constant angle for every object**. There is no force/current feedback, no
per-object width, and no check that anything was actually grasped:

- a **thicker** item than ~the closing angle allows → the servo stalls against it and holds current;
- a **thinner** item → the jaws close past it and grip nothing, and the arm carries air to the bin.

Today this is covered only *after the fact*, by the visual verification step (`perception/verify.py`),
which notices the item never arrived. Closing it properly means either commanding a width per detected
object (the detector already gives a bounding box) or reading servo current/load to stop on contact;
neither is in scope before the demo. Keep the demo objects close in width to whatever `GRIPPER_CLOSE`
was tuned against.
