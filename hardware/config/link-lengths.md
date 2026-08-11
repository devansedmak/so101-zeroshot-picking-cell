# SO-101 link lengths — measurement sheet

**Status: PLACEHOLDER.** [src/control/ik.py](../../src/control/ik.py) ships SO-ARM100-ballpark
values, not measured ones. The FK∘IK round-trip is self-consistent either way, so the *code* is
correct — but the mapping from a table `(x, y)` in mm to real joint angles is only as accurate as
these four numbers. Wrong lengths ⇒ the gripper reaches short/long by that error.

The assembled-version vendor manual lists no link lengths (O9), and
chasing an answer on Discord is no longer worth it. **Just measure with a ruler** — 5 minutes.

## The four numbers (mm)

Measure **pivot-to-pivot** (centre of one servo horn axis to the centre of the next), arm posed
straight out horizontally so each segment is easy to lay a ruler along. ±3 mm is fine.

| Constant in `ik.py` | What to measure | Placeholder |
|---|---|---|
| `L1_SHOULDER_TO_ELBOW_MM` | shoulder-lift pivot (J2) → elbow pivot (J3) | 116.0 |
| `L2_ELBOW_TO_WRIST_MM` | elbow pivot (J3) → wrist-flex pivot (J4) | 135.0 |
| `L3_WRIST_TO_TIP_MM` | wrist-flex pivot (J4) → **gripper contact point**, jaws closed | 100.0 |
| `BASE_HEIGHT_MM` | **table surface** → shoulder-lift pivot (J2) height | 120.0 |

Notes:
- `BASE_HEIGHT_MM` is measured from the table the arm is *clamped to* — if the arm gets re-mounted
  or shimmed, this changes. Re-measure if the arm moves.
- `L3` is a tool length: measure to where the jaws actually grip an object, not to the jaw tips.
- The base-pan pivot (J1) is assumed coaxial with the shoulder column — no length needed.

## After measuring

1. Edit the four constants in [src/control/ik.py](../../src/control/ik.py) (they are module-level,
   nothing else to change) and delete the `# TODO(verify)` block above them.
2. Re-run the suite — the FK∘IK round-trip test must stay green with the new numbers:
   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_ik.py
   ```
   (If `Unreachable` starts firing for demo table points, the reach envelope shrank — move the mat
   closer to the arm rather than fighting the solver.)
3. Sanity-check in **simulation** before any live motion: pick a known table point, run the loop
   with `--sim`, and confirm the twin's gripper lands where you expect.

## Frame convention (also assumed, also worth a sanity check)

`ik.py` assumes table coordinates share the arm base's origin and axis alignment — i.e. the
homography's world frame is the robot base frame. If the calibration sheet's origin is *not* at the
arm base, the offset must be applied when converting homography output → IK input. Cheapest fix:
**place the calibration sheet's origin corner at the arm base**, so the two frames coincide by
construction. Otherwise record the offset here and subtract it. Related: `# TODO(calibration)` in
`ik.py`, [cameras.md](cameras.md).
