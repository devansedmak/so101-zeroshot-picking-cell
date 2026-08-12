# Calibration notes: SO-101 (both arms)

Done 2026-07-15 (Session 3) via `sudo cyberwave pair` → guided dashboard calibration.
Both arms **calibrated + uploaded to twin `aa1dd0ad-...`**. Stored server-side (twin)
and in-container `setup.json` (`follower_port`, `leader_port`). Edge Core is a
systemd service (`cyberwave-edge-core.service`, auto-start on boot).

## Flow that worked
1. Driver auto-detects on boot: `SO101 ports: leader=/dev/ttyACM0, follower=/dev/ttyACM1`
   (by **voltage detect**, which matches our by-serial map in `ports.md`).
2. On "calibration missing", driver auto-starts guided calibration → prompts appear in
   **dashboard** (SO-101 twin), 2 steps per arm.
3. **Step 1/2 (zero position):** hand-move arm (torque released) to the folded rest pose
   shown in the photo; click Next. Sets each joint's 0.
4. **Step 2/2 (range of motion):** hand-sweep **all 6 joints to their stops**; click finish.
5. On success: torque enables, follower **snaps to home** (abrupt, no ramp, which is normal).

## Gotchas hit (avoid next time)
- ❌ **Gripper (ID 6) not recorded** → "span < 5% of full range". Must **fully open + close
  the gripper** during Step 2/2. Easy to forget.
- ❌ **wrist_roll (ID 5) wrapped** → recorded `[0, 4090]` "not physically possible". Cause:
  wrist_roll zero landed on the encoder 0↔4095 seam. Fix: keep **wrist_roll centered/neutral
  at the zero step**, and in ROM move it to natural limits only (don't spin multiple turns).
  Live logs show each joint's span; watch that wrist_roll is a *partial* number, not ~4090.
- **Re-trigger a clean run:** dashboard "Restart calibration" did **not** re-arm capture;
  `sudo systemctl restart cyberwave-edge-core.service` re-runs the missing-calibration flow
  from Step 1/2. Calibration is non-destructive + re-runnable anytime.
- Transient `Failed to resolve api.cyberwave.com` seen once (DNS blip) → blocks upload;
  just retry once network is back.

## Torque behavior (normal)
- **Follower** (12V, ttyACM1): torque **ON** after calib → snaps to home, holds position. Keep clear.
- **Leader** (5V, ttyACM0): torque **OFF** (back-driveable) → stays where you place it. Required for teleop.

## Re-calibrate if
Twin pose and physical arm visibly disagree (accumulated offset), or an arm is re-seated.
Just restart the edge service and redo the 2-step flow per arm.
