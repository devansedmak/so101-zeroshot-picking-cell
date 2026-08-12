# Cameras

## Physical camera (kit)
WowRobo SO-ARM101 kit camera = **2MP USB Camera Module, 30 FPS, 3 m USB cable**
(plug-and-play UVC). **Not** a Pi Camera v3 (that's CSI-only). Confirmed 2026-07-15 by
USB enumeration + vendor spec + decision D11.

| Property | Value |
|---|---|
| USB ID | `05a3:9230` "ARC International Camera" (UVC) |
| Device nodes | `/dev/video4` (stream) + `/dev/video5` (metadata) |
| Driver sees it as | `USB2.0_CAM1` (via v4l2-ctl inside so101-driver container) |
| Built-in laptop cam (ignore) | Acer Integrated RGB `5986:215d` -> `/dev/video0-3` |

Note: host lacks `v4l-utils`, so host-side `cyberwave edge cameras` can't enumerate.
Install if needed: `sudo apt-get install v4l-utils`. (The driver container already has it.)

## Twin
- **Overhead Camera** (Standard Camera, `cyberwave/standard-cam`):
  twin uuid `056c8c62-9368-4ae9-b582-ca3998f1cf90`, sensor `color_camera` (rgb).
  Default sensor params 1280×720 @ 30 fps, FOV 58.7° (physical cam is 2MP/1080p,
  bump twin resolution later only if it matters; homography measures real optics).
- Old **Pi-cameraV3** twin (`6f63d45d...`) **deleted** 2026-07-15 (wrong asset type).

## Bring-up sequence (~15 min, mechanical, run it in this order)

All tooling was proven offline on 2026-08-10 (`--selftest` on both tools, `pytest -q` green),
so nothing below should need debugging. Run from the repo root. Nothing here moves the robot.

**Step 0: prep before touching hardware (5 min, no camera needed).**
Print or draw a sheet with 4 clearly marked corners at known spacing and tape it **flat**
on the mat: A4 landscape = **297 × 210 mm**, A3 = 420 × 297 mm. Put its **origin corner at
the arm base**, long edge pointing away from the arm, and then the homography world frame IS
the IK base frame (origin under the base pivot, +X away from the arm, +Y to its left) and no
offset maths is needed. See [link-lengths.md](link-lengths.md) §Frame convention.
Sanity-check the tools with no hardware:
```bash
.venv/bin/python tools/calibrate_homography.py --selftest && .venv/bin/python tools/pick_bin_regions.py --selftest
```
→ both must end in `SELFTEST PASSED`.

**Step 1: plug the camera in and confirm the right node.**
```bash
.venv/bin/python -m src.perception.capture
```
Expect a device list plus `[capture] auto-selected: /dev/videoN` where N is **not** an
`Integrated RGB Camera` node (those are the laptop lid cam, always skipped).
*Fails* (`no external camera found`): another USB port, wait 2 s, re-run; then `dmesg | tail`.

**Step 2: mount the camera rigidly overhead, lens cap off.**
Fixed relative to the table, looking straight down, whole mat + all bins + arm base in view.
It must not move again after step 4; if it gets bumped, redo steps 4-7.

**Step 3: grab one still frame.**
```bash
.venv/bin/python -m src.perception.capture --save /tmp/frame.jpg
```
→ expect `saved → /tmp/frame.jpg (NNNNN bytes)`. Open it: mat, all 3 bins and the arm base
must be visible and in focus. *Black frame*: add `-w 20`. *Wrong camera*: `-d /dev/videoN`.
*cv2 cannot open the node*: `--backend ffmpeg`.

**Step 4: tape the calibration sheet down and re-grab THE frame.**
```bash
.venv/bin/python -m src.perception.capture --save /tmp/calib.jpg
```
Steps 5 and 7 both use this one file, so their pixel coordinates refer to the same camera pose.

**Step 5: calibrate the pixel -> table homography.**
```bash
.venv/bin/python tools/calibrate_homography.py --image /tmp/calib.jpg --sheet 297,210
```
Click the 4 sheet corners in this order: origin (arm base), +X, diagonal, +Y. Then **ENTER**
(`u` undoes, `ESC` aborts). With `--sheet` you type no millimetres.
→ expect per-point errors, then `PASS` and `saved → hardware/config/homography.json`.
*No window*: the tool pre-checks the display in a throwaway subprocess (Qt would otherwise
abort the whole process) and prints the fallback for you: read the 4 corner pixels off any
image viewer and use the keyboard-only path (identical code path):
```bash
.venv/bin/python tools/calibrate_homography.py --pixels "812,455" "1180,470" "1195,690" "800,672" --sheet 297,210
```
`--help` documents `--points "px,py=X,Y" ...` and `--points-file` too.

**Step 6: check the < 2 cm ship bar.** The tool enforces it: `max` reprojection error must be
≤ **20 mm** or it prints `*** FAIL ***` and **refuses to write the file**. Do not use `--force`.
*On FAIL*, in likelihood order: a mistyped mm value, corners clicked in the wrong order, the
sheet/camera moved between clicking and measuring, the mat is not flat. Re-click (step 5);
adding 2 extra spread-out points (`--points`) usually fixes a marginal fit.

**Step 7: survey the bin pixel rectangles (this is what closes the loop).**
```bash
.venv/bin/python tools/pick_bin_regions.py --image /tmp/calib.jpg
```
Click **2 opposite corners per bin**, bins in order **A, B** (4 clicks; bin C has no taught place
pose, so add `C` only if you teach one), just *inside* each
rim. A box slightly smaller than the bin mouth is safer than one that spills onto the table.
Expect `saved -> hardware/config/bin-regions.json` and `round-trip OK via load_bin_regions()`.
Keyboard-only path: `--regions "A=x0,y0,x1,y1" "B=..." "C=..." --frame-size W,H`.
Labels must stay `A`/`B`/`C`; that is what the order source posts.

**Step 8: measure the 4 IK link lengths** with a ruler and edit `src/control/ik.py`:
full procedure in [link-lengths.md](link-lengths.md) (5 min, ±3 mm is fine).

**Step 9: prove it and commit.**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
git add hardware/config/homography.json hardware/config/bin-regions.json src/control/ik.py
```
Then dry-run the loop in **simulation** before any live motion.

### Optional, only if the dashboard needs the real feed
The "Overhead Camera" live view shows the **virtual/sim render** (twin mesh from above) until a
streamer runs. Not required for the demo loop; `capture_still()` is local:
```bash
cyberwave camera -c <index> -t 056c8c62-9368-4ae9-b582-ca3998f1cf90 -e 9821dd80-9596-4eee-a572-254b254e7ab0
```
(clones a `cyberwave-edge-python` streamer project; `-c` camera index, `-f` fps.) Do not start it
before the mount is final, and stop it before capturing stills; it holds the device open.
