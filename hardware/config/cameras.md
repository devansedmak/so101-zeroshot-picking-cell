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
| Built-in laptop cam (ignore) | Acer Integrated RGB `5986:215d` → `/dev/video0–3` |

Note: host lacks `v4l-utils`, so host-side `cyberwave edge cameras` can't enumerate.
Install if needed: `sudo apt-get install v4l-utils`. (The driver container already has it.)

## Twin
- **Overhead Camera** (Standard Camera, `cyberwave/standard-cam`) —
  twin uuid `056c8c62-9368-4ae9-b582-ca3998f1cf90`, sensor `color_camera` (rgb).
  Default sensor params 1280×720 @ 30 fps, FOV 58.7° (physical cam is 2MP/1080p —
  bump twin resolution later only if it matters; homography measures real optics).
- Old **Pi-cameraV3** twin (`6f63d45d…`) **deleted** 2026-07-15 (wrong asset type).

## TODO (next sessions — prerequisites, in order)
1. Remove lens cap; **mount camera rigidly overhead** (fixed relative to table — required
   for planar homography, decisions D5/D11).
2. Set up real streaming to the twin (do NOT do before mount + uncap):
   `cyberwave camera -c <index> -t 056c8c62-9368-4ae9-b582-ca3998f1cf90 -e 9821dd80-9596-4eee-a572-254b254e7ab0`
   (clones a `cyberwave-edge-python` streamer project; `-c` is the camera index, `-f` fps).
3. Calibrate **camera→table homography** (checkerboard); save `homography.npz` + notes.
   Ship bar: reprojection error < 2 cm (checkpoint criterion).

The dashboard "Overhead Camera" live view currently shows the **virtual/sim render**
(twin mesh from above), not the real feed — real feed appears after step 2.
