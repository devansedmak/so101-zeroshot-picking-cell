# Reuse map: github.com/devansedmak/vision-language-pick-and-place-manipulation

Old stack: BERT NER + GroundingDINO + PyBullet, Franka Panda, simulation only.

## Reuse (port with light changes)

- **BERT NER target extraction**: command -> object phrase. Port as `src/perception/ner.py`; add unit tests (pure text, no hardware).
- **GroundingDINO prompting**: prompt construction, confidence thresholds, bbox postprocess. Port as `src/perception/grounding.py`. Re-tune thresholds on real overhead frames (lighting is not the same as sim).
- **Command -> structured task schema** (object, target location), which maps directly to the order payload `{"item", "bin"}`.

## Do NOT reuse

- **IK / motion code**: Franka is 7-DOF torque-controlled; SO-101 is 5-DOF + gripper with hobby servos. Write SO-101-specific `src/control/`, following the nl_arm_controller clamp+ramp pattern.
- **PyBullet env & camera model**: replaced by Cyberwave sim/live twins and a real calibrated homography.
- **Depth-based 3D localization** (if present): we use planar homography instead; no depth sensor.

## New pieces (no old equivalent)

- Pixel -> table homography calibration + projection (`src/perception/homography.py`, pytest-covered).
- Cyberwave integration: twins, `capture_frame`, `joints.set`, alerts, workflow webhook (`src/agent_service/`, `src/wms_mock/`).
- Feedback/verification step (re-capture + presence check).
