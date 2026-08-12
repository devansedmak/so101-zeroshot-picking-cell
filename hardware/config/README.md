# hardware/config

Store here (commit, no secrets):

- `ports.md`: which `/dev/tty*` is leader vs follower (record after first `sudo cyberwave pair`)
- `calibration-*.md/json`: per-arm calibration results/notes from the dashboard guided calibration
- `homography.npz` + `homography-notes.md`: overhead camera to table plane calibration (checkerboard corners, reprojection error, date; recalibrate if the camera moves)
- `cameras.md`: `sensor_id` mapping, wrist cam vs overhead cam
