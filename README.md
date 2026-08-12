# Order-Driven Zero-Shot Picking Cell (SO-101 + Cyberwave)

A small warehouse picking cell where the name of the item is the only setup you need.

An order arrives over HTTP, for example `{"order_id": "SO-1042", "item": "red marker", "bin": "A"}`.
A hosted vision-language model points at that item in a photo from an overhead camera. A calibrated
homography turns that pixel into millimetres on the table. Analytic IK turns the millimetres into
joint angles. After the move the cell takes a second photo and checks that the item really landed in
the right bin.

Built solo for the Cyberwave Builders Program, Cohort 2, by Devan Sedmak. Hardware is one SO-101
follower arm, one fixed overhead USB camera, and an Ubuntu laptop as the edge node.

**Why it matters.** Small fulfilment cells add new products every week. Normal pick-and-place has to
be engineered for each new product, with a fixture or a template or a trained model, so most of these
cells stay manual. Here a new product is just a new string in an order. And because the cell checks
the result instead of trusting the command, a pick that quietly failed gets reported as failed.

**Demo video: coming here shortly.**

## How it works

```
POST /orders --> order_api --> loop.fulfill_order
 (mock WMS)                          |
        1 PERCEIVE   capture.py      one photo from the overhead camera
        2 DETECT     detect.py       hosted VLM, detect_points, gives a pixel
        3 PROJECT    homography.py   pixel -> table mm
                     orient.py       item long axis -> grasp angle
        4 SOLVE      ik.py           table mm -> joint angles
        5 ACT        motion.py       validate, clamp per joint, ramp at 20 Hz
                     sim_session.py / live_session.py
        6 VERIFY     verify.py       photo again, is the item inside the bin?
                                     if not, retry once, then give up honestly
        7 ALERT      alerts.py       order_fulfilled or order_failed
```

| Step | Module | What it does |
|---|---|---|
| Order intake | [`src/agent_service/order_api.py`](src/agent_service/order_api.py), [`src/wms_mock/orders.py`](src/wms_mock/orders.py) | `POST /orders` and `GET /health`, on the Python standard library server. No web framework was added days before the deadline. Orders are handled one at a time because there is one arm. |
| Orchestration | [`src/agent_service/loop.py`](src/agent_service/loop.py) | `fulfill_order` runs pick, place, verify, one retry, then the alert. It never raises, it always returns a `Fulfillment`. |
| Capture | [`src/perception/capture.py`](src/perception/capture.py) | One still frame, never a video stream. MJPEG has to be requested before the frame size, otherwise the driver quietly drops to 640x480 and every saved pixel coordinate is wrong. |
| Zero-shot detection | [`src/perception/detect.py`](src/perception/detect.py), [`src/perception/locate.py`](src/perception/locate.py) | `cw.mlmodels.run(..., structured_task="detect_points")`. The model answers `[y, x]` on a 0 to 1000 grid, decoded in exactly one place. |
| Geometry | [`src/perception/homography.py`](src/perception/homography.py), [`src/perception/orient.py`](src/perception/orient.py) | Pixel to table millimetres, with a reprojection error check that refuses a bad fit. Grasp angle comes from the long axis of the item so the jaws close across a marker instead of along it. |
| Kinematics | [`src/control/ik.py`](src/control/ik.py) | Base rotation plus a 2-link planar solve, elbow up, tool pointing down. Link lengths come from the official SO-101 URDF, not from a ruler. Raises `Unreachable` outside the working area. |
| Motion | [`src/control/motion.py`](src/control/motion.py) | Allowed actions only, whole plan validated before anything moves, per-joint limits applied right before the SDK call, smooth ramp instead of jumps, dry run by default. |
| Sim and live | [`src/control/sim_session.py`](src/control/sim_session.py), [`src/control/live_session.py`](src/control/live_session.py) | One `cw.affect()` call is the whole difference. `live_session.verify_pose()` reads the encoders back and raises if there is no telemetry, because "I cannot tell" must never show up as "fine". The driver speaks radians, the repo speaks degrees, and that conversion lives in one place. |
| Verification | [`src/perception/verify.py`](src/perception/verify.py) | Same model call, then a plain point-in-rectangle test against the measured bin corners. |
| Alerts | [`src/agent_service/alerts.py`](src/agent_service/alerts.py) | `robot.alerts.create(...)` with types `order_fulfilled` and `order_failed`. |
| Demo dashboard | [`tools/dashboard.py`](tools/dashboard.py), [`src/gui/`](src/gui/) | One self-contained page, no CDN. It only displays, it never imports `live_session` and never opens a serial port. |

Everything before `cw.affect` is plain Python and tested offline. Calibration data lives in
[`hardware/config/`](hardware/config/).

## Run it without hardware

This is also the backup plan for the demo. No robot, no camera, no network, no API credit.

```bash
python3 -m venv .venv && .venv/bin/pip install "cyberwave[camera]" numpy pillow python-dotenv pytest

# 1. The dashboard, mock data by default, on http://127.0.0.1:8090
.venv/bin/python tools/dashboard.py
.venv/bin/python tools/dashboard.py --mode qc          # traffic-light quality view
.venv/bin/python tools/dashboard.py --mode fusion      # orders and quality together
.venv/bin/python tools/dashboard.py --self-test        # hits every route, then exits

# 2. The tests (a sourced ROS Humble breaks pytest plugins, hence the flag)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

# 3. The agent loop, dry run: order, pick, place, verify, alert
.venv/bin/python -m src.agent_service.run_order --verify                  # verified, fulfilled
.venv/bin/python -m src.agent_service.run_order --verify --verify-fail 1  # fails once, retries, fulfilled
.venv/bin/python -m src.agent_service.run_order --verify --verify-fail 2  # fails twice, failed plus error alert

# 4. Driven by an order over HTTP
.venv/bin/python -m src.agent_service.order_api --verify --verify-fail 1
curl -sS -X POST http://127.0.0.1:8080/orders -H 'Content-Type: application/json' \
     -d '{"order_id": "SO-1042", "item": "red marker", "bin": "A"}'
#  -> {"status": "fulfilled", "stages": ["picked","placed","verify-failed","retried",
#      "picked","placed","verified","alerted"], "attempts": 2, ...}

# 5. The calibration tools check themselves with no camera and no screen
.venv/bin/python tools/calibrate_homography.py --selftest
.venv/bin/python tools/pick_bin_regions.py --selftest
```

The dashboard shows a move in four states, not two. `COMMANDED` means sent but not confirmed,
`VERIFIED` means the encoders agree, `MISMATCH` means they disagree and shows the error per joint,
and `UNVERIFIED` means there was no telemetry at all. `UNVERIFIED` is never green. That rule is
enforced in Python, not in CSS, so a hopeful producer cannot paint a bad move green.

## Run it on hardware

**Safety.** This code drives a real arm. Live motion is never implicit. Every motion command in
[`tools/live_check.py`](tools/live_check.py) refuses to run without `--yes`, prints exactly what it
is about to do, and then waits for you to type `go`. Steps are never batched, so one command means
one small motion. In the code itself, actions are allow-listed, the whole plan is validated before
anything moves, joint limits are applied right before every SDK call, motion is ramped, and every
entry point defaults to a dry run. On the bench, the leader arm uses a 5 V 6 A supply and the
follower uses 12 V 8 A, and they must never be swapped. Max payload is 400 g. To stop, press Ctrl+C
or cut power to the follower.

```bash
.venv/bin/python tools/live_check.py read              # connect and read encoders, MOVES NOTHING
.venv/bin/python tools/live_check.py gripper --yes     # gripper only, then type: go
.venv/bin/python tools/live_check.py hover --yes       # hover 5 cm above a printed mark
```

## What actually works

This table is honest on purpose.

| Capability | Status | Evidence, or what is missing |
|---|---|---|
| Pixel to table homography, calibrated on the real cell | works on hardware | 5 clicked marks on a printed A4 target, rms 0.35 mm and max 0.56 mm against a 20 mm limit. See [`hardware/config/homography.json`](hardware/config/homography.json). The tool refuses to save a worse fit. |
| Bin corners measured from a real frame | works on hardware | Zones A, B and C in [`hardware/config/bin-regions.json`](hardware/config/bin-regions.json). |
| Live arm reached, and the servos confirmed the pose | works on hardware | `live_session.verify_pose()` read the encoders back. The arm tracked a 4-joint pose to within 2 degrees. |
| Wrist sign error found and fixed against the real arm | works on hardware | The servos tracked the command to within half a degree while the gripper pointed up, so the model was wrong and not the hardware. Tool heading is `q2 + q3 - q4`. One test pins the convention, because the FK and IK round trip cannot see it (both were flipped together). |
| Gripper open and close convention | works on hardware | Measured on the follower. Jaws touching read 6.1 degrees, and a 110, 60, 10, 110 sweep confirmed that high is open and low is shut on the 0 to 128.9 degree span. The old value of -40 was outside the span. |
| Jaw offset (`JAW_OFFSET_DEG`) | assumed, never measured | The grasp angle comes from the long axis of the item, but the fixed offset between the wrist-roll zero and the line the jaws close along is still a placeholder. A wrong value tilts every grasp by the same amount, which is the kind of error that hides because there is no scatter. The procedure is written out in [`src/control/ik.py`](src/control/ik.py) and needs one overhead photo of the open jaws. |
| Camera path, MJPEG before size, 1920x1080 | works on hardware | The silent 640x480 fallback is fixed for both backends and now warns loudly if it comes back. Scale is about 0.91 mm per pixel. |
| Order, pick, place, alert on the twin in simulation | simulated | Ran end to end against the project twin, alert dispatched. |
| Verify, retry once, error alert | simulated and offline, all three branches | The three `run_order --verify` commands above, plus unit tests. |
| HTTP order intake driving the loop | offline | A real `curl` returns the stage list above. Tests cover 200, 409, 422, 413, 405, 404 and 500, and concurrent orders. |
| Motion safety: validation, limits, ramping, scope | offline | `tests/test_motion.py`, including the ordering of the clamp and a pose for joints 1 and 2 that must never write to joint 6. |
| Analytic IK with URDF link lengths | offline | FK and IK round trip over a grid of reachable points. `L3` was corrected from 100 to 159.8 mm, and the old value would have driven the gripper about 60 mm into the table on the first live pick. |
| Demo dashboard | offline | `--self-test` hits every route. Real runs feed it through `python -m src.gui.emit --out run-events.jsonl --follow`, with no changes to `run_order.py`. |
| A real alert reaching the Cyberwave platform | verified against the platform | `alerts.send_alert()` was run once for real against the project twin and the platform returned an alert id, so the dispatch path and the field names are confirmed and not just mocked. The `order_fulfilled` and `order_failed` payloads are still only run with `--dry-run-alerts`, on purpose, to keep fake orders out of the live alert view. |
| Full order, pick and place loop on the live arm | not done | Simulation on the real twin, yes. The whole loop on the physical arm has never been run. |
| Grasp force and width | not modelled | One fixed closing angle for every object. No force sensing, no width control. |
| Cyberwave Workflow triggering | not built | Cyberwave SDK 0.5.3 has no `client.workflows`, and no workflow name was invented to hide that. The HTTP receiver works, but nothing on the platform calls it. |
| Bin reachability | diagnosed, fix computed, not applied | The three zone centres sit at 259.0, 262.1 and 246.4 mm from the base rotation axis, against a working band of 171.1 to 246.6 mm, so only zone C reaches and only by 0.2 mm. [`tools/check_reach.py`](tools/check_reach.py) reproduces this and solves it. Sliding the mat 3.0 cm toward the base puts all three at 233.8, 232.2 and 217.2 mm with at least 12.8 mm of margin. The number is chosen so it still works after the shoulder offset below is corrected. Needs a minute at the bench plus a re-run of `pick_bin_regions.py`. |
| IK zeros against the real servo zeros | unresolved | `ik.py` assumes the shoulder pivot is on the base rotation axis, and it is not. It sits 30.4 mm forward and 18.3 mm sideways, worth roughly 35 mm of systematic error. The correction is worked out but not implemented. The FK and IK test only proves the code agrees with itself. |
| Real VLM call from this code | model validated, spend deferred | The model was tested on real photos in the Cyberwave Playground, which is free. The scripted path was proven offline against saved responses, because the per-call credit cost is not published. |
| VLA / SmolVLA policy | dropped | Server-side training problems, 6 to 7 hours per run, and training takes over the twin that the demo needs. |

Two things were learned the hard way and each has exactly one test holding it in place. The model
answers `[y, x]` on a 0 to 1000 grid, and the twin exposes joints as `_1` to `_6` instead of `1` to
`6`. Both fail quietly rather than loudly when you get them wrong.

## Tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

No test needs a network, a robot, or a credit.

## Layout

| Path | What is in it |
|---|---|
| `src/perception/` | capture, detect, locate, homography, orient, verify |
| `src/control/` | ik, motion, sim_session, live_session, hello_sim |
| `src/agent_service/` | loop, order_api, run_order, poses, alerts |
| `src/gui/` | Dashboard internals: events, modes, pages, and the emit bridge |
| `src/wms_mock/` | Order object and a mock order source |
| `tools/` | dashboard, live_check, calibrate_homography, pick_bin_regions, check_reach, make_calibration_sheet |
| `hardware/config/` | Calibration files and notes: homography, bin regions, joint ranges, link lengths, ports, cameras |

## License

MIT, see [LICENSE](LICENSE).
