# Order-Driven Zero-Shot Picking Cell (SO-101 × Cyberwave)

A tabletop picking cell where **the item name is the only configuration**. A warehouse order arrives over
HTTP — `{"order_id": "SO-1042", "item": "red marker", "bin": "A"}` — a hosted vision-language model *points*
at that item in an overhead frame, a calibrated planar homography turns the pixel into table millimetres,
analytic IK turns the millimetres into joint angles, and after the move the cell **re-looks** to check the
item actually landed in the target bin.

Built solo for the **Cyberwave Builders Program — Cohort 2** by Devan Sedmak. Hardware: SO-101 follower arm
+ one fixed overhead USB camera + an Ubuntu laptop as the edge node.

**The problem.** Small fulfilment cells gain SKUs constantly, and classic pick-and-place answers every new
SKU with per-item engineering: a fixture, a template, a labelled dataset, a retrained policy. That cost is
fine for one SKU a million times and absurd for a shelf that gains three products a week — so those cells
stay manual. Adding a SKU here is a new string in an order: no retraining, no code change. And because the
loop closes on the *outcome* rather than on the command, a pick that silently failed is reported as failed
instead of being assumed fulfilled.

**Demo video: landing here within 24 h of submission.** *(placeholder — link goes here)*

---

## Architecture

```
POST /orders ──▶ order_api ──▶ loop.fulfill_order
 (mock WMS)                          │
        1 PERCEIVE   capture.py       one overhead still (OpenCV → ffmpeg → twin frame)
        2 DETECT     detect.py        hosted VLM, structured_task="detect_points" → pixel
        3 PROJECT    homography.py    pixel → table mm (DLT + Hartley normalisation)
                     orient.py        item elongation → grasp yaw
        4 SOLVE      ik.py            table mm → joint angles (base pan + 2-link, tool vertical)
        5 ACT        motion.py        validate → per-joint clamp → 20 Hz ramp
                     sim_session.py / live_session.py    ← cw.affect("simulation") | ("live")
        6 VERIFY     verify.py        re-capture, VLM points again, point-in-bin-region test
                                      fail → retry once → fail again → give up honestly
        7 ALERT      alerts.py        order_fulfilled (info) | order_failed (error)
```

| Step | Module | Notes |
|---|---|---|
| Order intake | [`src/agent_service/order_api.py`](src/agent_service/order_api.py), [`src/wms_mock/orders.py`](src/wms_mock/orders.py) | `POST /orders`, `GET /health`. Stdlib `http.server` on purpose — no web framework added days before ship. Fulfilment serialised behind a lock (one physical arm). |
| Orchestration | [`src/agent_service/loop.py`](src/agent_service/loop.py) | `fulfill_order`: pick → place → verify → retry once → alert. Never raises; always returns a `Fulfillment`. Verification is *injected*, so `verify=None` degrades to an open-loop skeleton. |
| Capture | [`src/perception/capture.py`](src/perception/capture.py) | One-shot V4L2 still, never a live stream. MJPEG is requested **before** the frame size (otherwise the driver silently falls back to 640×480 and every saved pixel coordinate is invalid). |
| Zero-shot detection | [`src/perception/detect.py`](src/perception/detect.py), [`src/perception/locate.py`](src/perception/locate.py) | `cw.mlmodels.run(..., structured_task="detect_points")`; output is `[y, x]` on a **0–1000** grid, decoded in exactly one place. `locate.py` composes capture → detect → homography. |
| Geometry | [`src/perception/homography.py`](src/perception/homography.py), [`src/perception/orient.py`](src/perception/orient.py) | Planar pixel→table homography with a reprojection-error ship bar; grasp yaw from the item's principal axis so the jaws close across a marker, not along it. |
| Kinematics | [`src/control/ik.py`](src/control/ik.py) | Analytic base-pan + 2-link planar IK, elbow-up, tool held vertical. Link lengths taken from the official SO-101 URDF, not a ruler. `Unreachable` outside the envelope. |
| Motion | [`src/control/motion.py`](src/control/motion.py) | `MotionExecutor`: allow-listed actions, all-or-nothing plan validation, **per-joint clamp immediately before the SDK call**, 20 Hz ramped interpolation, `dry_run` as the default. |
| Sim / live boundary | [`src/control/sim_session.py`](src/control/sim_session.py), [`src/control/live_session.py`](src/control/live_session.py) | One `cw.affect()` call separates them. `live_session.verify_pose()` reads the encoders back and **raises when telemetry is missing** — "cannot tell" must never render as "fine". Driver telemetry is radians, the repo speaks degrees; the conversion lives at that one boundary. |
| Verification | [`src/perception/verify.py`](src/perception/verify.py) | Same VLM call, then a pure point-in-rectangle test against surveyed bin regions (calibration data, not perception). |
| Alerts | [`src/agent_service/alerts.py`](src/agent_service/alerts.py) | `robot.alerts.create(...)`, `category="business"`, types `order_fulfilled` / `order_failed`. |
| Demo dashboard | [`tools/dashboard.py`](tools/dashboard.py), [`src/gui/`](src/gui/) | One self-contained page, inline CSS/JS, no CDN. A pure viewer: never imports `live_session`, never opens a serial port. |

Everything left of `cw.affect` is plain Python and unit-tested offline. Calibration inputs live in
[`hardware/config/`](hardware/config/) (`homography.json`, `bin-regions.json`, `joint-ranges.md`).

## Run it without hardware

This is also the demo fallback: no robot, no camera, no network, no API credits.

```bash
python3 -m venv .venv && .venv/bin/pip install "cyberwave[camera]" numpy pillow python-dotenv pytest

# 1. The dashboard — mock is the DEFAULT. Serves http://127.0.0.1:8090
.venv/bin/python tools/dashboard.py
.venv/bin/python tools/dashboard.py --mode qc          # traffic-light QC triage view
.venv/bin/python tools/dashboard.py --mode fusion      # order + QC together
.venv/bin/python tools/dashboard.py --self-test        # exercises every route and the whole run, then exits

# 2. The test suite (⚠ a sourced ROS Humble leaks a broken pytest plugin — hence the flag)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q     # -> 395 passed

# 3. The agent loop, dry-run: order -> pick -> place -> verify -> alert
.venv/bin/python -m src.agent_service.run_order --verify                  # verified -> fulfilled
.venv/bin/python -m src.agent_service.run_order --verify --verify-fail 1  # fails once -> retry -> fulfilled
.venv/bin/python -m src.agent_service.run_order --verify --verify-fail 2  # fails twice -> failed + ERROR alert

# 4. Order-driven: the receiver, then a POSTed order
.venv/bin/python -m src.agent_service.order_api --verify --verify-fail 1
curl -sS -X POST http://127.0.0.1:8080/orders -H 'Content-Type: application/json' \
     -d '{"order_id": "SO-1042", "item": "red marker", "bin": "A"}'
#  -> {"status": "fulfilled", "stages": ["picked","placed","verify-failed","retried",
#      "picked","placed","verified","alerted"], "attempts": 2, ...}

# 5. Calibration tools prove themselves with no camera and no display
.venv/bin/python tools/calibrate_homography.py --selftest
.venv/bin/python tools/pick_bin_regions.py --selftest
```

The dashboard renders motion in **four** states, never two: `COMMANDED` (sent, unconfirmed) /
`VERIFIED` (encoders agree) / `MISMATCH` (encoders disagree, per-joint degrees shown) / `UNVERIFIED`
(no telemetry — never green). That honesty is enforced in Python, not CSS: the event builder forces
`MISMATCH` whenever joint errors are present, so an optimistic producer cannot paint a bad move green.

## Run it on hardware

Full procedure, wiring, calibration order and troubleshooting: **[runbook.md](runbook.md)**.

> **Safety.** This repo drives a real arm. Live motion is never implicit: every motion subcommand of
> [`tools/live_check.py`](tools/live_check.py) refuses to run without `--yes`, prints exactly what it is
> about to do, and then waits for the operator to type `go`. Steps are never batched — one invocation,
> one small motion. In code: allow-listed actions, all-or-nothing validation, a per-joint clamp applied
> immediately before every SDK call, ramped interpolation instead of step commands, and dry-run as the
> default for every entrypoint. Hardware: **leader PSU 5 V/6 A, follower 12 V/8 A — never swapped**,
> max payload 400 g, E-stop is Ctrl+C or cutting follower power.

```bash
.venv/bin/python tools/live_check.py read              # connect + read encoders. MOVES NOTHING.
.venv/bin/python tools/live_check.py gripper --yes     # gripper only, then type: go
.venv/bin/python tools/live_check.py hover --yes       # hover 5 cm above a printed mark
```

## Status — what is actually true

Scrupulously honest; this table is the point.

| Capability | Status | Evidence / what is missing |
|---|---|---|
| Pixel→table homography, calibrated on the real cell | ✅ **verified on hardware** | 5 clicked marks on a printed A4 target → **rms 0.35 mm, max 0.56 mm** (ship bar 20 mm). [`hardware/config/homography.json`](hardware/config/homography.json). The tool refuses to save a fit worse than the bar. |
| Bin regions surveyed from a real frame | ✅ verified on hardware | Zones A/B/C in [`hardware/config/bin-regions.json`](hardware/config/bin-regions.json), 1920×1080. |
| Live arm reached; commanded pose confirmed by the servos | ✅ **verified on hardware** | `live_session.verify_pose()` read the encoders back; the arm tracked a 4-joint commanded pose inside the 5° tolerance (observed error ≤ 2°). |
| `wrist_flex` sign error found and fixed **against the real arm** | ✅ verified on hardware | The servos tracked the command to within half a degree while the gripper pointed *up*: the model was wrong, not the hardware. Tool heading is **`q2 + q3 − q4`**, not `q2 + q3 + q4`. Fixed in both `forward_kinematics` and `solve_ik`; one test pins the convention (the FK∘IK round-trip is blind to it — both functions flipped together). |
| Gripper open/close convention | ✅ verified on hardware | Measured on the follower: jaws touching read 6.1°, a 110→60→10→110° sweep confirmed **high = open, low = shut** on the calibrated 0→128.9° span. The previous `GRIPPER_OPEN = −40°` was outside the span entirely. |
| Jaw mechanical offset (`JAW_OFFSET_DEG`) | ⚠ **assumed 0°, never measured** | The rotation-invariant grasp derives wrist roll from the item's long axis, but the constant offset between the wrist-roll zero and the jaw-closing line is still a placeholder. A wrong value biases *every* oriented grasp by the same amount — a systematic error with no scatter to make it visible, and one indistinguishable from a bias in the perception axis estimate. The measurement procedure is written out in [`src/control/ik.py`](src/control/ik.py); it needs one overhead photo of the open jaws in a lit room. |
| Camera path: MJPEG-before-size, 1920×1080, focus | ✅ verified on hardware | Silent 640×480 fallback fixed for both backends, with a loud warning if it recurs. Scale ≈ 0.91 mm/px. |
| Order → pick → place → alert on the twin, in **simulation** | ✅ simulated | Ran end-to-end against the project's SO-101 twin; alert dispatched. |
| Closed verification loop: verify → retry once → ERROR alert | ✅ simulated / offline, all three branches | The three `run_order --verify[--verify-fail N]` commands above, plus unit tests. |
| HTTP order intake driving the loop | ✅ offline | A real `curl` returns the `stages` list above; tests cover 200/409/422/413/405/404/500 and concurrent orders. |
| Motion safety: validation, per-joint clamp, ramping, command scope | ✅ offline | `tests/test_motion.py`, including clamp-before-SDK ordering and a `{"1","2"}` pose that must never write to joint 6. |
| Analytic IK, link lengths from the official URDF | ✅ offline | FK∘IK round-trip over a reachable grid. `L3` was corrected 100 → 159.8 mm — the old value would have driven the gripper ~60 mm *into* the table on the first live pick. |
| Demo dashboard (mock, QC, fusion; real-run bridge) | ✅ offline | `--self-test` exercises every route; real runs bridge via `python -m src.gui.emit --out run-events.jsonl` + `--follow`, with zero edits to `run_order.py`. |
| **Full order → pick → place loop on the live arm** | ⬜ **not done** | Sim on the real twin, yes. The complete loop on the physical follower has never been run. |
| Grasp force / width | ⬜ not modelled | One fixed close angle for every object; no force sensing, no width control. Open-loop grasp. |
| A real platform alert reaching Cyberwave | ✅ **verified against the platform** | `alerts.send_alert()` was run once for real against the project twin and the platform returned alert `1aa82d7f-8ed6-478b-94e0-2d022607bbbd` — so the dispatch path, the kwarg set and the severity/category vocabulary are confirmed end to end, not just mocked. The `order_fulfilled` / `order_failed` payloads themselves are still only exercised in `--dry-run-alerts`, deliberately, to keep synthetic order events out of the live alerts view. |
| Cyberwave **Workflow** triggering | ⬜ not built | Cyberwave SDK **0.5.3 exposes no `client.workflows`**, and no workflow slug was invented to paper over that. The HTTP receiver exists and works; nothing on the platform points at it. |
| Bin reachability | 🔴 **known geometry problem** | The three zone centres project to **r = 259 / 262 / 246 mm** from the shoulder axis against a vertical-gripper limit of **246.5 mm** — only one of the three is (marginally) reachable. The fix is a few-centimetre slide of the mat toward the base plus a re-run of `pick_bin_regions.py`, not a redesign; it needs hardware access. |
| IK zeros vs real servo zeros | ⚠ unresolved | The URDF's link vectors carry perpendicular offsets, and `ik.py` assumes the shoulder pivot is coaxial with the pan axis (it is 30.4 mm forward, 18.3 mm lateral) — a systematic error of roughly 35 mm. The correction is derived but not implemented. The FK∘IK test proves *internal* consistency only. |
| Real VLM API call from this code | 🟡 model validated, spend deferred | The model was validated on real photos in the Cyberwave Playground (free); the scripted path was proven offline against canned responses because per-call credit cost is not published. |
| VLA / SmolVLA policy | ❌ dropped | Server-side training issues, 6–7 h per run, and training seizes the twin the demo needs. |

Two things were learned the expensive way and are each pinned by exactly one test: the VLM's `[y, x]`/0–1000
output order, and the twin exposing joints as `_1`..`_6` rather than `1`..`6`. Both fail silently, not
loudly, when you get them wrong.

## Tests

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q   ->  395 passed
```

No test needs a network, a robot, or a credit.

## Repo layout

| Path | What |
|---|---|
| `src/perception/` | `capture` · `detect` · `locate` · `homography` · `orient` · `verify` |
| `src/control/` | `ik` · `motion` · `sim_session` · `live_session` · `hello_sim` |
| `src/agent_service/` | `loop` · `order_api` · `run_order` · `poses` · `alerts` |
| `src/gui/` | Demo dashboard internals (events, modes, pages, the `emit` bridge) |
| `src/wms_mock/` | Order value object + mock order source |
| `tools/` | `dashboard.py` · `live_check.py` · `calibrate_homography.py` · `pick_bin_regions.py` · `make_calibration_sheet.py` |
| `hardware/config/` | Calibration artifacts and notes: homography, bin regions, joint ranges, link lengths, ports, cameras |
| `docs/` | Demo scenario, demo script, video script, SO-101 vendor notes |
| `runbook.md` | Architecture, every command, safety, troubleshooting |
| `roadmap.md` | Dated plan |

Some source docstrings cite the builder's private working notes (`decisions.md`, `progress.md`,
`CLAUDE.md`) for the reasoning behind a choice. Those files are deliberately not published; the code and
this README are self-contained without them.

## License

MIT — see [LICENSE](LICENSE).
