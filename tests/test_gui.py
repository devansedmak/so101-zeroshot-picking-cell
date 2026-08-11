"""Unit tests for the demo dashboard (src/gui) — everything except the pixels.

The page itself is deliberately dumb: it draws ``DashboardApp.snapshot()`` and nothing
else, so all the behaviour worth testing is Python. Covered here:

* **mode routing** — the one thing order/qc/fusion disagree about, including the fusion
  diversion and the alert it raises;
* **the event state machine** — folding, derived stage timing, and the guarantee that a
  malformed event is dropped rather than fatal;
* **the four motion states** — the honesty rule this dashboard exists for: a commanded
  move must never be able to render as a verified one (2026-08-11 ghost-arm session);
* **JSON shapes** — the keys the page indexes into, and the overlay geometry;
* **the run_order bridge** — real stdout → events, with no import of run_order at all;
* **the HTTP surface** — on a real socket, since that is how the demo is served.

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook). No network, no camera, no SDK.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from src.gui import emit, frames as F
from src.gui.app import DashboardApp
from src.gui.events import (
    MOTION_STAGES,
    MOTION_STATES,
    OUTCOMES,
    STAGES,
    DashboardState,
    motion_event,
    motion_report,
    motion_summary,
    stage_event,
)
from src.gui.mock import MOTION_PLAN, SCENARIOS, build_script
from src.gui.modes import (
    GRADES,
    MODES,
    ZONES,
    ModeError,
    QCVerdict,
    check_mode,
    names_bin,
    needs_qc,
    route,
)
from src.gui.page import render_page
from src.gui.platform import SOURCE_TYPE_FOR, PlatformIdentity, PlatformLink
from src.gui.server import make_server, parse_events, serve_in_thread


# --------------------------------------------------------------- mode routing
def test_modes_are_the_three_advertised_ones():
    assert MODES == ("order", "qc", "fusion")
    assert [check_mode(m) for m in (" ORDER ", "Qc", "fusion")] == ["order", "qc", "fusion"]
    with pytest.raises(ModeError):
        check_mode("nope")


def test_zones_are_a_traffic_light_over_the_real_bin_labels():
    assert [z.color for z in ZONES.values()] == ["green", "orange", "red"]
    assert [z.grade for z in ZONES.values()] == ["pass", "review", "reject"]
    assert set(ZONES) == {"A", "B", "C"}  # the labels bin-regions.json already carries


def test_mode_declares_what_it_needs():
    assert (needs_qc("order"), names_bin("order")) == (False, True)
    assert (needs_qc("qc"), names_bin("qc")) == (True, False)
    assert (needs_qc("fusion"), names_bin("fusion")) == (True, True)


def test_order_mode_routes_to_the_ordered_bin():
    r = route("order", item="red marker", order_bin="b")
    assert (r.bin, r.requested_bin, r.diverted, r.qc, r.alert) == ("B", "B", False, None, None)
    assert r.zone.color == "orange"


def test_order_mode_needs_a_bin():
    with pytest.raises(ModeError):
        route("order", item="red marker", order_bin=None)


@pytest.mark.parametrize(
    "grade,expect",
    [("pass", "A"), ("review", "B"), ("reject", "C")],
)
def test_qc_mode_routes_by_the_traffic_light_and_ignores_the_order(grade, expect):
    r = route("qc", item="eraser", order_bin="A", qc=QCVerdict(grade))
    assert r.bin == expect
    assert r.requested_bin is None  # qc mode: the order named no bin
    assert not r.diverted


def test_qc_mode_needs_a_verdict():
    with pytest.raises(ModeError):
        route("qc", item="eraser")


def test_fusion_honours_the_ordered_bin_when_qc_passes():
    r = route("fusion", item="red marker", order_bin="A", qc=QCVerdict("pass"))
    assert (r.bin, r.diverted, r.alert) == ("A", False, None)


def test_fusion_diverts_a_defect_to_red_and_explains_why():
    qc = QCVerdict("reject", "cracked barrel", 0.91)
    r = route("fusion", item="blue marker", order_bin="A", qc=qc)
    assert (r.bin, r.requested_bin, r.diverted) == ("C", "A", True)
    assert r.alert is not None
    # The alert must use the surface agent_service.alerts verified: description, not message.
    assert set(r.alert) <= {"name", "description", "alert_type", "severity", "category"}
    assert r.alert["severity"] == "error"
    assert "cracked barrel" in r.alert["description"]
    assert "bin A" in r.alert["description"] and "zone C" in r.alert["description"]


def test_fusion_review_diverts_to_orange_with_a_warning():
    r = route("fusion", item="eraser", order_bin="A", qc=QCVerdict("review", "scuffed label"))
    assert (r.bin, r.diverted) == ("B", True)
    assert r.alert["severity"] == "warning"


def test_fusion_pass_into_the_same_zone_raises_no_alert():
    # Ordered to C and graded reject → destination unchanged, so nothing was "diverted".
    r = route("fusion", item="eraser", order_bin="C", qc=QCVerdict("reject", "torn"))
    assert (r.bin, r.diverted, r.alert) == ("C", False, None)


def test_qc_verdict_rejects_an_unknown_grade():
    with pytest.raises(ModeError):
        QCVerdict("maybe")
    assert set(GRADES) == {"pass", "review", "reject"}


# ------------------------------------------------------- the four motion states
def test_there_are_exactly_four_motion_states():
    assert MOTION_STATES == ("commanded", "verified", "mismatch", "unverified")
    assert MOTION_STAGES == ("PICK", "PLACE")


@pytest.mark.parametrize("state,tone", [
    ("commanded", "warn"), ("verified", "good"), ("mismatch", "bad"), ("unverified", "warn"),
])
def test_each_motion_state_has_its_own_tone_and_only_verified_is_green(state, tone):
    r = motion_report("PICK", state)
    assert r["tone"] == tone
    assert r["confirmed"] is (state == "verified")


def test_commanded_is_never_confirmed():
    """The bug: 'sent' looked like 'landed'. It must not be representable as confirmed."""
    r = motion_report("PICK", "commanded")
    assert r["confirmed"] is False
    assert r["tone"] != "good"
    assert r["label"] == "COMMANDED"


def test_unverified_is_a_warning_never_a_success():
    r = motion_report("PLACE", "unverified")
    assert r["confirmed"] is False
    assert r["tone"] == "warn"  # never "good": no telemetry means we cannot tell


def test_encoder_errors_force_mismatch_even_if_the_producer_claimed_verified():
    """An optimistic caller must not be able to paint a bad move green."""
    r = motion_report("PLACE", "verified", errors={"elbow": 7.4})
    assert r["state"] == "mismatch"
    assert r["confirmed"] is False
    assert r["errors"] == {"elbow": 7.4}
    assert r["max_error_deg"] == 7.4


def test_motion_report_rejects_an_unknown_state():
    with pytest.raises(ValueError):
        motion_report("PICK", "probably-fine")


def test_motion_report_drops_unparseable_joint_errors():
    r = motion_report("PICK", "mismatch", errors={"elbow": 7.4, "wrist": "n/a"})
    assert r["errors"] == {"elbow": 7.4}


def test_motion_summary_shows_per_joint_degrees_worst_first():
    assert motion_summary({"elbow": 7.4, "wrist_flex": -9.1}) == "wrist_flex -9.1° · elbow +7.4°"


def test_motion_panel_distinguishes_never_commanded_from_the_four_states():
    st = DashboardState()
    rows = {r["stage"]: r for r in st.motion_panel()}
    assert rows["PICK"]["state"] is None  # an empty slot is a fifth, distinct look
    assert rows["PICK"]["tone"] == "idle"
    st.apply(motion_event("PICK", "commanded"))
    assert st.motion_panel()[0]["state"] == "commanded"


def test_recommanding_a_stage_clears_its_previous_confirmation():
    """Attempt 2 must not inherit attempt 1's VERIFIED badge."""
    st = DashboardState()
    st.apply(stage_event("PICK", "active"))
    st.apply(motion_event("PICK", "verified"))
    assert st.stages["PICK"].motion["state"] == "verified"
    st.apply(stage_event("PICK", "active", attempt=2))
    assert st.stages["PICK"].motion is None


def test_a_motion_event_for_a_non_stage_is_dropped_not_fatal():
    st = DashboardState()
    assert st.apply(motion_event("WIGGLE", "verified")) is False
    assert st.dropped == 1


# --------------------------------------------------------- event state machine
def test_pipeline_is_the_seven_advertised_stages_in_order():
    assert STAGES == ("ORDER", "DETECT", "LOCATE", "PICK", "PLACE", "VERIFY", "ALERT")


def test_activating_a_stage_closes_the_earlier_ones_and_derives_their_timing():
    st = DashboardState()
    st.apply({"kind": "stage", "stage": "ORDER", "status": "active", "t": 100.0})
    st.apply({"kind": "stage", "stage": "DETECT", "status": "active", "t": 101.5})
    assert st.stages["ORDER"].status == "done"
    assert st.stages["ORDER"].ms == pytest.approx(1500.0)
    assert st.stages["DETECT"].status == "active"


def test_an_explicit_done_records_its_own_duration():
    st = DashboardState()
    st.apply({"kind": "stage", "stage": "PICK", "status": "active", "t": 10.0})
    st.apply({"kind": "stage", "stage": "PICK", "status": "done", "t": 12.25})
    assert st.stages["PICK"].ms == pytest.approx(2250.0)


def test_unknown_and_malformed_events_are_counted_never_raised():
    st = DashboardState()
    for bad in ({"kind": "nonsense"}, {}, {"kind": "stage", "stage": "NOPE"}, "not a dict", None):
        assert st.apply(bad) is False  # type: ignore[arg-type]
    assert st.dropped == 5
    assert st.seq == 0


def test_reset_starts_a_clean_run_and_bumps_the_run_id():
    st = DashboardState()
    st.apply({"kind": "order", "order_id": "SO-1", "item": "eraser", "bin": "A"})
    st.apply({"kind": "alert", "name": "x"})
    before = st.run_id
    st.apply({"kind": "reset", "mode": "qc"})
    assert (st.order, st.alerts, st.outcome, st.mode) == (None, [], "running", "qc")
    assert st.run_id == before + 1
    assert all(s.status == "idle" for s in st.stages.values())


def test_outcome_closes_every_still_active_stage():
    st = DashboardState()
    st.apply(stage_event("VERIFY", "active"))
    st.apply({"kind": "outcome", "status": "fulfilled", "detail": "done"})
    assert st.outcome == "fulfilled"
    assert st.stages["VERIFY"].status == "done"


def test_an_unknown_outcome_degrades_to_failed_never_to_success():
    st = DashboardState()
    st.apply({"kind": "outcome", "status": "probably-ok"})
    assert st.outcome == "failed"
    assert set(OUTCOMES) == {"running", "fulfilled", "retried", "failed"}


def test_the_log_is_bounded():
    st = DashboardState()
    st.log_limit = 10
    for i in range(50):
        st.apply({"kind": "log", "text": f"line {i}"})
    assert len(st.log) == 10
    assert st.log[-1]["text"] == "line 49"


def test_alerts_record_who_raised_them():
    st = DashboardState()
    st.apply({"kind": "alert", "name": "a"})
    st.apply({"kind": "alert", "name": "b", "origin": "run_order", "dispatched": False})
    assert st.alerts[0]["origin"] == "dashboard"
    assert st.alerts[1]["origin"] == "run_order"


# ----------------------------------------------------------------- JSON shapes
def test_snapshot_carries_every_key_the_page_indexes():
    doc = DashboardApp().snapshot()
    required = {
        "mode", "mode_blurb", "source", "stages", "order", "detection", "qc", "routing",
        "verification", "alerts", "outcome", "outcome_detail", "attempts", "zones",
        "motion", "motion_legend", "frame_token", "frame_size", "elapsed", "log",
        "run_id", "overlay", "platform", "view", "calibrated", "controls",
    }
    assert required <= set(doc)


def test_snapshot_is_json_serializable_from_the_first_paint():
    json.dumps(DashboardApp().snapshot())  # must not raise: no dataclasses, no None keys


def test_snapshot_stage_and_legend_shapes():
    doc = DashboardApp().snapshot()
    assert [s["name"] for s in doc["stages"]] == list(STAGES)
    assert [s["moves"] for s in doc["stages"]] == [n in MOTION_STAGES for n in STAGES]
    assert [m["state"] for m in doc["motion_legend"]] == list(MOTION_STATES)


def test_overlay_projects_zones_and_points_into_the_served_crop():
    app = DashboardApp()
    app.ingest([
        {"kind": "order", "order_id": "SO-1", "item": "red marker", "bin": "A"},
        {"kind": "detection", "pixel": [app.view.x + app.view.w / 2,
                                        app.view.y + app.view.h / 2], "label": "red marker"},
    ])
    ov = app.snapshot()["overlay"]
    assert ov["point"] == {"x": 0.5, "y": 0.5}  # centre of the crop
    assert {z["label"] for z in ov["zones"]} == {"A", "B", "C"}
    assert [z for z in ov["zones"] if z["active"]][0]["label"] == "A"
    assert all(-2 <= z["x"] <= 2 for z in ov["zones"])  # fractions, not pixels


def test_overlay_marks_the_verification_verdict_on_the_target_zone_only():
    app = DashboardApp()
    app.ingest([
        {"kind": "order", "order_id": "SO-1", "item": "eraser", "bin": "C"},
        {"kind": "verification", "ok": False, "reason": "outside", "point": [900, 500], "bin": "C"},
    ])
    zones = {z["label"]: z for z in app.snapshot()["overlay"]["zones"]}
    assert zones["C"]["verdict"] == "fail"
    assert zones["A"]["verdict"] is None and zones["B"]["verdict"] is None


def test_mode_can_be_switched_at_runtime():
    app = DashboardApp()
    assert app.set_mode("fusion") == "fusion"
    assert app.snapshot()["mode"] == "fusion"
    with pytest.raises(ModeError):
        app.set_mode("banana")


# ------------------------------------------------------------------- platform
def test_mock_mode_sends_nothing_and_says_so():
    app = DashboardApp(platform=PlatformLink(runtime="MOCK", enabled=False))
    app.ingest([{"kind": "alert", "name": "Order fulfilled", "severity": "info"}])
    alert = app.snapshot()["alerts"][0]
    assert alert["platform"] == {"dispatched": False, "mock": True, "detail": "MOCK — not sent"}
    assert app.snapshot()["platform"]["sent"] == 0


def test_an_alert_raised_by_the_run_itself_is_not_re_sent():
    """run_order already called robot.alerts.create — relaying would duplicate it."""
    sent: list[dict] = []
    link = PlatformLink(runtime="SIMULATION", enabled=True)
    link.send = lambda payload, done: sent.append(payload)  # type: ignore[assignment]
    app = DashboardApp(platform=link)
    app.ingest([{"kind": "alert", "name": "Order fulfilled", "origin": "run_order",
                 "dispatched": True}])
    assert sent == []
    assert app.snapshot()["alerts"][0]["platform"]["detail"] == "raised by run_order"


def test_source_type_follows_the_runtime():
    assert SOURCE_TYPE_FOR == {"MOCK": "simulation", "SIMULATION": "simulation", "LIVE": "edge"}
    assert PlatformLink(runtime="LIVE", enabled=True).source_type == "edge"
    assert PlatformLink(runtime="SIMULATION", enabled=True).source_type == "simulation"


def test_identity_exposes_the_api_key_only_as_a_boolean():
    ident = PlatformIdentity(environment_id="env-1", twin_id="twin-1", api_key=True).to_dict()
    assert ident["api_key"] is True
    assert "secret" not in json.dumps(ident).lower()
    assert set(ident) == {"environment_id", "twin_id", "twin_name", "api_key",
                          "dashboard_url", "workflow"}


def test_an_unknown_runtime_degrades_to_mock():
    assert PlatformLink(runtime="cowboy").runtime == "MOCK"


# ----------------------------------------------------------------- mock script
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_scripts_a_complete_run(mode):
    st = DashboardState()
    for beat in build_script(mode, "ok", SCENARIOS[mode][0]):
        st.apply_all(beat.events)
    doc = st.to_dict()
    assert doc["outcome"] == "fulfilled"
    assert st.dropped == 0  # the mock must never emit an event its own state machine drops
    assert all(s["status"] in ("done", "skipped") for s in doc["stages"])
    assert doc["routing"] is not None and doc["order"] is not None


def test_qc_mode_order_carries_no_bin_but_the_run_still_routes():
    st = DashboardState()
    for beat in build_script("qc", "ok", SCENARIOS["qc"][2]):  # the "reject" scenario
        st.apply_all(beat.events)
    assert st.order["bin"] is None
    assert st.routing["bin"] == "C"  # graded reject → the red zone
    assert st.qc["grade"] == "reject"


def test_fusion_mode_scripts_the_diversion_and_its_alert():
    st = DashboardState()
    for beat in build_script("fusion", "ok", SCENARIOS["fusion"][1]):  # cracked barrel
        st.apply_all(beat.events)
    assert st.routing["diverted"] is True
    assert st.routing["requested_bin"] == "A" and st.routing["bin"] == "C"
    assert any(a["alert_type"] == "qc_diverted" for a in st.alerts)


def test_the_ok_run_ends_with_both_moves_verified():
    st = DashboardState()
    for beat in build_script("order", "ok"):
        st.apply_all(beat.events)
    assert [m["state"] for m in st.motion_panel()] == ["verified", "verified"]
    assert st.outcome == "fulfilled"


def test_the_retry_run_actually_shows_a_mismatch_with_per_joint_degrees():
    seen: list[dict] = []
    st = DashboardState()
    for beat in build_script("order", "retry"):
        st.apply_all(beat.events)
        seen += [dict(st.stages[s].motion) for s in MOTION_STAGES if st.stages[s].motion]
    mismatch = [m for m in seen if m["state"] == "mismatch"]
    assert mismatch, "the retry script must render a MISMATCH"
    assert mismatch[0]["errors"] == {"elbow": 7.4, "wrist_flex": -6.1}
    assert st.outcome == "retried"
    assert any(a["alert_type"] == "motion_mismatch" for a in st.alerts)


def test_the_fail_run_shows_unverified_and_never_reports_success():
    st = DashboardState()
    for beat in build_script("order", "fail"):
        st.apply_all(beat.events)
    assert st.outcome == "failed"
    assert any(m["state"] == "unverified" for m in st.motion_panel())
    assert all(not m["confirmed"] for m in st.motion_panel())
    assert any(a["alert_type"] == "motion_unverified" for a in st.alerts)
    assert any(a["severity"] == "error" for a in st.alerts)


def test_the_three_canned_endings_cover_all_four_motion_states():
    """The whole point of scripting three endings: the video shows every state."""
    states = {s for plan in MOTION_PLAN.values() for (s, _e, _d) in plan.values()}
    assert states == {"verified", "mismatch", "unverified"}  # + commanded, always emitted first
    commanded = [
        e for beat in build_script("order", "ok") for e in beat.events
        if e["kind"] == "motion" and e["state"] == "commanded"
    ]
    assert len(commanded) == 2  # PICK and PLACE are each COMMANDED before being confirmed


def test_the_mock_uses_the_real_calibrated_zone_rectangles():
    rects = F.load_zone_rects()
    assert set(rects) == {"A", "B", "C"}
    st = DashboardState()
    for beat in build_script("order", "ok", SCENARIOS["order"][0]):
        st.apply_all(beat.events)
    x0, y0, x1, y1 = rects[st.routing["bin"]]
    px, py = st.verification["point"]
    assert x0 <= px <= x1 and y0 <= py <= y1  # verdict computed on the surveyed rectangle


# --------------------------------------------------- run_order → events bridge
#: Verbatim stdout of `python -m src.agent_service.run_order --verify --verify-fail 1`,
#: trimmed to the lines that carry meaning. Nothing here imports run_order.
RUN_ORDER_STDOUT = """\
[run_order] DRY RUN — no connection, no motion.
▶ order: pick 'red marker' → bin 'A'
  💬  picking red marker
  ▶  [1/3] home over 1.00s
  ✅  plan complete  (final pose: 1=+30.0°, 2=+25.0°)
  💬  placing in bin A
  ✅  plan complete  (final pose: 1=+0.0°, 2=+0.0°)
  👁 NOT verified: (stub) red marker not seen in bin A
  ↻ retry 2/2: (stub) red marker not seen in bin A
  💬  picking red marker
  ✅  plan complete  (final pose: 1=+30.0°)
  💬  placing in bin A
  ✅  plan complete  (final pose: 1=+0.0°)
  👁 verified: (stub) red marker seen inside bin A
  🔔  (stub) Order fulfilled: red marker → bin A
✅ fulfilled: red marker → bin A
"""


def _translate(text: str) -> tuple[DashboardState, list[dict]]:
    events = emit.RunOrderTranslator().feed_all(text.splitlines())
    st = DashboardState()
    st.apply_all(events)
    return st, events


def test_the_bridge_turns_a_real_run_into_a_complete_dashboard_run():
    st, _ = _translate(RUN_ORDER_STDOUT)
    assert st.source == "live"
    assert st.order == {"order_id": "run_order", "item": "red marker", "bin": "A"}
    assert st.outcome == "retried"
    assert st.attempts == 2
    assert st.verification["ok"] is True
    assert st.dropped == 0


def test_a_real_run_is_shown_COMMANDED_and_never_VERIFIED():
    """run_order has no encoder read-back, so nothing may render green. This is the bug."""
    st, _ = _translate(RUN_ORDER_STDOUT)
    assert [m["state"] for m in st.motion_panel()] == ["commanded", "commanded"]
    assert not any(m["confirmed"] for m in st.motion_panel())
    assert st.stages["PICK"].status == "done"  # the STAGE finished …
    assert st.stages["PICK"].motion["tone"] == "warn"  # … but the MOVE is unconfirmed


def test_the_bridge_marks_detect_and_locate_skipped_without_a_perceived_point():
    st, _ = _translate(RUN_ORDER_STDOUT)
    assert st.stages["DETECT"].status == "skipped"
    assert st.stages["LOCATE"].status == "skipped"


def test_the_bridge_reads_a_perceived_pick_off_the_locate_line():
    line = "[run_order] 👁 'red marker' at px=(1094, 22) → table (188, -130) mm, axis 12°"
    st, _ = _translate("▶ order: pick 'red marker' → bin 'A'\n" + line + "\n")
    assert st.detection["pixel"] == [1094.0, 22.0]
    assert st.detection["table_mm"] == [188.0, -130.0]
    assert st.detection["label"] == "red marker"
    assert st.stages["LOCATE"].status == "done"


def test_the_bridge_records_a_failed_run_and_its_error_alert():
    text = (
        "▶ order: pick 'eraser' → bin 'C'\n"
        "  💬  picking eraser\n"
        "  ✅  plan complete\n"
        "  👁 NOT verified: (stub) eraser not seen in bin C\n"
        "  🔔  alert dispatched: Order FAILED: eraser → bin C\n"
        "❌ failed: eraser → bin C  (verification failed after 2 attempt(s): not seen)\n"
    )
    st, _ = _translate(text)
    assert st.outcome == "failed"
    assert st.alerts[0]["severity"] == "error"
    assert st.alerts[0]["dispatched"] is True
    assert st.alerts[0]["origin"] == "run_order"  # dashboard must not re-send it
    assert st.stages["ALERT"].status == "failed"


def test_the_bridge_never_raises_on_junk():
    t = emit.RunOrderTranslator()
    for junk in ("", "   ", "▶ order: malformed", "🔔", "\x00\x01", "👁 verified:"):
        t.feed(junk)  # must not raise


def test_unmatched_lines_become_log_entries():
    events = emit.RunOrderTranslator().feed("[run_order] ⚠ perceived picking disabled: no file")
    assert [e["kind"] for e in events] == ["log"]
    assert "perceived picking disabled" in events[0]["text"]


def test_motion_step_lines_do_not_clear_a_commanded_badge():
    """A '[2/3] pose ...' line must not re-activate the stage and wipe its motion state."""
    st, _ = _translate(
        "▶ order: pick 'eraser' → bin 'A'\n"
        "  💬  picking eraser\n"
        "  ▶  [2/3] pose {1=+30.0°} over 1.50s\n"
    )
    assert st.stages["PICK"].motion["state"] == "commanded"


def test_a_producer_with_telemetry_can_upgrade_a_move_via_the_same_file(tmp_path):
    """The JSONL is the whole API: verify_pose output slots straight in."""
    path = tmp_path / "events.jsonl"
    with emit.EventLog(path) as log:
        log.write_all(emit.RunOrderTranslator().feed_all(RUN_ORDER_STDOUT.splitlines()))
        log.write(motion_event("PLACE", "mismatch", errors={"elbow": 7.4}))
    st = DashboardState()
    st.apply_all(emit.read_events(path))
    assert st.stages["PLACE"].motion["state"] == "mismatch"
    assert st.stages["PLACE"].motion["errors"] == {"elbow": 7.4}


def test_event_log_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    with emit.EventLog(path) as log:
        log.write({"kind": "log", "text": "hello"})
        log.write({"kind": "order", "item": "eraser", "bin": "A"})
    assert [e["kind"] for e in emit.read_events(path)] == ["log", "order"]


def test_bad_jsonl_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind": "log", "text": "ok"}\nnot json\n\n{"no": "kind"}\n[1,2]\n')
    assert [e["kind"] for e in emit.read_events(path)] == ["log"]


def test_read_events_on_a_missing_file_is_empty():
    assert emit.read_events("/nonexistent/never.jsonl") == []


def test_tail_lines_follows_a_growing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind": "log", "text": "first"}\n')
    stop = threading.Event()
    got: list[str] = []

    def reader() -> None:
        for line in emit.tail_lines(path, stop=stop, poll=0.02):
            got.append(line)
            if len(got) == 2:
                stop.set()
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for _ in range(200):
        if got:
            break
        threading.Event().wait(0.01)
    with path.open("a") as fh:
        fh.write('{"kind": "log", "text": "second"}\n')
    t.join(timeout=5)
    assert [json.loads(g)["text"] for g in got] == ["first", "second"]


# ------------------------------------------------------------------- HTTP surface
@pytest.fixture()
def server():
    """A real dashboard on a real socket — the demo is served, so test it served."""
    app = DashboardApp(platform=PlatformLink(runtime="MOCK", enabled=False))
    httpd = make_server(app, host="127.0.0.1", port=0, quiet=True)
    serve_in_thread(httpd)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield app, base
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read(), r.headers


def _post(url: str, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_the_page_is_one_self_contained_document(server):
    _, base = server
    status, html, headers = _get(base + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    text = html.decode()
    # No CDN, no font host, no external anything: the demo laptop has no network.
    assert "http://" not in text and "https://" not in text
    assert "<style>" in text and "<script>" in text
    assert "src=\"//" not in text


def test_the_page_is_exactly_what_render_page_returns(server):
    _, base = server
    assert _get(base + "/")[1] == render_page()


def test_state_endpoint_serves_the_snapshot(server):
    app, base = server
    status, blob, headers = _get(base + "/state")
    assert status == 200
    assert headers["Cache-Control"] == "no-store, max-age=0"
    assert json.loads(blob)["mode"] == app.state.mode


def test_frame_endpoint_serves_a_jpeg_with_a_token(server):
    _, base = server
    status, blob, headers = _get(base + "/frame.jpg")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert blob[:2] == b"\xff\xd8"  # JPEG SOI
    assert headers["X-Frame-Token"]


def test_health_endpoint(server):
    _, base = server
    status, blob, _ = _get(base + "/health")
    assert status == 200
    assert json.loads(blob)["status"] == "ok"


def test_posting_events_advances_the_run(server):
    app, base = server
    status, body = _post(base + "/events", [
        {"kind": "order", "order_id": "SO-9", "item": "eraser", "bin": "B"},
        {"kind": "stage", "stage": "PICK", "status": "active"},
        {"kind": "motion", "stage": "PICK", "state": "commanded"},
    ])
    assert (status, body["applied"]) == (200, 3)
    doc = json.loads(_get(base + "/state")[1])
    assert doc["order"]["item"] == "eraser"
    assert doc["motion"][0]["state"] == "commanded"


def test_posting_the_mode_switches_it(server):
    _, base = server
    assert _post(base + "/mode", {"mode": "fusion"})[1]["mode"] == "fusion"
    assert json.loads(_get(base + "/state")[1])["mode"] == "fusion"


def test_an_unknown_mode_is_422_not_500(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/mode", {"mode": "banana"})
    assert e.value.code == 422


def test_invalid_json_is_422(server):
    _, base = server
    req = urllib.request.Request(base + "/events", data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 422


def test_unknown_route_is_404_and_lists_the_real_ones(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/nope")
    assert e.value.code == 404
    assert "/state" in json.loads(e.value.read())["routes"]


def test_wrong_method_is_405_with_an_allow_header(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/state", {})
    assert e.value.code == 405
    assert "GET" in e.value.headers["Allow"]


def test_auto_view_frames_the_zones_and_stays_inside_the_frame():
    rects = F.load_zone_rects()
    view = F.auto_view(rects, (1920, 1080))
    assert (view.w, view.h) == (F.VIEW_W, F.VIEW_H)
    assert view.x >= 0 and view.y >= 0
    assert view.x + view.w <= 1920 and view.y + view.h <= 1080
    for x0, y0, x1, y1 in rects.values():  # every zone is actually visible
        assert view.x <= x0 and x1 <= view.x + view.w
        assert view.y <= y0 and y1 <= view.y + view.h


def test_auto_view_recentres_on_extra_points():
    """The mock frames the props too, so the pick area is not pushed off-shot."""
    rects = F.load_zone_rects()
    plain = F.auto_view(rects, (1920, 1080))
    with_props = F.auto_view(rects, (1920, 1080), extra=list(F.PROP_HOME.values()))
    assert with_props.x < plain.x  # props lie left of the zones → the window shifts left
    assert (with_props.w, with_props.h) == (plain.w, plain.h)


def test_view_round_trips_through_the_cli_spec():
    assert F.parse_view("500,350,700,394").to_dict() == {"x": 500, "y": 350, "w": 700, "h": 394}
    with pytest.raises(ValueError):
        F.parse_view("1,2,3")


def test_pixel_to_view_fraction_maps_the_crop_corners_to_0_and_1():
    view = F.View(100, 50, 800, 400)
    assert F.pixel_to_view_fraction(100, 50, view) == (0.0, 0.0)
    assert F.pixel_to_view_fraction(900, 450, view) == (1.0, 1.0)


def test_parse_events_accepts_the_three_producer_shapes():
    one = {"kind": "log", "text": "x"}
    assert parse_events(one) == [one]
    assert parse_events([one, one]) == [one, one]
    assert parse_events({"events": [one]}) == [one]
    assert parse_events({"no": "kind"}) == []
    assert parse_events("nonsense") == []


# ---------------------------------------------------------------- the launcher
@pytest.fixture(scope="module")
def launcher():
    """``tools/dashboard.py`` is a script, not a package — load it by path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "dashboard.py"
    spec = importlib.util.spec_from_file_location("dashboard_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_bare_command_is_a_mock_run(launcher):
    """`python tools/dashboard.py` with no flags must need no network and no hardware."""
    args = launcher.resolve_source(launcher.build_parser().parse_args([]))
    assert args.mock is True
    assert args.follow is None
    assert args.runtime is None  # → MOCK, so nothing is ever sent
    assert args.mode == "order"
    assert args.outcome == "auto"


def test_follow_opts_out_of_the_canned_script(launcher):
    parse = lambda argv: launcher.resolve_source(launcher.build_parser().parse_args(argv))
    assert parse(["--follow", "run.jsonl"]).mock is False
    assert parse(["--follow", "run.jsonl", "--mock"]).mock is False  # the real file wins
    assert parse(["--no-mock"]).mock is False


def test_the_launcher_offers_every_mode_and_ending(launcher):
    parser = launcher.build_parser()
    assert parser.parse_args(["--mode", "fusion"]).mode == "fusion"
    assert parser.parse_args(["--outcome", "fail"]).outcome == "fail"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "banana"])


def test_a_mock_run_can_never_be_given_a_sending_runtime(launcher):
    """--mock forces MOCK: a canned alert must never be able to look real."""
    args = launcher.resolve_source(
        launcher.build_parser().parse_args(["--mock", "--runtime", "live"])
    )
    app = launcher.build_app(args)
    assert app.platform.runtime == "MOCK"
    assert app.platform.enabled is False
    assert app.snapshot()["platform"]["status"] == "mock"


def test_a_follow_run_defaults_to_simulation_source_type(launcher):
    args = launcher.resolve_source(launcher.build_parser().parse_args(["--follow", "x.jsonl"]))
    assert args.mock is False
    link = PlatformLink(runtime="SIMULATION", enabled=True)
    assert link.source_type == "simulation"  # "edge" only once it runs on real hardware
