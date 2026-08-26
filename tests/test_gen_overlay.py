import json
import re
from pathlib import Path

import pytest

from gen_overlay import (build_outputs, cap_zone_windows, fmt_events,
                         pick_segment)

FPS = 30.0


def frame(i, climbing, state="3pt"):
    return {"i": i, "t": i / FPS, "climbing": climbing, "state": state,
            "com": [540.0, 800.0], "kpts": None, "angles": {}, "contacts": {},
            "n": 3, "margin": 0.2, "inside": True, "bd": 0.1, "st_t0": 0.0}


def event(t0, t1, disp=0.5, moved=("右手",), dx=0.3, dy=-0.2, s0=None, s1=None):
    e = {"t0": round(t0, 2), "t1": round(t1, 2), "dur": round(t1 - t0, 2),
         "moved": list(moved), "dx": dx, "dy": dy, "disp": disp}
    if s0 is not None:
        e["s0"], e["s1"] = s0, s1
    return e


# ---------- pick_segment ----------

def test_pick_segment_keeps_head_from_frame_zero():
    """Preparation footage is never trimmed: the film starts at frame 0."""
    fr = ([frame(i, False) for i in range(10)] +
          [frame(i, True) for i in range(10, 70)] +
          [frame(i, False) for i in range(70, 150)])
    a, b = pick_segment(fr, FPS)
    assert a == 0
    assert 69 <= b <= 70 + 45                # tail follows the climbing end


def test_pick_segment_picks_longest_run_for_tail():
    fr = ([frame(i, False) for i in range(10)] +
          [frame(i, True) for i in range(10, 70)] +       # 60f run (longest)
          [frame(i, False) for i in range(70, 150)] +
          [frame(i, True) for i in range(150, 190)] +     # 40f run
          [frame(i, False) for i in range(190, 220)])
    a, b = pick_segment(fr, FPS)
    assert a == 0
    assert 69 <= b <= 70 + 45


def test_pick_segment_tolerates_short_dropouts():
    fr = [frame(i, True) for i in range(120)]
    for i in (62, 63, 70):  # <2.5s gaps keep the run alive
        fr[i]["climbing"] = False
    a, b = pick_segment(fr, FPS)
    assert b - a >= 100


def test_pick_segment_caps_duration_from_the_tail():
    fr = [frame(i, False) for _ in range(0)] + \
         [frame(i, i >= 300) for i in range(600)]
    a, b = pick_segment(fr, FPS, max_s=5.0)
    assert a == 0                            # head is still kept
    assert (b - a + 1) / FPS <= 5.0


def test_pick_segment_no_climbing_fallback():
    fr = [frame(i, False) for i in range(300)]
    a, b = pick_segment(fr, FPS, max_s=10.0)
    assert a == 0
    assert (b - a + 1) / FPS == pytest.approx(10.0)


# ---------- fmt_events ----------

def test_fmt_events_filters_early_and_late():
    seg = [frame(i, True) for i in range(30 * 30)]  # 30s segment
    evs = [event(1.0, 2.0),          # ends before title card -> dropped
           event(3.0, 4.5),          # t1 < 4.6 -> dropped
           event(10.0, 12.0),        # kept
           event(29.9, 30.5)]        # starts beyond seg-1s -> dropped
    out = fmt_events(seg, evs, FPS, 0)
    assert len(out) == 1
    assert out[0][0]["t0"] == 10.0


def test_fmt_events_one_caption_at_a_time():
    seg = [frame(i, True) for i in range(30 * 30)]
    evs = [event(10.0, 11.0, disp=0.9),   # higher displacement wins
           event(10.5, 11.5, disp=0.4),   # <1.05s apart, smaller -> yields
           event(14.0, 15.0, disp=0.3)]   # no overlap -> kept
    out = fmt_events(seg, evs, FPS, 0)
    assert [e["t0"] for e, _, _ in out] == [10.0, 14.0]


def test_fmt_events_chains_captions_without_dropping():
    """3pt->3pt transfers chain back-to-back: queue them, don't skip."""
    seg = [frame(i, True) for i in range(30 * 30)]
    evs = [event(10.0, 13.0, disp=0.22),   # long dur, clamped to fit
           event(12.5, 13.6, disp=0.28),
           event(15.0, 16.4, disp=0.24)]
    out = fmt_events(seg, evs, FPS, 0)
    assert [e["t0"] for e, _, _ in out] == [10.0, 12.5, 15.0]
    # earlier caption shortened so it hands off before the next one
    assert out[0][0]["cap_dur"] == pytest.approx(0.85, abs=0.01)
    assert out[1][0]["cap_dur"] == pytest.approx(0.85, abs=0.01)
    assert out[2][0]["cap_dur"] == pytest.approx(1.4, abs=0.01)  # unclamped


def test_fmt_events_unfittable_pair_yields_to_larger():
    seg = [frame(i, True) for i in range(30 * 30)]
    evs = [event(10.0, 14.0, disp=0.5),
           event(11.4, 15.0, disp=0.9)]   # 1.4s apart: both can't fit
    out = fmt_events(seg, evs, FPS, 0)
    assert [e["t0"] for e, _, _ in out] == [11.4]  # bigger displacement wins


def test_fmt_events_offsets_by_segment_start():
    seg = [frame(i, True) for i in range(30 * 30)]
    evs = [event(40.0, 42.0)]
    out = fmt_events(seg, evs, FPS, 30 * 20)  # segment starts at t=20s
    assert out[0][1] == pytest.approx(20.0)


# ---------- build_outputs ----------

_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = next(p for p in [_ROOT / "overlay_template.html",
                                 _ROOT / "scripts" / "overlay_template.html"]
                     if p.exists())


def _analysis_dict():
    n = 30 * 40
    frames = [frame(i, i > 60) for i in range(n)]
    frames[0:30] = [frame(i, False, state="idle") for i in range(30)]
    return {
        "meta": {"fps": FPS},
        "stats": {"fps": FPS, "n_frames": n, "torso_px": 120.0,
                  "duration_s": n / FPS, "climbing_s": 30.0,
                  "state_s": {"4pt": 10.0, "3pt": 12.0, "2pt": 5.0,
                              "1pt": 2.0, "idle": 11.0},
                  "transfers": 3, "max_barndoor": 1.4,
                  "com_detected_pct": 100.0},
        "frames": frames,
        "events": [event(10.0, 12.0), event(20.0, 21.5)],
    }


def test_build_outputs_replaces_all_tokens():
    html, data, tim = build_outputs(_analysis_dict(), 0, 30 * 40 - 1,
                                    "segment.mp4", "测试副标题",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert "{{" not in html and "}}" not in html
    assert "测试副标题" in html
    assert 'src="segment.mp4"' in html


def test_build_outputs_balanced_divs_and_event_clips():
    html, data, tim = build_outputs(_analysis_dict(), 0, 30 * 40 - 1,
                                    "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert html.count("<div") == html.count("</div>")
    assert html.count('class="clip evcap"') == 2
    assert 'id="evi0"' in html and 'id="evi1"' in html


def test_build_outputs_hud_timing_gates():
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    # first climbing frame at i=61 -> t=2.03s -> clamped to title end 4.3
    assert tim["hud_bot_start"] == pytest.approx(4.3)
    assert tim["hud_top_start"] == pytest.approx(4.3)
    assert tim["ov_end"] < tim["dur"]
    assert tim["hud_bot_start"] + tim["hud_bot_dur"] <= tim["dur"] + 0.01
    assert re.search(r'id="hud-bottom"[^>]*data-start="4\.30"', html)
    assert 'data-duration="' in html


def test_build_outputs_pose_layer_gated_on_climb_start():
    """Analysis layers only appear once climbing starts; clean footage before."""
    d = _analysis_dict()
    frames = d["frames"]
    for i in range(0, 180):  # climbing starts at t=6.0s, not before
        frames[i] = frame(i, False, state="idle")
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert tim["pose_t0"] == pytest.approx(6.0)
    assert tim["pose_pre"] == pytest.approx(5.2)
    # overlay clip mounts just before, HUDs gated at max(title, pose_pre)
    assert re.search(r'id="overlay-svg"[^>]*data-start="5\.20"', html)
    assert tim["hud_top_start"] == pytest.approx(5.2)
    assert tim["hud_bot_start"] == pytest.approx(5.2)
    assert data["pose_t0"] == pytest.approx(6.0)


def test_build_outputs_first_frame_is_raw_footage_plus_local_title():
    """No full-screen mask, no entrance animation: frame 1 shows the gym."""
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert re.search(r'id="title-panel"[^>]*data-start="0"', html)
    assert "#tp-card" in html
    assert 'background: rgba(8, 13, 20, 0.62)' in html   # local, translucent
    # no entrance tweens on the title; only the gentle fade-out remains
    assert 'tl.from("#tc-' not in html
    assert 'tl.from("#tp-card"' not in html
    assert 'tl.to("#tp-card", { opacity: 0' in html


def test_build_outputs_no_joint_angle_labels():
    """Angle arcs/numbers clutter the 3pt story - they must be gone."""
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert "ANGLE_GEO" not in html
    assert "atxt" not in html
    assert "rgba(255,211,92,0.5)" not in html       # angle arc stroke


def test_build_outputs_tech_badges_follow_and_rebase():
    d = _analysis_dict()
    tech = {"phases": [{"name": "side_on", "t0": 10.0, "t1": 15.0, "side": "左"}],
            "events": [{"name": "foot_swap", "t": 12.0}]}
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    tech=tech)
    assert 'id="tech-follow"' in html
    assert data["tech"]["phases"][0]["t0"] == pytest.approx(10.0)
    assert data["tech"]["events"][0]["t"] == pytest.approx(12.0)
    assert tim["n_tech"] == 2
    # re-based to segment start: a=20s offset drops out-of-window phases
    html2, data2, _ = build_outputs(d, 30 * 20, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    tech=tech)
    assert data2["tech"]["phases"] == []           # 10-15s is before the segment
    # phases overlapping the segment window are kept and shifted
    tech3 = {"phases": [{"name": "side_on", "t0": 25.0, "t1": 28.0}],
             "events": []}
    _, data3, _ = build_outputs(d, 30 * 20, 30 * 40 - 1, "segment.mp4", "t",
                                TEMPLATE_PATH.read_text(encoding="utf-8"),
                                tech=tech3)
    assert data3["tech"]["phases"][0]["t0"] == pytest.approx(5.0)


def test_build_outputs_summary_includes_tech_count():
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    tech={"phases": [{"name": "flagging",
                                                      "t0": 8, "t1": 9}],
                                          "events": [{"name": "high_step",
                                                      "t": 8.5}]})
    assert 'id="sv-tech"' in html
    assert '"#sv-tech"' in html          # count-up tween present


def test_build_outputs_hud_hide_when_person_in_panel_zone():
    """Bottom panel + captions hide while the climber sits in the panel zone."""
    d = _analysis_dict()
    n = 30 * 20
    frames = []
    for i in range(n):
        low = i < 5 * 30  # person low (in panel zone) for the first 5s
        kpts = [None] * 17
        kpts[15] = [540.0, 1800.0 if low else 600.0]
        fr = frame(i, True)
        fr["kpts"] = kpts
        frames.append(fr)
    d["frames"] = frames
    _, data, _ = build_outputs(d, 0, n - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"))
    wins = data["hud_hide"]
    assert wins, "expected a hide window while the person is low"
    a, b = wins[0]
    assert a <= 0.5                      # hidden from the start
    assert 4.0 <= b <= 5.5               # released shortly after they clear


def _captions():
    return [
        {"start": 0.2, "end": 3.0, "text": "起步然后往左",
         "terms": [{"off": 0, "term": "起步"}]},
        {"start": 3.0, "end": 6.5, "text": "侧身起勾住小勾勾",
         "terms": [{"off": 0, "term": "侧身"}]},
        {"start": 30.0, "end": 33.0, "text": "交叉脚", "terms": []},
    ]


def test_build_outputs_speech_captions_sequential():
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    captions=_captions())
    # three caption clips on the subtitle track, term-highlighted
    assert html.count('class="clip cap"') == 3
    assert html.count('data-track-index="10"') == 3
    assert '<span class="term">起步</span>' in html
    # strictly sequential, non-overlapping windows
    starts = [float(x) for x in
              re.findall(r'id="cap\d+" class="clip cap" data-start="([\d.]+)"', html)]
    durs = [float(x) for x in re.findall(
        r'id="cap\d+" class="clip cap" data-start="[\d.]+" data-duration="([\d.]+)"', html)]
    assert len(starts) == 3 and len(durs) == 3
    for (s1, w1), s2 in zip(zip(starts, durs), starts[1:]):
        assert s1 + w1 <= s2 + 0.01


def test_build_outputs_captions_trimmed_before_summary_card():
    d = _analysis_dict()
    caps = _captions() + [{"start": 38.5, "end": 41.0, "text": "收尾", "terms": []}]
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    captions=caps)
    starts = [float(x) for x in
              re.findall(r'id="cap\d+" class="clip cap" data-start="([\d.]+)"', html)]
    durs = [float(x) for x in re.findall(
        r'id="cap\d+" class="clip cap" data-start="[\d.]+" data-duration="([\d.]+)"', html)]
    assert starts and max(s + w for s, w in zip(starts, durs)) <= tim["sum_t0"] + 0.5


def test_build_outputs_no_captions_leaves_no_tokens():
    d = _analysis_dict()
    html, _, _ = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert "{{CAPTION" not in html
    assert 'class="clip cap"' not in html


# ---------- cap_zone_windows (subtitle slot vs high climber) ----------

def _kpt_frame(i, nose_y, climbing=True):
    fr = frame(i, climbing)
    kpts = [None] * 17
    kpts[0] = [540.0, nose_y]           # nose
    fr["kpts"] = kpts
    return fr


def test_cap_zone_windows_head_high_moves_subtitles():
    # head below the top zone (base 470) for 5s, then rises into it
    n = 30 * 12
    frs = [_kpt_frame(i, 900.0 if i < 5 * 30 else 300.0) for i in range(n)]
    wins = cap_zone_windows(frs, FPS, n / FPS, scale=1.0)
    assert wins, "expected a low-slot window once the head is high"
    a, b = wins[0]
    assert 4.0 <= a <= 5.5              # fires shortly after the head rises
    assert b > 10.0                     # stays low while the head stays up


def test_cap_zone_windows_head_low_no_window():
    n = 30 * 10
    frs = [_kpt_frame(i, 900.0) for i in range(n)]
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.0) == []


def test_cap_zone_windows_missing_head_carries_state():
    # head high, then keypoints drop out, then high again: one merged window
    n = 30 * 12
    frs = [_kpt_frame(i, 300.0) for i in range(n)]
    for i in range(6 * 30, 7 * 30):     # detection dropout mid-window
        frs[i]["kpts"] = None
    wins = cap_zone_windows(frs, FPS, n / FPS, scale=1.0)
    assert len(wins) == 1 and wins[0][1] > 11.0


def test_cap_zone_windows_toggles_back_when_low_band_occluded():
    """Slot switching is a two-way toggle on TRUE occlusion: top band
    covered -> low slot; later the low band itself gets covered (climber
    down low) -> back to top, closing the low window. Between the moves
    the slot stays put."""
    n = 30 * 26
    frs = []
    for i in range(n):
        if i < 5 * 30:
            y = 900.0        # between the bands: stays top
        elif i < 12 * 30:
            y = 300.0        # occludes the top band (base 200..400)
        elif i < 18 * 30:
            y = 900.0        # between the bands: slot must STAY low
        else:
            y = 1400.0       # occludes the low band (base 1350..1550)
        frs.append(_kpt_frame(i, y))
    wins = cap_zone_windows(frs, FPS, n / FPS, scale=1.0)
    assert len(wins) == 1                    # one low episode, closed on return
    a, b = wins[0]
    assert 4.0 <= a <= 5.5                   # moves low after the top covers
    assert 17.5 <= b <= 19.0                 # returns once the low band covers


def test_cap_zone_windows_hovering_edge_never_oscillates():
    """The old head-threshold proxy bounced whenever the climber hovered
    near the line; now nothing moves unless the band truly covers them
    for a sustained stretch (0.8s), so flickering input = no window."""
    n = 30 * 12
    frs = []
    for i in range(n):
        # in/out of the band every 10 frames - never sustained
        y = 300.0 if (i // 10) % 2 == 0 else 470.0
        frs.append(_kpt_frame(i, y))
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.0) == []


def test_cap_zone_windows_band_off_center_person_not_covered():
    """In a wide (landscape) frame the centered subtitle band covers only
    the middle: a climber high up at the far left is NOT occluded and the
    slot stays put; the same height dead-center does move it."""
    n = 30 * 12
    frs = [_kpt_frame(i, 350.0) for i in range(n)]   # band y = 225..405
    for fr in frs:
        fr["kpts"][0] = [500.0, 350.0]               # far left of the band
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.125, vw=3840) == []
    for fr in frs:
        fr["kpts"][0] = [1920.0, 350.0]              # dead center: covered
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.125, vw=3840)


def _bone_frame(nose_y, lsho, lhip, rsho=None, rhip=None):
    fr = _kpt_frame(0, nose_y)
    k = fr["kpts"]
    k[5] = lsho
    k[11] = lhip
    if rsho:
        k[6] = rsho
    if rhip:
        k[12] = rhip
    return fr


def test_cap_zone_windows_skeleton_segment_collision_counts():
    """A bone crossing the band triggers the move even when no keypoint is
    inside the band (true skeleton collision, not a bbox approximation)."""
    n = 30 * 10
    frs = []
    for i in range(n):
        fr = _bone_frame(60.0, (540.0, 100.0), (540.0, 700.0))
        frs.append(fr)
    # shoulder-hip bone at x=540 crosses the top band (y 200..360)
    wins = cap_zone_windows(frs, FPS, n / FPS, scale=1.0)
    assert wins and wins[0][0] <= 1.5


def test_cap_zone_windows_bbox_overlap_without_bone_does_not_count():
    """Limbs hug the left/right edges: the keypoint bbox overlaps the band
    but no keypoint or bone is inside it - the subtitle must stay put."""
    n = 30 * 10
    frs = []
    for i in range(n):
        # vertical bones at x=20 and x=1041 (outside band x 40..1040),
        # horizontal bones at y=100 and y=700 (outside band y 200..360)
        fr = _bone_frame(60.0, (20.0, 100.0), (20.0, 700.0),
                         rsho=(1041.0, 100.0), rhip=(1041.0, 700.0))
        frs.append(fr)
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.0) == []


def test_cap_zone_windows_never_moves_into_occupied_slot():
    """If the climber's skeleton occupies BOTH slots, the subtitle stays
    top instead of bouncing into an equally covered low slot."""
    n = 30 * 12
    frs = []
    for i in range(n):
        fr = _bone_frame(300.0, (540.0, 300.0), (540.0, 1400.0))
        # nose at 300 (top band), hip at 1400 (low band 1350..1510)
        frs.append(fr)
    assert cap_zone_windows(frs, FPS, n / FPS, scale=1.0) == []


def test_cap_zone_windows_dwell_blocks_rapid_reswitching():
    """Occlusion bursts closer together than dwell_s produce at most one
    switch - the position cannot oscillate."""
    n = 30 * 16
    frs = []
    for i in range(n):
        t = i / FPS
        in_top = (1.0 <= t <= 3.0) or (5.0 <= t <= 7.0) or (9.0 <= t <= 11.0)
        frs.append(_kpt_frame(i, 300.0 if in_top else 900.0))
    wins = cap_zone_windows(frs, FPS, n / FPS, scale=1.0)
    # first burst switches to low; the 2s-apart re-occlusions (top band
    # occupied again while dwelling in low) must not flip it back and forth
    assert len(wins) <= 2


def test_build_outputs_emits_cap_low():
    d = _analysis_dict()
    n = 30 * 15
    frs = [_kpt_frame(i, 900.0 if i < 5 * 30 else 300.0, i >= 90)
           for i in range(n)]
    d["frames"] = frs
    _, data, _ = build_outputs(d, 0, n - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert data["cap_low"], "expected cap_low windows in data.js"


def test_build_outputs_native_resolution_tokens():
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    vw=2160, vh=3840)
    assert "{{" not in html and "}}" not in html
    assert 'data-width="2160"' in html and 'data-height="3840"' in html
    assert 'width="2160" height="3840" viewBox="0 0 2160 3840"' in html
    assert html.count("transform: scale(2)") >= 4   # hud + cards + captions
    assert "bottom: 696px" in html                   # evcap = 348 * 2


def test_build_outputs_landscape_scales_by_height():
    """Landscape 4K (3840x2160): scale = min(W/1080, H/1920) so the base
    grid fits BOTH axes - W/1080 would make the panel eat half the frame."""
    d = _analysis_dict()
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"),
                                    vw=3840, vh=2160)
    assert html.count("transform: scale(1.125)") >= 4
    assert "bottom: 392px" in html                   # evcap = 348 * 1.125


def test_build_outputs_landscape_panel_hide_checks_x_extent():
    """In landscape the panel spans only part of the width: a climber low
    but far right does not hide it."""
    d = _analysis_dict()
    n = 30 * 10
    frames = []
    for i in range(n):
        kpts = [None] * 17
        kpts[16] = [3300.0, 2000.0]   # low, far right of the panel span
        fr = frame(i, True)
        fr["kpts"] = kpts
        frames.append(fr)
    d["frames"] = frames
    _, data, _ = build_outputs(d, 0, n - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"),
                               vw=3840, vh=2160)
    assert data["hud_hide"] == []


def test_build_outputs_bottom_panel_support_and_tech_not_gauges():
    """The stability/barn-door gauges and COM sparklines are gone; the
    panel shows the 3pt support limbs and the current technique."""
    d = _analysis_dict()
    html, _, _ = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"))
    for nm in ("lh", "rh", "lf", "rf"):
        assert f'id="limb-{nm}"' in html
    assert 'id="tech-now"' in html and 'id="tech-cnt"' in html
    assert "STABILITY MARGIN" not in html       # gauges retired
    assert "BARN-DOOR ARM" not in html
    assert "spark-x" not in html and "spark-y" not in html


def test_build_outputs_3pt_transfer_label():
    d = _analysis_dict()
    d["events"] = [event(10.0, 12.0, s0="3pt", s1="3pt"),
                   event(20.0, 21.5, s0="3pt", s1="4pt")]
    html, data, tim = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert "三点平衡转移" in html        # 3pt -> 3pt chain gets its own label
    assert "重心转移" in html            # other transitions stay generic


def test_build_outputs_boundary_hard_kill_when_fade_crosses_clip():
    """Event exit fades must not cross a later clip's start without a tl.set
    on that boundary (hyperframes gsap_exit_missing_hard_kill)."""
    d = _analysis_dict()
    d["events"] = [event(10.0, 13.2)]        # cap_dur 3.2 -> fade ends 14.25
    caps = [{"start": 12.0, "end": 13.0, "text": "字幕切入", "terms": []}]
    html, _, _ = build_outputs(d, 0, 30 * 40 - 1, "segment.mp4", "t",
                               TEMPLATE_PATH.read_text(encoding="utf-8"),
                               captions=caps)
    assert 'tl.set("#evi0", {opacity:0}, 14.35)' in html   # own kill intact
    assert 'tl.set("#evi0", {opacity:0}, 11.85)' in html   # boundary 11.85 (cap clip start)


def test_build_outputs_data_caps_do_not_overlap():
    html, data, tim = build_outputs(_analysis_dict(), 0, 30 * 40 - 1,
                                    "segment.mp4", "t",
                                    TEMPLATE_PATH.read_text(encoding="utf-8"))
    caps = data["caps"]
    assert caps == sorted(caps, key=lambda c: c[0])
    for (a0, a1), (b0, b1) in zip(caps, caps[1:]):
        assert b0 >= a1 - 0.01
    assert data["ov_end"] > 0
