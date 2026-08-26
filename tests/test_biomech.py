import json

import numpy as np
import pytest

from biomech import (SEGMENTS, angle_at, body_scale, clean_keypoints,
                     compute_com, detect_contacts, dist_point_seg, point_in_tri,
                     smooth, analyze)
from conftest import KP_ORDER, person, person_xy
from validate import validate_analysis


FPS = 30.0


def frames_from_kpts(arr, conf=0.9):
    """(N,17,2) -> pose-json frames list."""
    arr = np.asarray(arr, dtype=float)
    out = []
    for i in range(len(arr)):
        out.append({"i": i, "t": i / FPS, "track_id": 1,
                    "bbox": [0, 0, 1080, 1920],
                    "kpts": [[float(x), float(y), conf] for x, y in arr[i]]})
    return out


# ---------- pure helpers ----------

PAIRED = {"uarm", "farm", "hand", "thigh", "shank", "foot"}


def test_segment_masses_sum_to_one():
    total = sum(v["m"] for k, v in SEGMENTS.items() if k not in PAIRED) +         2 * sum(v["m"] for k, v in SEGMENTS.items() if k in PAIRED)
    assert abs(total - 1.0) < 1e-9


def test_clean_keypoints_interpolates_gaps():
    frames = frames_from_kpts(
        np.array([person_xy(540, 800), person_xy(540, 800), person_xy(540, 900)]))
    frames[1]["kpts"][0] = [540.0, 800.0, 0.05]  # nose dropout
    P, C = clean_keypoints(frames)
    assert not np.isnan(P[1, 0, 1])
    assert P[1, 0, 1] == pytest.approx(700.0)  # 650/750 interpolation


def test_smooth_handles_all_nan_column():
    frames = frames_from_kpts(np.array([person_xy(540, 800) for _ in range(12)]))
    for f in frames:  # left ear never confident
        f["kpts"][3] = [0.0, 0.0, 0.0]
    P, _ = clean_keypoints(frames)
    S = smooth(P)  # must not raise
    assert np.all(S[:, 3] == 0.0)


def test_angle_at_cardinals():
    b = np.array([0.0, 0.0])
    right = np.array([1.0, 0.0])
    up = np.array([0.0, 1.0])
    left = np.array([-1.0, 0.0])
    assert angle_at(right, b, up) == pytest.approx(90.0)
    assert angle_at(right, b, left) == pytest.approx(180.0)
    assert angle_at(right, b, right) == pytest.approx(0.0)


def test_body_scale_known_torso():
    P = np.array([person_xy(540, 800, torso=120.0)])
    torso, _, _ = body_scale(P)
    assert torso == pytest.approx(120.0)


def test_com_symmetric_pose_lies_on_axis():
    P = np.array([person_xy(540, 800)])
    com = compute_com(P)
    assert com[0, 0] == pytest.approx(540.0, abs=1.0)


def test_com_within_body_span():
    P = np.array([person_xy(540, 800)])
    com = compute_com(P)
    ys = P[0, :, 1]
    assert ys.min() - 1 <= com[0, 1] <= ys.max() + 1


def test_com_shifts_with_arms_up():
    base = np.array([person_xy(540, 800)])
    raised = base.copy()
    raised[0][KP_ORDER.index("lwr")] = [540 - 66, 800 - 145]  # arms overhead
    raised[0][KP_ORDER.index("rwr")] = [540 + 66, 800 - 145]
    com_base = compute_com(base)[0, 1]
    com_raised = compute_com(raised)[0, 1]
    assert com_raised < com_base  # CoM moves up in image coords


# ---------- contacts ----------

def _wrist_series(positions, torso=120.0):
    """(N,17,2) with only the right wrist animated."""
    arr = np.array([person_xy(540, 800, torso) for _ in positions])
    for i, (wx, wy) in enumerate(positions):
        arr[i][KP_ORDER.index("rwr")] = [wx, wy]
    return arr


def test_contact_stationary_wrist():
    pos = [(606, 788) for _ in range(60)]
    contacts, _ = detect_contacts(_wrist_series(pos), 120.0, FPS)
    assert contacts["rh"].sum() >= 55


def test_contact_moving_wrist_never_latches():
    pos = [(606 + 8 * i, 788) for i in range(60)]  # 240 px/s = 2 torso/s
    contacts, _ = detect_contacts(_wrist_series(pos), 120.0, FPS)
    assert contacts["rh"].sum() == 0


def test_contact_hysteresis_survives_single_frame_spike():
    pos = [(606, 788) for _ in range(60)]
    pos[30] = (606 + 40, 788)  # one-frame jitter
    contacts, _ = detect_contacts(_wrist_series(pos), 120.0, FPS)
    assert contacts["rh"][35]  # still latched after the spike


# ---------- geometry ----------

def test_point_in_tri_inside_and_outside():
    tri = (np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 10.0]))
    inside, m_in = point_in_tri(np.array([2.0, 2.0]), *tri)
    outside, m_out = point_in_tri(np.array([8.0, 8.0]), *tri)
    assert inside and not outside
    assert m_in == pytest.approx(2.0, abs=0.01)
    assert m_out == pytest.approx(4.24, abs=0.05)  # dist to hypotenuse x+y=10


def test_dist_point_seg():
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    assert dist_point_seg(np.array([5.0, 3.0]), a, b) == pytest.approx(3.0)
    assert dist_point_seg(np.array([15.0, 4.0]), a, b) == pytest.approx(
        np.hypot(5, 4))  # clamped to endpoint


# ---------- end-to-end on synthetic clip ----------

def test_analyze_synthetic_passes_validator(synth_pose, tmp_path):
    out = tmp_path / "analysis.json"
    analyze(str(synth_pose), str(out))
    errs = validate_analysis(str(out), min_com=0.85, w=1080, h=1920)
    assert errs == []


def test_analyze_ground_person_is_idle(tmp_path):
    frames = frames_from_kpts(np.array([person_xy(540, 1500) for _ in range(60)]))
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps({"meta": {"fps": 30}, "frames": frames}),
                    encoding="utf-8")
    out = tmp_path / "analysis.json"
    analyze(str(pose), str(out))
    d = json.loads(out.read_text(encoding="utf-8"))
    assert all(f["state"] == "idle" for f in d["frames"])
    assert d["stats"]["climbing_s"] == 0.0


def test_analyze_detects_right_hand_transfer(synth_pose, tmp_path):
    out = tmp_path / "analysis.json"
    analyze(str(synth_pose), str(out))
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["stats"]["state_s"]["3pt"] > 0
    moved_events = [e for e in d["events"]
                    if "右手" in e["moved"] and e["t0"] > 1.5]  # skip teleport
    assert moved_events, f"no transfer attributed to 右手: {d['events']}"
    e = moved_events[0]
    assert e["disp"] > 0.20
    assert e["t1"] > e["t0"]
    assert 60 / 30 - 1 <= e["t0"] <= 80 / 30 + 1  # around the move (2.0-3.7s)
    # transfers carry the bracketing states (3pt -> 3pt is the common chain)
    valid = {"4pt", "3pt", "2pt", "1pt", "idle"}
    assert e["s0"] in valid and e["s1"] in valid
    assert all(x["s0"] in valid and x["s1"] in valid for x in d["events"])
