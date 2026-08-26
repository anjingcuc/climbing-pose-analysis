import json

import pytest

from validate import validate_analysis, validate_pose
from conftest import person


def _good_pose(tmp_path):
    frames = []
    for i in range(30):
        frames.append({"i": i, "t": i / 30, "track_id": 1,
                       "bbox": [0, 0, 1080, 1920],
                       "kpts": person(540, 800)})
    p = tmp_path / "pose.json"
    p.write_text(json.dumps({"meta": {"fps": 30}, "frames": frames}),
                 encoding="utf-8")
    return p, frames


def test_validate_pose_ok(tmp_path):
    p, _ = _good_pose(tmp_path)
    assert validate_pose(str(p), min_detect=0.9) == []


def test_validate_pose_low_detection(tmp_path):
    p, frames = _good_pose(tmp_path)
    for f in frames[10:]:  # 2/3 missing
        f["kpts"] = None
    p.write_text(json.dumps({"meta": {"fps": 30}, "frames": frames}),
                 encoding="utf-8")
    errs = validate_pose(str(p), min_detect=0.75)
    assert any("detection rate" in e for e in errs)


def test_validate_pose_bad_conf_and_range(tmp_path):
    p, frames = _good_pose(tmp_path)
    frames[0]["kpts"][0][2] = 1.5          # conf out of range
    frames[1]["kpts"][5][0] = 99999        # x outside frame
    p.write_text(json.dumps({"meta": {"fps": 30}, "frames": frames}),
                 encoding="utf-8")
    errs = validate_pose(str(p), w=1080, h=1920)
    assert any("conf" in e for e in errs)
    assert any("outside frame" in e for e in errs)


def test_validate_pose_index_gap(tmp_path):
    p, frames = _good_pose(tmp_path)
    frames[5]["i"] = 9
    p.write_text(json.dumps({"meta": {"fps": 30}, "frames": frames}),
                 encoding="utf-8")
    assert any("index gap" in e for e in validate_pose(str(p)))


# ---------- analysis ----------

def good_frame(i, state="3pt", n=3, climbing=True):
    return {"i": i, "t": i / 30, "kpts": person(540, 800), "com": [540.0, 800.0],
            "angles": {"lel": 90.0}, "contacts": {"lh": True, "rh": False,
            "lf": True, "rf": n == 3}, "n": n, "state": state, "st_t0": 0.0,
            "bd": 0.2, "margin": 0.3, "inside": True, "climbing": climbing}


def _good_analysis(tmp_path, frames=None):
    n = 30
    frames = frames or [good_frame(i) for i in range(n)]
    d = {"meta": {"fps": 30}, "frames": frames,
         "events": [{"t0": 0.5, "t1": 1.2, "dur": 0.7, "moved": ["右手"],
                     "dx": 0.2, "dy": -0.1, "disp": 0.3}],
         "stats": {"fps": 30, "n_frames": n, "torso_px": 120.0,
                   "duration_s": 1.0, "climbing_s": 1.0,
                   "state_s": {"3pt": 1.0}, "transfers": 1,
                   "max_barndoor": 0.4, "com_detected_pct": 100.0}}
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p, d


def test_validate_analysis_ok(tmp_path):
    p, _ = _good_analysis(tmp_path)
    assert validate_analysis(str(p), w=1080, h=1920) == []


def test_validate_analysis_catches_state_inconsistency(tmp_path):
    bad = [good_frame(0, state="4pt", n=3)]  # says 4pt but 3 contacts
    p, _ = _good_analysis(tmp_path, bad)
    errs = validate_analysis(str(p))
    assert any("state" in e and "expected" in e for e in errs)


def test_validate_analysis_catches_count_mismatch(tmp_path):
    f = good_frame(0)
    f["n"] = 2  # != actual contacts
    p, _ = _good_analysis(tmp_path, [f])
    assert any("n=2" in e for e in validate_analysis(str(p)))


def test_validate_analysis_catches_bad_angle_and_margin(tmp_path):
    f = good_frame(0)
    f["angles"] = {"lel": 200.0}
    f["margin"] = -0.5
    p, _ = _good_analysis(tmp_path, [f])
    errs = validate_analysis(str(p))
    assert any("angle" in e for e in errs)
    assert any("margin" in e for e in errs)


def test_validate_analysis_catches_com_outside_and_low_rate(tmp_path):
    frames = [good_frame(i) for i in range(10)]
    frames[0]["com"] = [5000.0, 800.0]
    frames[1]["com"] = None
    frames[2]["com"] = None
    p, _ = _good_analysis(tmp_path, frames)
    errs = validate_analysis(str(p), min_com=0.85, w=1080)
    assert any("outside frame" in e for e in errs)
    assert any("com detected" in e for e in errs)


def test_validate_analysis_catches_stats_mismatch_and_bad_event(tmp_path):
    frames = [good_frame(i) for i in range(10)]
    p, d = _good_analysis(tmp_path, frames)
    d["stats"]["state_s"] = {"3pt": 9.9}  # sum != 10/30s duration
    d["events"].append({"t0": 2.0, "t1": 1.0, "dur": -1, "moved": [],
                        "dx": 0, "dy": 0, "disp": 0})
    p.write_text(json.dumps(d), encoding="utf-8")
    errs = validate_analysis(str(p))
    assert any("state_s sum" in e for e in errs)
    assert any("t1<=t0" in e for e in errs)
    assert any("no moved limbs" in e for e in errs)
