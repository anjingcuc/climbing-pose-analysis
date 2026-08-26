import numpy as np

from conftest import KP_ORDER, person
from tech_moves import detect

FPS = 30.0
TORSO = 120.0
BASE_STATS = {"fps": FPS, "torso_px": TORSO}


def build_frames(n, base=(540, 800), contacts=("lh", "rh", "lf", "rf"),
                 climbing=True, edits=None):
    """analysis.json-style frames; edits: {frame_idx: {kpt_name: (x, y)}}."""
    frames = []
    for i in range(n):
        k = person(base[0], base[1], TORSO)
        for j, name in enumerate(KP_ORDER):
            if edits and i in edits and name in edits[i]:
                k[j][0], k[j][1] = edits[i][name]
        cs = {c: (c in contacts) for c in ("lh", "rh", "lf", "rf")}
        frames.append({
            "i": i, "t": i / FPS, "kpts": [p[:2] for p in k],
            "angles": {"lel": 90.0, "rel": 90.0, "lknee": 170.0, "rknee": 170.0},
            "contacts": cs, "n": sum(cs.values()), "state": "3pt",
            "st_t0": 0.0, "bd": 0.1, "margin": 0.3, "inside": True,
            "climbing": climbing, "com": [base[0], base[1]],
        })
    return frames


def _names(out, kind):
    items = out["phases"] if kind == "phases" else out["events"]
    return [it["name"] for it in items]


# ---------- phases ----------

def test_side_on_detected_when_shoulders_collapse():
    edits = {i: {"lsho": (400, 680), "rsho": (420, 680)} for i in range(40)}
    out = detect(build_frames(60, edits=edits), BASE_STATS)
    assert "side_on" in _names(out, "phases")


def test_side_on_not_fired_when_frontal():
    out = detect(build_frames(60), BASE_STATS)  # natural wide shoulders
    assert "side_on" not in _names(out, "phases")


def test_heel_hook_when_ankle_above_knee():
    # right ankle raised above the right knee while rf contacts
    edits = {}
    for i in range(40):
        k = person(540, 800, TORSO)
        rk = k[KP_ORDER.index("rknee")]
        edits[i] = {"rank": (rk[0] + 30, rk[1] - 0.5 * TORSO)}
    out = detect(build_frames(60, edits=edits), BASE_STATS)
    hooks = [p for p in out["phases"] if p["name"] == "heel_hook"]
    assert hooks and hooks[0]["side"] == "右"


def test_heel_hook_not_fired_when_foot_below_knee():
    out = detect(build_frames(60), BASE_STATS)
    assert "heel_hook" not in _names(out, "phases")


def test_cross_feet_when_ankle_order_flips():
    edits = {}
    for i in range(30, 48):  # 0.6s of flipped ankles
        edits[i] = {"lank": (610, 920), "rank": (470, 920)}
    out = detect(build_frames(80, edits=edits), BASE_STATS)
    assert "cross_feet" in _names(out, "phases")


def test_match_hands_when_wrists_together():
    edits = {i: {"lwr": (540, 788), "rwr": (548, 788)} for i in range(30)}
    out = detect(build_frames(60, edits=edits), BASE_STATS)
    assert "match_hands" in _names(out, "phases")


def test_straight_arm_when_both_elbows_locked():
    frames = build_frames(60, contacts=("lh", "rh", "lf"))
    for f in frames:
        f["angles"] = {"lel": 168.0, "rel": 172.0,
                       "lknee": 170.0, "rknee": 170.0}
    out = detect(frames, BASE_STATS)
    assert "straight_arm" in _names(out, "phases")


def test_flagging_when_free_leg_extended_sideways():
    # right foot free, swung far right of the support base, knee straight
    edits = {}
    for i in range(40):
        edits[i] = {"rank": (540 + 1.4 * TORSO, 950)}
    frames = build_frames(60, contacts=("lh", "rh", "lf"), edits=edits)
    out = detect(frames, BASE_STATS)
    flags = [p for p in out["phases"] if p["name"] == "flagging"]
    assert flags and flags[0]["side"] == "右"


# ---------- events ----------

def test_high_step_on_elevated_recontact():
    # left foot planted 30f -> off 12f -> recontact high above the other knee
    n = 80
    edits = {}
    hi_y = 800 - 0.9 * TORSO  # well above hips/knees
    for i in range(42, 60):
        edits[i] = {"lank": (470, hi_y)}
    frames = build_frames(n, contacts=("lh", "rh", "rf"), edits=edits)
    for i in range(30, 42):  # lf lifts (was contacting in builder? no: rf only)
        pass
    frames = [dict(f) for f in frames]
    for i in range(n):  # lf contact only when planted or re-planted
        frames[i]["contacts"]["lf"] = (i < 30) or (i >= 42)
        frames[i]["contacts"]["rf"] = True
        frames[i]["n"] = sum(frames[i]["contacts"].values())
    out = detect(frames, BASE_STATS)
    hs = [e for e in out["events"] if e["name"] == "high_step"]
    assert hs and hs[0]["side"] == "左脚"


def test_foot_swap_when_opposite_foot_takes_same_spot():
    n = 70
    frames = build_frames(n, contacts=("lh", "rh"))
    spot = (470, 920)
    for i in range(n):
        k = frames[i]["kpts"]
        frames[i]["contacts"]["lf"] = i < 30
        frames[i]["contacts"]["rf"] = i >= 40
        frames[i]["n"] = 2 + frames[i]["contacts"]["lf"] + frames[i]["contacts"]["rf"]
        frames[i]["kpts"][15] = list(spot)          # lank at the spot
        frames[i]["kpts"][16] = list(spot)          # rank takes the same spot
    out = detect(frames, BASE_STATS)
    assert "foot_swap" in _names(out, "events")


def test_rock_over_after_foothold_recontact():
    n = 70
    frames = build_frames(n, contacts=("lh", "rh", "rf"))
    ank_x = 480.0
    for i in range(n):
        frames[i]["kpts"][15] = [ank_x, 920]        # left foothold
        # lf planted 0-19, lifts 10 frames, re-plants at 30
        frames[i]["contacts"]["lf"] = (i < 20) or (i >= 30)
        frames[i]["n"] = sum(frames[i]["contacts"].values())
        if i < 30:
            frames[i]["com"] = [ank_x + 120, 820]   # CoM right of the foot
        else:
            u = min(1.0, (i - 30) / 15)             # gradual rock-over
            frames[i]["com"] = [ank_x + 120 - 110 * u, 820 - 36 * u]
    out = detect(frames, BASE_STATS)
    assert "rock_over" in _names(out, "events")


def test_nothing_on_ground_demo():
    out = detect(build_frames(60, climbing=False), BASE_STATS)
    assert out["phases"] == [] and out["events"] == []


# ---------- v2 library: drop_knee / lock_off / knee_bar / dyno / mantle ----------

def test_drop_knee_when_knee_twists_inside():
    # right knee pulled toward midline, right foot planted outside it
    edits = {}
    for i in range(40):
        edits[i] = {"rknee": (546, 860), "rknee_ang": None}
    frames = build_frames(60, edits=edits)
    for f in frames:
        f["angles"]["rknee"] = 120.0   # bent
    out = detect(frames, BASE_STATS)
    dk = [p for p in out["phases"] if p["name"] == "drop_knee"]
    assert dk and dk[0]["side"] == "右"


def test_drop_knee_not_fired_neutral_stance():
    frames = build_frames(60)
    for f in frames:
        f["angles"]["rknee"] = 170.0
    out = detect(frames, BASE_STATS)
    assert "drop_knee" not in _names(out, "phases")


def test_lock_off_when_one_arm_bent_other_hand_moves():
    frames = build_frames(60, contacts=("lh", "lf", "rf"))
    for f in frames:
        f["angles"]["lel"] = 75.0      # left arm loaded and bent
        f["angles"]["rel"] = 140.0
    out = detect(frames, BASE_STATS)
    lo = [p for p in out["phases"] if p["name"] == "lock_off"]
    assert lo and lo[0]["side"] == "左手"


def test_knee_bar_when_free_ankle_on_opposite_knee():
    # left ankle hooked at the right knee point, left foot free
    k = person(540, 800, TORSO)
    rk = k[KP_ORDER.index("rknee")]
    edits = {i: {"lank": (rk[0], rk[1])} for i in range(40)}
    frames = build_frames(60, contacts=("lh", "rh", "rf"), edits=edits)
    out = detect(frames, BASE_STATS)
    kb = [p for p in out["phases"] if p["name"] == "knee_bar"]
    assert kb and kb[0]["side"] == "左"


def test_dyno_when_both_feet_leave_and_com_launches():
    frames = build_frames(60, contacts=("lh", "rh", "lf", "rf"))
    for i in range(15, 33):           # both feet off for 0.6s, hand off too
        frames[i]["contacts"] = {"lh": True, "rh": False,
                                 "lf": False, "rf": False}
    # CoM launches upward (y decreases fast) through the flight
    for i in range(15, 34):
        frames[i]["com"] = [540.0, 800.0 - 1.2 * TORSO * (i - 14) / 8.0]
    out = detect(frames, BASE_STATS)
    assert "dyno" in _names(out, "events")


def test_mantle_when_low_hands_extend_and_com_rises():
    k = person(540, 800, TORSO)
    frames = build_frames(60, contacts=("lh", "rh", "lf", "rf"))
    sho_y = k[KP_ORDER.index("lsho")][1]
    for i in range(60):
        # wrists below the shoulder line (palms on a low shelf)
        frames[i]["kpts"][KP_ORDER.index("lwr")][1] = sho_y + 0.3 * TORSO
        frames[i]["kpts"][KP_ORDER.index("rwr")][1] = sho_y + 0.3 * TORSO
        if i < 20:
            frames[i]["angles"]["lel"] = frames[i]["angles"]["rel"] = 100.0
            frames[i]["com"] = [540.0, 800.0]
        else:
            frames[i]["angles"]["lel"] = frames[i]["angles"]["rel"] = 160.0
            frames[i]["com"] = [540.0, 800.0 - 0.5 * TORSO]
    out = detect(frames, BASE_STATS)
    assert "mantle" in _names(out, "events")


# ---------- tightened precision gates ----------

def test_match_hands_not_fired_when_both_feet_off():
    """Hands passing mid-move used to false-positive match: now a foot
    must be planted."""
    edits = {i: {"lwr": (540, 788), "rwr": (548, 788)} for i in range(30)}
    frames = build_frames(60, contacts=("lh", "rh"), edits=edits)
    out = detect(frames, BASE_STATS)
    assert "match_hands" not in _names(out, "phases")


def test_match_hands_not_fired_at_loose_wrist_gap():
    edits = {i: {"lwr": (540, 788), "rwr": (578, 788)} for i in range(30)}
    out = detect(build_frames(60, edits=edits), BASE_STATS)
    assert "match_hands" not in _names(out, "phases")


def test_flagging_not_fired_when_other_foot_also_free():
    """A flag needs a planted support foot - both feet off is a dynamic
    move, not a flag."""
    edits = {i: {"lank": (300.0, 860.0)} for i in range(40)}  # swung out
    frames = build_frames(60, contacts=("lh", "rh"), edits=edits)
    for f in frames:
        f["angles"]["lknee"] = 170.0
    out = detect(frames, BASE_STATS)
    assert "flagging" not in _names(out, "phases")
