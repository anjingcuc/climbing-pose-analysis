import numpy as np

from conftest import person
from extract_pose import HIP_GAIN, hip_mid, select_target


def _det(cx, cy, torso=120.0, conf=0.9, x0=None, x1=None, y0=None, y1=None):
    """One detection: bbox + keypoints for a person at (cx, cy)."""
    k = np.array(person(cx, cy, torso, conf))
    x0 = x0 if x0 is not None else cx - torso
    x1 = x1 if x1 is not None else cx + torso
    y0 = y0 if y0 is not None else cy - 1.3 * torso
    y1 = y1 if y1 is not None else cy + 1.1 * torso
    return np.array([x0, y0, x1, y1]), k


def test_hip_mid_valid():
    k = np.array(person(540, 800))
    hm = hip_mid(k)
    assert hm == (540.0, 800.0)


def test_hip_mid_low_conf_is_none():
    k = np.array(person(540, 800))
    k[11][2] = 0.1  # left hip unconfident
    assert hip_mid(k) is None


def test_select_target_no_history_picks_highest():
    climber = _det(500, 600)   # higher on screen (smaller y)
    spotter = _det(520, 1500)
    xyxy = np.array([spotter[0], climber[0]])
    kpts = np.array([spotter[1], climber[1]])
    assert select_target(xyxy, kpts, None) == 1


def test_select_target_prefers_nearest_within_gate():
    climber = _det(500, 600)
    spotter = _det(520, 1500)
    xyxy = np.array([spotter[0], climber[0]])
    kpts = np.array([spotter[1], climber[1]])
    prev = (505.0, 610.0)  # near the climber
    assert select_target(xyxy, kpts, prev) == 1


def test_select_target_gate_prefers_highest_when_both_in_gate():
    """Two detections both within the gate: the higher one wins."""
    a = _det(500, 700)
    b = _det(508, 760)
    prev = (505.0, 750.0)  # nearer to b, but a is higher
    xyxy = np.array([b[0], a[0]])
    kpts = np.array([b[1], a[1]])
    assert select_target(xyxy, kpts, prev) == 1


def test_select_target_falls_back_to_highest_when_far():
    climber = _det(500, 600)
    other = _det(500, 620 + HIP_GAIN)  # beyond the gate from prev
    prev = (500.0, 200.0)
    xyxy = np.array([other[0], climber[0]])
    kpts = np.array([other[1], climber[1]])
    assert select_target(xyxy, kpts, prev) == 1  # climber (highest)


def test_select_target_skips_unconfident_hips_in_gate():
    near = _det(500, 700)
    near[1][11][2] = 0.1  # unusable hips -> skipped by the gate
    far = _det(500, 620)
    prev = (500.0, 705.0)  # only 'near' is within the gate
    xyxy = np.array([far[0], near[0]])
    kpts = np.array([far[1], near[1]])
    # gate finds nothing usable -> falls back to highest = far (index 0)
    assert select_target(xyxy, kpts, prev) == 0
