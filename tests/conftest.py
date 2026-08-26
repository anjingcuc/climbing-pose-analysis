import sys
from pathlib import Path

import numpy as np
import pytest

# works both in-repo (scripts beside tests/) and in-skill (scripts/ subdir)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# canonical COCO-17 layout, in units of torso length, origin at hip-mid
CANON = {
    "nose": (0.0, -1.25), "leye": (-0.08, -1.32), "reye": (0.08, -1.32),
    "lear": (-0.16, -1.25), "rear": (0.16, -1.25),
    "lsho": (-0.45, -1.0), "rsho": (0.45, -1.0),
    "lel": (-0.60, -0.55), "rel": (0.60, -0.55),
    "lwr": (-0.55, -0.10), "rwr": (0.55, -0.10),
    "lhip": (-0.25, 0.0), "rhip": (0.25, 0.0),
    "lknee": (-0.30, 0.5), "rknee": (0.30, 0.5),
    "lank": (-0.32, 1.0), "rank": (0.32, 1.0),
}
KP_ORDER = ["nose", "leye", "reye", "lear", "rear",
            "lsho", "rsho", "lel", "rel", "lwr", "rwr",
            "lhip", "rhip", "lknee", "rknee", "lank", "rank"]


def person(cx, cy, torso=120.0, conf=0.9, dx=0.0):
    """17 keypoints of a standing person centered at hip-mid (cx, cy)."""
    kpts = []
    for nm in KP_ORDER:
        x, y = CANON[nm]
        kpts.append([cx + (x + dx) * torso, cy + y * torso, conf])
    return kpts


def person_xy(cx, cy, torso=120.0):
    """(17,2) ndarray variant for biomechanics functions."""
    return np.array(person(cx, cy, torso))[:, :2]


def pose_frames_builder():
    """Frame factory for synthetic pose JSONs."""
    frames = []

    def add(i, kpts, bbox=None, track_id=1, fps=30.0):
        frames.append({"i": i, "t": i / fps, "track_id": track_id,
                       "bbox": bbox or [0, 0, 1080, 1920], "kpts": kpts})

    def build():
        return frames

    return add, build


@pytest.fixture
def synth_pose(tmp_path):
    """150-frame synthetic clip: stand (30f) -> climb static (30f) ->
    shift torso right while the right hand reaches a new hold, left wrist and
    both ankles stay fixed (16f) -> settle static (74f)."""
    add, build = pose_frames_builder()
    W_MID, STAND_Y, WALL_Y, T = 540, 1500, 800, 120.0
    pose_a = person(W_MID, WALL_Y, T)
    for i in range(150):
        if i < 30:                       # standing on the mat
            k = person(W_MID, STAND_Y, T)
        elif i < 60:                     # on wall, base pose A
            k = person(W_MID, WALL_Y, T)
        elif i < 76:                     # torso shifts, right hand reaches
            u = (i - 60) / 15.0
            k = person(W_MID + 50 * u, WALL_Y - 20 * u, T)
            k[10][0] += 200 * u          # right wrist travels to next hold
            k[10][1] -= 60 * u
            k[9] = list(pose_a[9])       # left wrist stays on its hold
            k[15] = list(pose_a[15])     # feet stay planted
            k[16] = list(pose_a[16])
        else:                            # pose B static
            k = person(W_MID + 50, WALL_Y - 20, T)
            k[10][0] += 200
            k[10][1] -= 60
            k[9] = list(pose_a[9])
            k[15] = list(pose_a[15])
            k[16] = list(pose_a[16])
        add(i, k)
    path = tmp_path / "pose.json"
    path.write_text(__import__("json").dumps({"meta": {"fps": 30}, "frames": build()}),
                    encoding="utf-8")
    return path
