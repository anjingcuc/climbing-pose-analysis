"""Extract per-frame pose keypoints from a climbing video using YOLO11x-pose + BoT-SORT tracking on GPU.

Output: JSON {meta, frames:[{i, t, track_id, bbox, kpts:[[x,y,conf]*17]}]}
Keypoint order is COCO-17:
0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear, 5 L-shoulder, 6 R-shoulder,
7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
"""
import argparse
import json
from pathlib import Path

import numpy as np

COCO_NAMES = [
    "nose", "leye", "reye", "lear", "rear",
    "lsho", "rsho", "lel", "rel", "lwr", "rwr",
    "lhip", "rhip", "lknee", "rknee", "lank", "rank",
]

HIP_GAIN = 250.0  # px gate for single-target continuity


def hip_mid(kpts):
    """Hip midpoint of one person's (17,3) keypoints, or None if unusable."""
    l, r = kpts[11], kpts[12]
    if l[2] > 0.3 and r[2] > 0.3:
        return ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2)
    return None


def select_target(xyxy, all_kpts, prev_hip):
    """Index of the climber among this frame's detections.

    Strategy: candidates ordered highest-first (the climber is above the
    spotter); take the first whose hip is within HIP_GAIN px of the previous
    selection; if none qualifies (or no history), take the highest.
    """
    n = len(xyxy)
    order = sorted(range(n), key=lambda c: (xyxy[c, 1] + xyxy[c, 3]) / 2)
    if prev_hip is not None:
        for c in order:
            hm = hip_mid(all_kpts[c])
            if hm is None:
                continue
            d = ((hm[0] - prev_hip[0]) ** 2 + (hm[1] - prev_hip[1]) ** 2) ** 0.5
            if d < HIP_GAIN:
                return c
    return order[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    from ultralytics import YOLO  # deferred: keeps unit tests import-light
    model = YOLO(args.model)
    results = model.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        conf=args.conf,
        iou=0.5,
        imgsz=args.imgsz,
        device=args.device,
        classes=[0],
        verbose=False,
    )

    frames = []
    track_area = {}  # track_id -> total bbox area (persistence evidence)
    prev_hip = None  # hip-midpoint of the selected person (single-target follower)

    for i, r in enumerate(results):
        fr = {"i": i, "t": i / 30.0, "track_id": None, "bbox": None, "kpts": None}
        if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else None
            xyxy = r.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            all_kpts = r.keypoints.data.cpu().numpy()
            if ids is not None:
                for tid, a in zip(ids, areas):
                    track_area[int(tid)] = track_area.get(int(tid), 0.0) + float(a)
            k = select_target(xyxy, all_kpts, prev_hip)
            prev_hip = hip_mid(all_kpts[k]) or prev_hip
            fr["bbox"] = [round(float(v), 1) for v in xyxy[k]]
            fr["kpts"] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                          for x, y, c in all_kpts[k]]
            if ids is not None:
                fr["track_id"] = int(ids[k])
        frames.append(fr)

    # frame timing from actual fps
    meta = {
        "video": str(Path(args.video).resolve()),
        "model": args.model,
        "fps": float(getattr(r, "fps", 30) if r is not None else 30),
        "n_frames": len(frames),
        "names": COCO_NAMES,
        "track_area": {str(k): round(v, 1) for k, v in
                       sorted(track_area.items(), key=lambda kv: -kv[1])[:8]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f, ensure_ascii=False)
    det = sum(1 for fr in frames if fr["kpts"] is not None)
    print(f"frames={len(frames)} detected={det} ({det/max(1,len(frames))*100:.1f}%)")
    print("top tracks by area:", meta["track_area"])


if __name__ == "__main__":
    main()
