"""Quick OpenCV debug render of the analysis overlay (validation before hyperframes)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
ANGLE_AT = {"lel": (5, 7, 9), "rel": (6, 8, 10), "lsho": (7, 5, 11),
            "rsho": (8, 6, 12), "lhip": (5, 11, 13), "rhip": (6, 12, 14),
            "lknee": (11, 13, 15), "rknee": (12, 14, 16)}
CONTACT_KP = {"lh": 9, "rh": 10, "lf": 15, "rf": 16}
STATE_TXT = {"4pt": "四点支撑", "3pt": "三点平衡", "2pt": "两点·动态",
             "1pt": "移动中", "idle": "休息"}
STATE_CLR = {"4pt": (90, 180, 255), "3pt": (90, 255, 120), "2pt": (90, 160, 255),
             "1pt": (90, 90, 255), "idle": (160, 160, 160)}


def main(analysis_json, video, out_mp4, trail=90):
    d = json.loads(Path(analysis_json).read_text(encoding="utf-8"))
    frames = d["frames"]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    com_hist = []
    for f in frames:
        ok, img = cap.read()
        if not ok:
            break
        if f["kpts"]:
            K = {i: (int(p[0]), int(p[1])) for i, p in enumerate(f["kpts"]) if p}
            for a, b in BONES:
                if a in K and b in K:
                    cv2.line(img, K[a], K[b], (255, 200, 60), 4, cv2.LINE_AA)
            for i, p in K.items():
                cv2.circle(img, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, 6, (60, 60, 220), 1, cv2.LINE_AA)
            for name, (a, b, c) in ANGLE_AT.items():
                if name in f.get("angles", {}) and b in K:
                    x, y = K[b]
                    cv2.putText(img, f"{f['angles'][name]:.0f}", (x + 12, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(img, f"{f['angles'][name]:.0f}", (x + 12, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 120, 255), 2, cv2.LINE_AA)
        if f["com"]:
            com_hist.append(f["com"])
            if len(com_hist) > trail:
                com_hist.pop(0)
            pts = np.array(com_hist, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (120, 220, 120), 3, cv2.LINE_AA)
            cx, cy = int(f["com"][0]), int(f["com"][1])
            cv2.circle(img, (cx, cy), 9, (90, 255, 90), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, H), (90, 255, 90), 1, cv2.LINE_AA)
        cps = {k: i for k, i in CONTACT_KP.items() if f["contacts"].get(k)}
        if f["state"] in ("3pt", "4pt") and len(cps) >= 3:
            tri = np.array([f["kpts"][i] for i in list(cps.values())[:3]], np.int32)
            cv2.polylines(img, [tri.reshape(-1, 1, 2)], True, (90, 255, 120), 3, cv2.LINE_AA)
            overlay = img.copy()
            cv2.fillPoly(overlay, [tri.reshape(-1, 1, 2)], (90, 255, 120))
            cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
        for k, i in cps.items():
            if f["kpts"][i]:
                p = (int(f["kpts"][i][0]), int(f["kpts"][i][1]))
                cv2.circle(img, p, 16, (60, 60, 255), 3, cv2.LINE_AA)
        st = f["state"]
        cv2.rectangle(img, (24, 24), (330, 78), (20, 20, 20), -1)
        cv2.putText(img, f"{STATE_TXT.get(st, st)}  n={f['n']}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, STATE_CLR.get(st, (255, 255, 255)), 2, cv2.LINE_AA)
        m = f.get("margin")
        if m is not None:
            cv2.putText(img, f"margin={m:.2f} {'IN' if f.get('inside') else 'OUT'}", (360, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (90, 255, 90) if f.get("inside") else (90, 90, 255), 2, cv2.LINE_AA)
        vw.write(img)
    cap.release()
    vw.release()
    print("wrote", out_mp4)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
