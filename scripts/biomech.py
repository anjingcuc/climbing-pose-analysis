"""Climbing biomechanics analysis from COCO-17 keypoints.

Pipeline: raw keypoints -> confidence interpolation -> Savitzky-Golay smoothing
-> joint angles -> whole-body CoM (Winter/Dempster anthropometric table)
-> limb contact detection (velocity hysteresis) -> support polygon /
stability margin -> 3-point-balance state machine -> barn-door moment arm
-> CoG transfer events -> aggregate stats.

2D image-plane approximation: valid for climbing, where movement is mostly
parallel to the wall plane.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# Winter (2009) Table 4.1, Dempster-based. mass fraction / CM from proximal end
SEGMENTS = {
    "head":   {"m": 0.081,  "f": None},   # placed at ear-mid (fallback: nose)
    "trunk":  {"m": 0.497,  "f": 0.45},   # hip-mid -> shoulder-mid
    "uarm":   {"m": 0.028,  "f": 0.436},  # shoulder -> elbow
    "farm":   {"m": 0.016,  "f": 0.682},  # elbow -> wrist
    "hand":   {"m": 0.006,  "f": 1.0},    # approximated at wrist
    "thigh":  {"m": 0.100,  "f": 0.433},  # hip -> knee
    "shank":  {"m": 0.0465, "f": 0.433},  # knee -> ankle
    "foot":   {"m": 0.0145, "f": 1.0},    # approximated at ankle
}

KP = {n: i for i, n in enumerate(
    ["nose", "leye", "reye", "lear", "rear",
     "lsho", "rsho", "lel", "rel", "lwr", "rwr",
     "lhip", "rhip", "lknee", "rknee", "lank", "rank"])}

CONTACT_KP = {"lh": "lwr", "rh": "rwr", "lf": "lank", "rf": "rank"}
LIMB_CN = {"lh": "左手", "rh": "右手", "lf": "左脚", "rf": "右脚"}


def clean_keypoints(frames, min_conf=0.30):
    """Stack keypoints, interpolate low-confidence gaps, return (N,17,2) + conf."""
    N = len(frames)
    P = np.full((N, 17, 2), np.nan)
    C = np.zeros((N, 17))
    for fr in frames:
        if fr["kpts"] is None:
            continue
        i = fr["i"]
        for j, (x, y, c) in enumerate(fr["kpts"]):
            if c >= min_conf:
                P[i, j] = (x, y)
                C[i, j] = c
    for j in range(17):
        ok = ~np.isnan(P[:, j, 0])
        if ok.sum() >= 2:
            t = np.arange(N)
            P[:, j, 0] = np.interp(t, t[ok], P[ok, j, 0])
            P[:, j, 1] = np.interp(t, t[ok], P[ok, j, 1])
            C[:, j] = np.interp(t, t[ok], C[ok, j])
        else:
            P[:, j] = np.nan
            C[:, j] = 0.0
    return P, C


def smooth(P, win=9, order=2):
    S = P.copy()
    n = len(P)
    for j in range(17):
        for d in range(2):
            col = S[:, j, d]
            bad = np.isnan(col)
            if bad.all():
                col[:] = 0.0
                continue
            if bad.any():
                col[bad] = np.nanmedian(col)
    w = min(win, n if n % 2 == 1 else n - 1)
    if w >= order + 2:
        for j in range(17):
            for d in range(2):
                S[:, j, d] = savgol_filter(S[:, j, d], w, order)
    return S


def angle_at(a, b, c):
    """Angle ABC in degrees at b, NaN-safe vectorized."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compute_angles(P):
    A = {}
    for side in "lr":
        s, e, w = KP[f"{side}sho"], KP[f"{side}el"], KP[f"{side}wr"]
        h, k, a = KP[f"{side}hip"], KP[f"{side}knee"], KP[f"{side}ank"]
        A[f"{side}el"] = angle_at(P[:, s], P[:, e], P[:, w])    # elbow flexion
        A[f"{side}sho"] = angle_at(P[:, e], P[:, s], P[:, h])   # shoulder
        A[f"{side}hip"] = angle_at(P[:, s], P[:, h], P[:, k])   # hip flexion
        A[f"{side}knee"] = angle_at(P[:, h], P[:, k], P[:, a])  # knee flexion
    return A


def body_scale(P):
    """Median shoulder-mid to hip-mid distance = 1 body unit (torso)."""
    sho_mid = (P[:, KP["lsho"]] + P[:, KP["rsho"]]) / 2
    hip_mid = (P[:, KP["lhip"]] + P[:, KP["rhip"]]) / 2
    d = np.linalg.norm(sho_mid - hip_mid, axis=-1)
    return float(np.nanmedian(d)), sho_mid, hip_mid


def compute_com(P):
    """Whole-body CoM as mass-weighted sum of segment CoMs (2D projection)."""
    N = len(P)
    com = np.full((N, 2), np.nan)
    ok_frame = ~np.isnan(P[:, :, 0]).any(axis=1)
    if ok_frame.sum() == 0:
        return com

    def seg(p_prox, p_dist, f):
        return p_prox + f * (p_dist - p_prox)

    parts = []
    for side in "lr":
        s, e, w = P[:, KP[f"{side}sho"]], P[:, KP[f"{side}el"]], P[:, KP[f"{side}wr"]]
        h, k, a = P[:, KP[f"{side}hip"]], P[:, KP[f"{side}knee"]], P[:, KP[f"{side}ank"]]
        parts += [
            (SEGMENTS["uarm"]["m"], seg(s, e, SEGMENTS["uarm"]["f"])),
            (SEGMENTS["farm"]["m"], seg(e, w, SEGMENTS["farm"]["f"])),
            (SEGMENTS["hand"]["m"], w),
            (SEGMENTS["thigh"]["m"], seg(h, k, SEGMENTS["thigh"]["f"])),
            (SEGMENTS["shank"]["m"], seg(k, a, SEGMENTS["shank"]["f"])),
            (SEGMENTS["foot"]["m"], a),
        ]
    sho_mid = (P[:, KP["lsho"]] + P[:, KP["rsho"]]) / 2
    hip_mid = (P[:, KP["lhip"]] + P[:, KP["rhip"]]) / 2
    parts.append((SEGMENTS["trunk"]["m"], seg(hip_mid, sho_mid, SEGMENTS["trunk"]["f"])))
    ear_mid = (P[:, KP["lear"]] + P[:, KP["rear"]]) / 2
    head_pt = np.where(np.isnan(ear_mid[:, :1]), P[:, KP["nose"]], ear_mid)
    parts.append((SEGMENTS["head"]["m"], head_pt))

    m_tot = sum(m for m, _ in parts)
    acc = sum(m * pt for m, pt in parts)
    com[ok_frame] = acc[ok_frame] / m_tot
    return com


def detect_contacts(P, torso, fps, k=5,
                    on_t=0.09, off_t=0.45, on_frames=4, off_frames=3):
    """Contact = limb endpoint stationary (hysteresis, normalized by torso).

    Speed is estimated from centered displacement over +/-k frames so keypoint
    jitter is averaged out, then median-filtered before hysteresis.
    """
    from scipy.ndimage import median_filter
    N = len(P)
    idx = np.arange(N)
    iu = np.minimum(idx + k, N - 1)
    il = np.maximum(idx - k, 0)
    dt = (iu - il) / fps
    dt[dt == 0] = 1 / fps
    speed = np.zeros((N, 17))
    for j in range(17):
        d = np.linalg.norm(P[iu, j] - P[il, j], axis=-1)
        speed[:, j] = median_filter(d / dt / max(torso, 1e-6), size=5)
    contacts = {}
    for name, kp in CONTACT_KP.items():
        sp = speed[:, KP[kp]]
        on = np.zeros(N, bool)
        cnt_on = cnt_off = 0
        state = False
        for i in range(N):
            s = sp[i] if not np.isnan(sp[i]) else 99.0
            if state:
                cnt_off = cnt_off + 1 if s > off_t else 0
                if cnt_off >= off_frames:
                    state, cnt_off = False, 0
            else:
                cnt_on = cnt_on + 1 if s < on_t else 0
                if cnt_on >= on_frames:
                    state, cnt_on = True, 0
            on[i] = state
        contacts[name] = on
    return contacts, speed


def point_in_tri(p, a, b, c, eps=1e-9):
    """Barycentric inside test + signed distance to nearest edge (px)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < eps:
        return False, 0.0
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    inside = (u >= 0) and (v >= 0) and (u + v <= 1)

    def cross2(u_, v_):
        return u_[0] * v_[1] - u_[1] * v_[0]

    dmin = min(
        abs(cross2(b - a, p - a)) / max(np.linalg.norm(b - a), eps),
        abs(cross2(c - b, p - b)) / max(np.linalg.norm(c - b), eps),
        abs(cross2(a - c, p - c)) / max(np.linalg.norm(a - c), eps),
    )
    return inside, float(dmin)


def dist_point_seg(p, a, b):
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.clip((p - a) @ ab / (L * L), 0, 1)
    return float(np.linalg.norm(p - a - t * ab))


def analyze(pose_json_path, out_json_path, fps=None, on_t=0.09, off_t=0.45):
    data = json.loads(Path(pose_json_path).read_text(encoding="utf-8"))
    frames = data["frames"]
    fps = fps or data["meta"].get("fps", 30.0)
    N = len(frames)
    P, C = clean_keypoints(frames)
    S = smooth(P)
    A = compute_angles(S)
    com = compute_com(S)
    torso, sho_mid, hip_mid = body_scale(S)

    contacts, speed = detect_contacts(S, torso, fps, on_t=on_t, off_t=off_t)
    off_ground = np.zeros(N, bool)
    # ground baseline from the lowest hip positions (standing / resting on mats);
    # climbing = hips raised > 0.6 torso above that baseline
    hip_y = hip_mid[:, 1]
    stand_y = np.nanpercentile(hip_y, 90)
    climbing = hip_y < (stand_y - 0.6 * torso)

    hand_cn = {"lh", "rh"}
    foot_cn = {"lf", "rf"}

    out_frames = []
    barndoor = np.zeros(N)
    state_t0 = np.zeros(N)  # time when the current state began
    for i in range(N):
        cs = {k: bool(contacts[k][i]) for k in CONTACT_KP}
        n = sum(cs.values())
        state = "idle"
        if climbing[i] and not bool(np.isnan(com[i, 0])):
            if n >= 4:
                state = "4pt"
            elif n == 3:
                state = "3pt"
            elif n == 2:
                state = "2pt"
            else:
                state = "1pt"
        if i == 0 or out_frames[i - 1]["state"] != state:
            state_t0[i] = i / fps
        else:
            state_t0[i] = state_t0[i - 1]
        pts = {k: S[i, KP[v]] for k, v in CONTACT_KP.items() if cs[k]}
        margin = inside = None
        if state in ("3pt", "4pt") and len(pts) >= 3:
            tri = list(pts.values())[:3] if len(pts) == 3 else \
                sorted(pts.values(), key=lambda p: p[1])[:3]
            inside, margin = point_in_tri(com[i], *tri)
            if state == "3pt":
                # barn-door: gravity moment about the line joining the two
                # same-type contacts, resisted only by the third limb
                same = [p for k, p in pts.items()
                        if (k in hand_cn) == (sum(k in hand_cn for k in pts) >= 2)]
                if len(same) == 2:
                    a, b = same
                    ab = b - a
                    L = max(np.linalg.norm(ab), 1e-9)
                    barndoor[i] = abs(ab[0] * (com[i, 1] - a[1])
                                      - ab[1] * (com[i, 0] - a[0])) / L / torso
        elif state == "2pt" and len(pts) == 2:
            a, b = list(pts.values())
            margin = dist_point_seg(com[i], a, b)
            inside = False
        com_vel = np.linalg.norm(np.gradient(com, axis=0) * fps, axis=-1)[i] \
            if not np.isnan(com[i, 0]) else None
        out_frames.append({
            "i": i, "t": round(i / fps, 3),
            "kpts": [[round(float(x), 1), round(float(y), 1)] if not np.isnan(x) else None
                     for x, y in S[i]],
            "angles": {k: round(float(A[k][i]), 1) for k in A
                       if not np.isnan(A[k][i])},
            "com": [round(float(com[i, 0]), 1), round(float(com[i, 1]), 1)]
                   if not np.isnan(com[i, 0]) else None,
            "com_vel": round(float(com_vel), 2) if com_vel and not np.isnan(com_vel) else None,
            "contacts": cs, "n": n, "state": state,
            "st_t0": round(float(state_t0[i]), 2),
            "bd": round(float(barndoor[i]), 2),
            "margin": round(float(margin / torso), 3) if margin is not None else None,
            "inside": bool(inside) if inside is not None else None,
            "climbing": bool(climbing[i]),
        })

    # CoG transfer events: per-limb contact-break -> recontact episodes.
    # CoM displacement is measured across the whole move cycle
    # (0.4s before break .. 0.6s after recontact), because the weight shift
    # happens while the limb is off, not inside the static 3pt hold.
    episodes = []
    max_gap = int(4.0 * fps)
    for limb in CONTACT_KP:
        on = contacts[limb]
        i = 1
        while i < N:
            if on[i - 1] and not on[i] and on[max(0, i - 8):i].sum() >= 6:
                j = i
                recontact = None
                while j < min(N, i + max_gap):
                    if on[j] and on[j:j + 5].sum() >= 4:
                        recontact = j
                        break
                    j += 1
                if recontact is not None and recontact - i >= int(0.25 * fps):
                    episodes.append((limb, i, recontact))
                i = (recontact or i) + 1
            else:
                i += 1
    # merge time-overlapping episodes of different limbs into one event
    episodes.sort(key=lambda e: e[1])
    merged = []
    for limb, a, b in episodes:
        if merged and a < merged[-1][2]:  # true overlap, not just adjacency
            merged[-1][0].append(limb)
            merged[-1][2] = max(merged[-1][2], b)
        else:
            merged.append([[limb], a, b])
    events = []
    for limbs, a, b in merged:
        if not (climbing[a] or climbing[b]):
            continue  # limb moves on the ground are not transfers
        i0 = max(0, a - int(0.4 * fps))
        i1 = min(N - 1, b + int(0.6 * fps))
        c0, c1 = com[i0], com[i1]
        if np.isnan(c0[0]) or np.isnan(c1[0]):
            continue
        dx = (c1[0] - c0[0]) / torso
        dy = (c1[1] - c0[1]) / torso
        if math.hypot(dx, dy) > 0.20:
            # states just before the break / just after the recontact: most
            # climbing is a chain of 3pt -> 3pt transfers via one moving limb
            events.append({
                "t0": round(a / fps, 2), "t1": round(b / fps, 2),
                "dur": round((b - a) / fps, 2),
                "moved": [LIMB_CN[k] for k in limbs],
                "dx": round(dx, 2), "dy": round(dy, 2),
                "disp": round(math.hypot(dx, dy), 2),
                "s0": out_frames[max(0, a - 1)]["state"],
                "s1": out_frames[min(N - 1, b + 2)]["state"],
            })

    st = [f["state"] for f in out_frames]
    stats = {
        "fps": fps, "n_frames": N, "torso_px": round(torso, 1),
        "duration_s": round(N / fps, 1),
        "climbing_s": round(sum(climbing) / fps, 1),
        "state_s": {s: round(st.count(s) / fps, 1)
                    for s in ["4pt", "3pt", "2pt", "1pt", "idle"]},
        "transfers": len(events),
        "max_barndoor": round(float(barndoor.max()), 2),
        "com_detected_pct": round(100 * sum(1 for f in out_frames if f["com"]) / N, 1),
    }
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json_path).write_text(json.dumps({
        "meta": data["meta"], "stats": stats,
        "frames": out_frames, "events": events,
    }, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"events={len(events)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_json")
    ap.add_argument("out_json")
    ap.add_argument("--on-t", type=float, default=0.09,
                    help="contact-on speed threshold, torso/s (raise when the "
                         "climber is small in frame and keypoint noise is high)")
    ap.add_argument("--off-t", type=float, default=0.45,
                    help="contact-off speed threshold, torso/s")
    a = ap.parse_args()
    analyze(a.pose_json, a.out_json, on_t=a.on_t, off_t=a.off_t)
