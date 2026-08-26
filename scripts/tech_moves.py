"""Rule-based climbing technique detectors on analysis.json frames.

Implements the detector library selected in TECHNIQUES.md (V0-V5, 2D
geometry reliable), reviewed against published technique guides
(REI / Movement Gyms / r/climbharder library):
phases: high-frequency postures - side_on / heel_hook / cross_feet /
match_hands / straight_arm / flagging / drop_knee / lock_off / knee_bar
events: recontact / dynamic moves - high_step / foot_swap / cross_hands /
rock_over / dyno / mantle

All thresholds are torso-normalized; image y grows downward.
Output: {"phases":[{name,t0,t1,side,conf}], "events":[{name,t,side,conf}]}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_opening, uniform_filter1d

SIDE_CN = {"l": "左", "r": "右"}

# keypoint indices
SHO, EL, WR, HIP, KNEE, ANK = 5, 7, 9, 11, 13, 15


def _stack(frames):
    N = len(frames)
    K = np.full((N, 17, 2), np.nan)
    contacts = {k: np.zeros(N, bool) for k in ("lh", "rh", "lf", "rf")}
    climbing = np.zeros(N, bool)
    com = np.full((N, 2), np.nan)
    angles = {a: np.full(N, np.nan) for a in
              ("lel", "rel", "lknee", "rknee")}
    for i, f in enumerate(frames):
        if f["kpts"]:
            for j, p in enumerate(f["kpts"]):
                if p:
                    K[i, j] = p
        for k in contacts:
            contacts[k][i] = bool(f["contacts"].get(k))
        climbing[i] = bool(f["climbing"])
        if f["com"]:
            com[i] = f["com"]
        for a in angles:
            v = f.get("angles", {}).get(a)
            if v is not None:
                angles[a][i] = v
    return K, contacts, climbing, com, angles


def _phases(mask, fps, min_s, structure=None):
    """Deblipped runs of a boolean mask with a minimum duration."""
    if mask.sum() == 0:
        return []
    m = binary_opening(mask, structure=structure, border_value=0)
    runs, i, N = [], 0, len(m)
    while i < N:
        if m[i]:
            j = i
            while j + 1 < N and m[j + 1]:
                j += 1
            if (j - i + 1) / fps >= min_s:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _smooth_mask(mask, fps, win_s=0.27):
    w = max(3, int(win_s * fps) | 1)
    return uniform_filter1d(mask.astype(float), w) > 0.6


def _episodes(on, fps, min_off_s=0.25, min_pre=6, min_on=4, max_gap_s=4.0):
    """(break_frame, recontact_frame) per limb contact cycle."""
    eps, i, N = [], 0, len(on)
    while i < N:
        if i > 0 and on[i - 1] and not on[i] and on[max(0, i - min_pre):i].sum() >= min_pre * 0.7:
            j, rec = i, None
            while j < min(N, i + int(max_gap_s * fps)):
                if on[j] and on[j:j + min_on].sum() >= min_on - 1:
                    rec = j
                    break
                j += 1
            if rec is not None and (rec - i) / fps >= min_off_s:
                eps.append((i, rec))
            i = (rec or i) + 1
        else:
            i += 1
    return eps


def _hysteresis(enter, exit_):
    """Boolean state machine: latch on `enter`, release on `exit_`."""
    out = np.zeros(len(enter), bool)
    state = False
    for i in range(len(enter)):
        state = (state and not exit_[i]) or bool(enter[i])
        out[i] = state
    return out


def detect(frames, stats):
    fps = stats["fps"]
    torso = stats["torso_px"]
    K, C, climbing, com, A = _stack(frames)
    N = len(frames)
    T = max(torso, 1e-6)
    phases, events = [], []

    def side_pt(i, s, j):
        return K[:, i + (0 if s == "l" else 1)]

    # ---- side_on: projected shoulder width collapses --------------------
    proj_sho = np.abs(K[:, SHO, 0] - K[:, SHO + 1, 0]) / T
    ok = climbing & np.isfinite(proj_sho) & (C["lh"] | C["rh"])
    side_mask = _smooth_mask(_hysteresis(ok & (proj_sho < 0.38),
                                         proj_sho > 0.48), fps)
    for a, b in _phases(side_mask, fps, 0.8):
        conf = min(1.0, 0.5 + (b - a) / fps / 4)
        phases.append({"name": "side_on", "t0": round(a / fps, 2),
                       "t1": round(b / fps, 2), "side": "", "conf": round(conf, 2)})

    # ---- drop_knee (内扣膝): knee twisted in toward the midline while the
    # foot stays planted outside it (hip rolls perpendicular to the wall) ----
    mid_x = (K[:, HIP, 0] + K[:, HIP + 1, 0]) / 2
    for s in ("l", "r"):
        hip_x = K[:, HIP + (0 if s == "l" else 1), 0]
        knee_x = K[:, KNEE + (0 if s == "l" else 1), 0]
        ank_x = K[:, ANK + (0 if s == "l" else 1), 0]
        d_hip = np.abs(hip_x - mid_x)
        d_knee = np.abs(knee_x - mid_x)
        d_ank = np.abs(ank_x - mid_x)
        kang = A["lknee" if s == "l" else "rknee"]
        twisted = (d_knee + 0.08 * T < np.minimum(d_hip, d_ank)) & \
                  np.isfinite(kang) & (kang < 155)
        mask = _smooth_mask(climbing & C["lf" if s == "l" else "rf"] & twisted, fps)
        for a, b in _phases(mask, fps, 0.5):
            phases.append({"name": "drop_knee", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2), "side": SIDE_CN[s],
                           "conf": round(min(1.0, 0.55 + (b - a) / fps / 3), 2)})

    # ---- lock_off (锁臂): DEEP lock (elbow <82 deg) on the supporting arm
    # while the other hand stays off the wall for a real reach. Ordinary
    # one-arm reaching sits around 85-110 deg and must NOT fire this. -------
    for s in ("l", "r"):
        eang = A["lel" if s == "l" else "rel"]
        c_sup = C["lh" if s == "l" else "rh"]
        c_move = C["rh" if s == "l" else "lh"]
        mask = _smooth_mask(climbing & c_sup & ~c_move &
                            np.isfinite(eang) & (eang < 82), fps)
        for a, b in _phases(mask, fps, 0.6):
            phases.append({"name": "lock_off", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2),
                           "side": SIDE_CN[s] + "手",
                           "conf": round(min(1.0, 0.55 + (b - a) / fps / 2), 2)})

    # ---- knee_bar (膝勾): free ankle hooked AT the opposite knee (same
    # height, very close) - a step-through merely passes the knee vertically
    for s in ("l", "r"):
        ank = K[:, ANK + (0 if s == "l" else 1)]
        oknee = K[:, KNEE + (1 if s == "l" else 0)]
        d = np.linalg.norm(ank - oknee, axis=-1)
        dy = np.abs(ank[:, 1] - oknee[:, 1])
        mask = _smooth_mask(climbing & ~C["lf" if s == "l" else "rf"] &
                            np.isfinite(d) & (d < 0.18 * T) &
                            (dy < 0.15 * T), fps)
        for a, b in _phases(mask, fps, 0.6):
            phases.append({"name": "knee_bar", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2), "side": SIDE_CN[s],
                           "conf": 0.6})

    # ---- heel_hook: contacting ankle raised above its own knee with the
    # leg bent (straight-leg raises are high steps, not hooks) -------------
    for s, c_ank, c_knee in (("l", C["lf"], "lknee"), ("r", C["rf"], "rknee")):
        ank_y, knee_y = K[:, ANK + (0 if s == "l" else 1), 1], K[:, KNEE + (0 if s == "l" else 1), 1]
        lift = (knee_y - ank_y) / T  # positive when ankle above knee
        kang = A[c_knee]
        mask = _smooth_mask(climbing & c_ank & (lift > 0.25) &
                            np.isfinite(kang) & (kang < 175), fps)
        for a, b in _phases(mask, fps, 0.5):
            conf = min(1.0, 0.5 + np.nanmax(lift[a:b + 1]) / 1.5)
            phases.append({"name": "heel_hook", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2), "side": SIDE_CN[s],
                           "conf": round(conf, 2)})

    # ---- cross_feet: ankle x-order flips vs the clip median -------------
    dx = K[:, ANK, 0] - K[:, ANK + 1, 0]
    base = np.nanmedian(dx)
    if np.isfinite(base):
        cross_amt = -np.sign(base) * dx  # positive when order is flipped
        # geometric only: during a step-through both feet may be moving, so
        # no contact requirement here — the climbing gate does the filtering
        flipped = _hysteresis(climbing & np.isfinite(cross_amt)
                              & (cross_amt > 0.06 * T),
                              ~(np.isfinite(cross_amt)
                                & (cross_amt > -0.05 * T)))
        mask = _smooth_mask(flipped, fps)
        for a, b in _phases(mask, fps, 0.3):
            phases.append({"name": "cross_feet", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2), "side": "",
                           "conf": round(min(1.0, 0.4 + np.nanmax(
                               cross_amt[a:b + 1]) / T), 2)})

    # ---- match_hands: both wrists together while both contact AND feet
    # planted (tightened 0.35->0.28 torso + foot gate: hands passing each
    # other mid-move used to false-positive here) ---------------------------
    wrd = np.linalg.norm(K[:, WR] - K[:, WR + 1], axis=-1) / T
    mask = _smooth_mask(climbing & C["lh"] & C["rh"] &
                        (C["lf"] | C["rf"]) & (wrd < 0.28), fps)
    for a, b in _phases(mask, fps, 0.4):
        phases.append({"name": "match_hands", "t0": round(a / fps, 2),
                       "t1": round(b / fps, 2), "side": "",
                       "conf": round(min(1.0, 0.6 + (b - a) / fps / 3), 2)})

    # ---- straight_arm rest: both elbows locked while hanging ------------
    ok_el = np.isfinite(A["lel"]) & np.isfinite(A["rel"])
    mask = _smooth_mask(climbing & (C["lh"] | C["rh"]) & ok_el
                        & (A["lel"] > 158) & (A["rel"] > 158), fps)
    for a, b in _phases(mask, fps, 1.0):
        phases.append({"name": "straight_arm", "t0": round(a / fps, 2),
                       "t1": round(b / fps, 2), "side": "",
                       "conf": round(min(1.0, 0.5 + (b - a) / fps / 4), 2)})

    # ---- flagging: free foot straight, swung outside the support base,
    # while the OTHER foot carries (planted) -------------------------------
    xs = {"lh": K[:, WR, 0], "rh": K[:, WR + 1, 0],
          "lf": K[:, ANK, 0], "rf": K[:, ANK + 1, 0]}
    for s, c_key, knee_ang in (("l", "lf", "lknee"), ("r", "rf", "rknee")):
        sup_foot = C["rf" if s == "l" else "lf"]
        free = ~C[c_key]
        sup = [xs[k] for k in ("lh", "rh", "lf", "rf") if k != c_key]
        lo = np.nanmin(np.stack(sup), axis=0)
        hi = np.nanmax(np.stack(sup), axis=0)
        ax = xs[c_key]
        outside = (ax < lo - 0.5 * T) | (ax > hi + 0.5 * T)
        straight = A[knee_ang] > 150
        mask = _smooth_mask(climbing & free & sup_foot & outside & straight, fps)
        for a, b in _phases(mask, fps, 0.5):
            phases.append({"name": "flagging", "t0": round(a / fps, 2),
                           "t1": round(b / fps, 2), "side": SIDE_CN[s],
                           "conf": round(min(1.0, 0.5 + (b - a) / fps / 2), 2)})

    # ---- high_step: recontact with the foot above the other knee --------
    def ankle_y(s):
        return K[:, ANK + (0 if s == "l" else 1), 1]

    def knee_y(s):
        return K[:, KNEE + (0 if s == "l" else 1), 1]

    def hip_y(s):
        return K[:, HIP + (0 if s == "l" else 1), 1]

    cooldown = {}
    for s, c_key in (("l", "lf"), ("r", "rf")):
        for brk, rec in _episodes(C[c_key], fps):
            if not climbing[rec]:
                continue
            margin = (knee_y("r" if s == "l" else "l")[rec] - ankle_y(s)[rec]) / T
            above_hip = (hip_y(s)[rec] - ankle_y(s)[rec]) / T
            if margin > 0.12 and above_hip > 0.05 and rec - cooldown.get(s, -999) > 30:
                cooldown[s] = rec
                conf = min(1.0, 0.6 + margin / 1.2)
                events.append({"name": "high_step", "t": round(rec / fps, 2),
                               "side": SIDE_CN[s] + "脚", "conf": round(conf, 2)})

    # ---- foot_swap: A leaves, B lands on the same spot soon after -------
    # (A does not come back — a swap, not a step-and-return)
    def break_points(on, min_pre=6):
        pts = []
        for i in range(1, N):
            if on[i - 1] and not on[i] and on[max(0, i - min_pre):i].sum() >= min_pre * 0.7:
                pts.append(i)
        return pts

    def first_landing(on, after, min_on=4):
        i = after
        while i < N:
            if on[i] and on[i:i + min_on].sum() >= min_on - 1:
                return i
            i += 1
        return None

    for a_key, b_key in (("lf", "rf"), ("rf", "lf")):
        a_idx = ANK + (0 if a_key == "lf" else 1)
        b_idx = ANK + (0 if b_key == "lf" else 1)
        for brk in break_points(C[a_key]):
            p_leave = K[brk - 1, a_idx]
            if not np.isfinite(p_leave).all() or C[b_key][brk - 1]:
                continue  # B already standing there -> not a swap
            land = first_landing(C[b_key], brk)
            if land is None or land - brk > int(0.9 * fps):
                continue
            p_land = K[land, b_idx]
            if np.isfinite(p_land).all() and \
                    np.hypot(*(p_land - p_leave)) < 0.35 * T and climbing[land]:
                events.append({"name": "foot_swap", "t": round(land / fps, 2),
                               "side": "", "conf": 0.85})

    # ---- cross_hands: moving wrist lands across the other wrist ---------
    for s, c_key, o_idx in (("l", "lh", WR + 1), ("r", "rh", WR)):
        m_idx = WR if s == "l" else WR + 1
        base = np.nanmedian(K[:, m_idx, 0] - K[:, o_idx, 0])
        if not np.isfinite(base):
            continue
        for brk, rec in _episodes(C[c_key], fps):
            if not climbing[rec]:
                continue
            crossed = -(np.sign(base)) * (K[rec, m_idx, 0] - K[rec, o_idx, 0]) > 0.1 * T
            if crossed and C[c_key][rec] and (C["lh"][rec] or C["rh"][rec]):
                events.append({"name": "cross_hands", "t": round(rec / fps, 2),
                               "side": SIDE_CN[s] + "手", "conf": 0.8})

    # ---- rock_over: CoM moves over the fresh foothold and rises ---------
    for s, c_key in (("l", "lf"), ("r", "rf")):
        idx = ANK + (0 if s == "l" else 1)
        for brk, rec in _episodes(C[c_key], fps):
            if not climbing[rec] or not np.isfinite(com[rec]).all():
                continue
            ank_x = K[rec, idx, 0]
            for j in range(rec, min(N, rec + int(1.2 * fps))):
                if not (C["lh"][j] or C["rh"][j]):
                    continue
                over = abs(com[j, 0] - ank_x) < 0.3 * T
                rise = (com[rec, 1] - com[j, 1]) / T
                if over and rise >= 0.15:
                    events.append({"name": "rock_over", "t": round(j / fps, 2),
                                   "side": SIDE_CN[s] + "脚", "conf": 0.8})
                    break

    # ---- dyno (动态跳跃): both feet leave while CoM shoots upward with a
    # hand releasing - the lunge signature ----------------------------------
    feet_off = ~(C["lf"] | C["rf"])
    runs, i = [], 0
    while i < N:
        if feet_off[i] and climbing[i]:
            j = i
            while j + 1 < N and feet_off[j + 1]:
                j += 1
            if (j - i + 1) >= int(0.25 * fps):
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    vy = np.gradient(np.convolve(np.nan_to_num(com[:, 1]), np.ones(5) / 5,
                                 mode="same")) * fps  # smoothed dy/dt
    for a, b in runs:
        if not np.isfinite(vy[a:b + 1]).any():
            continue
        burst = np.nanmin(vy[a:b + 1]) / T  # most negative = fastest up
        if burst < -1.0:   # both feet off >=0.25s + fast CoM rise = lunge
            events.append({"name": "dyno", "t": round(a / fps, 2),
                           "side": "", "conf": round(min(1.0, 0.5 - burst / 2), 2)})

    # ---- mantle (翻上): both hands planted low, elbows extend from bent
    # to straight while the body rises over the shelf ----------------------
    sho_mid_y = (K[:, SHO, 1] + K[:, SHO + 1, 1]) / 2
    hands_low = (K[:, WR, 1] > sho_mid_y + 0.15 * T) & \
                (K[:, WR + 1, 1] > sho_mid_y + 0.15 * T)
    ok_el = np.isfinite(A["lel"]) & np.isfinite(A["rel"])
    prep = climbing & C["lh"] & C["rh"] & hands_low & ok_el & \
        (A["lel"] < 115) & (A["rel"] < 115)
    for i in np.where(prep)[0][::3]:
        if any(e["name"] == "mantle" and abs(e["t"] * fps - i) < 1.5 * fps
               for e in events):
            continue
        for j in range(i, min(N, i + int(1.8 * fps))):
            ext = A["lel"][j] > 150 and A["rel"][j] > 150
            if ext and np.isfinite(com[i]).all() and np.isfinite(com[j]).all():
                rise = (com[i, 1] - com[j, 1]) / T
                if rise >= 0.25:
                    events.append({"name": "mantle", "t": round(i / fps, 2),
                                   "side": "", "conf": 0.75})
                    break

    phases.sort(key=lambda p: p["t0"])
    events.sort(key=lambda e: e["t"])
    return {"phases": phases, "events": events}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--from-t", type=float, default=0.0,
                    help="detect only from this time (demo section start)")
    args = ap.parse_args()
    d = json.loads(Path(args.src).read_text(encoding="utf-8"))
    fps = d["stats"]["fps"]
    lo = int(round(args.from_t * fps))
    sub = d["frames"][lo:]
    out = detect(sub, d["stats"])
    # times are relative to the slice -> restore absolute video time
    for p in out["phases"]:
        p["t0"] = round(p["t0"] + args.from_t, 2)
        p["t1"] = round(p["t1"] + args.from_t, 2)
    for e in out["events"]:
        e["t"] = round(e["t"] + args.from_t, 2)
    Path(args.dst).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"phases={len(out['phases'])} events={len(out['events'])} "
          f"(from t={args.from_t}s)")
    for p in out["phases"]:
        print(f"  phase {p['name']:<12} {p['t0']:6.1f}-{p['t1']:6.1f}s "
              f"side={p['side'] or '-'} conf={p['conf']}")
    for e in out["events"]:
        print(f"  event {e['name']:<12} t={e['t']:6.1f}s side={e['side'] or '-'}")


if __name__ == "__main__":
    main()
