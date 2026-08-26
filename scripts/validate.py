"""Stage-output validators: gate the pipeline on qualified intermediate data.

Usage:
  python validate.py pose   <pose.json>      [--min-detect 0.75] [--w W --h H]
  python validate.py analysis <analysis.json> [--min-com 0.85] [--w W --h H]

Exits 0 when all checks pass; prints each failure otherwise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

STATES = {"4pt", "3pt", "2pt", "1pt", "idle"}
REQUIRED_FRAME_KEYS = {"i", "t", "kpts", "angles", "com", "contacts", "n",
                       "state", "st_t0", "bd", "margin", "inside", "climbing"}


def _err(errs, msg):
    errs.append(msg)


def validate_pose(path, min_detect=0.75, w=None, h=None):
    errs = []
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = d.get("frames", [])
    if not frames:
        return ["pose: no frames"]
    n = len(frames)
    detected = 0
    for i, fr in enumerate(frames):
        if fr["i"] != i:
            _err(errs, f"pose: frame index gap at {i} (got {fr['i']})")
        k = fr.get("kpts")
        if k is None:
            continue
        detected += 1
        if len(k) != 17:
            _err(errs, f"pose: frame {i} has {len(k)} keypoints (need 17)")
            continue
        for j, p in enumerate(k):
            if not (isinstance(p, (list, tuple)) and len(p) == 3):
                _err(errs, f"pose: frame {i} kpt {j} malformed: {p!r}")
                continue
            x, y, c = p
            if not (0.0 <= c <= 1.0):
                _err(errs, f"pose: frame {i} kpt {j} conf {c} out of [0,1]")
            if w and not (-5 <= x <= w + 5):
                _err(errs, f"pose: frame {i} kpt {j} x={x} outside frame")
            if h and not (-5 <= y <= h + 5):
                _err(errs, f"pose: frame {i} kpt {j} y={y} outside frame")
    rate = detected / n
    if rate < min_detect:
        _err(errs, f"pose: detection rate {rate:.1%} < {min_detect:.0%}")
    if not errs:
        print(f"pose OK: {n} frames, detection {rate:.1%}")
    return errs


def validate_analysis(path, min_com=0.85, w=None, h=None):
    errs = []
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = d.get("frames", [])
    stats = d.get("stats", {})
    if not frames:
        return ["analysis: no frames"]
    n = len(frames)
    fps = stats.get("fps", 30.0)
    com_ok = 0
    for i, fr in enumerate(frames):
        missing = REQUIRED_FRAME_KEYS - set(fr.keys())
        if missing:
            _err(errs, f"analysis: frame {i} missing keys {sorted(missing)}")
            continue
        if fr["state"] not in STATES:
            _err(errs, f"analysis: frame {i} unknown state {fr['state']!r}")
        n_c = sum(bool(v) for v in fr["contacts"].values())
        if fr["n"] != n_c:
            _err(errs, f"analysis: frame {i} n={fr['n']} != contacts count {n_c}")
        exp_state = ("idle" if (not fr["climbing"] or fr["com"] is None)
                     else {0: "1pt", 1: "1pt", 2: "2pt", 3: "3pt"}.get(n_c, "4pt"))
        if fr["state"] != exp_state:
            _err(errs, f"analysis: frame {i} state {fr['state']} != expected "
                       f"{exp_state} (climbing={fr['climbing']}, n={n_c})")
        if fr["angles"]:
            for nm, v in fr["angles"].items():
                if not (0.0 <= v <= 180.0):
                    _err(errs, f"analysis: frame {i} angle {nm}={v} out of [0,180]")
        c = fr["com"]
        if c is not None:
            com_ok += 1
            if w and not (0 <= c[0] <= w):
                _err(errs, f"analysis: frame {i} com x={c[0]} outside frame")
            if h and not (0 <= c[1] <= h):
                _err(errs, f"analysis: frame {i} com y={c[1]} outside frame")
        m = fr["margin"]
        if m is not None and m < 0:
            _err(errs, f"analysis: frame {i} margin {m} negative")
        bd = fr["bd"]
        if bd is None or bd < 0 or not math.isfinite(bd):
            _err(errs, f"analysis: frame {i} barn-door {bd!r} invalid")
    rate = com_ok / n
    if rate < min_com:
        _err(errs, f"analysis: com detected {rate:.1%} < {min_com:.0%}")
    # stats consistency: state durations sum ~= duration
    ssum = sum(stats.get("state_s", {}).values())
    dur = n / fps
    if ssum and abs(ssum - dur) > 0.5 * max(1.0, dur * 0.01):
        _err(errs, f"analysis: state_s sum {ssum:.1f}s != duration {dur:.1f}s")
    if stats.get("torso_px", 0) <= 0:
        _err(errs, "analysis: torso_px must be positive")
    for e in d.get("events", []):
        for k in ("t0", "t1", "dur", "dx", "dy", "disp"):
            if k not in e:
                _err(errs, f"analysis: event missing field {k}: {e!r}")
        if e.get("t1", 0) <= e.get("t0", 0):
            _err(errs, f"analysis: event t1<=t0: {e!r}")
        if not e.get("moved"):
            _err(errs, f"analysis: event has no moved limbs: {e!r}")
    if not errs:
        print(f"analysis OK: {n} frames, com {rate:.1%}, "
              f"{len(d.get('events', []))} events")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["pose", "analysis"])
    ap.add_argument("path")
    ap.add_argument("--min-detect", type=float, default=0.75)
    ap.add_argument("--min-com", type=float, default=0.85)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    args = ap.parse_args()
    errs = (validate_pose(args.path, args.min_detect, args.w, args.h) if args.kind == "pose"
            else validate_analysis(args.path, args.min_com, args.w, args.h))
    for e in errs:
        print("FAIL:", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
