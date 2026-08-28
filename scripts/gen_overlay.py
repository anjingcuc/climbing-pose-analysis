"""Generate the hyperframes overlay composition from analysis.json.

Pure logic lives in build_outputs() (unit-testable, no IO); main() handles
video cutting (ffmpeg, re-encoded for exact frame alignment + rotation bake,
kept at the source's native resolution) and file writing.

Design rules confirmed by the user (do not regress):
- The film starts at frame 0 of the source: the pre-climb preparation plays
  uncut, with no pose overlay and no HUD - analysis layers fade in only once
  climbing starts (pose_t0 = first climbing frame).
- First frame = raw gym footage + a local semi-transparent title panel (no
  full-screen mask, no entrance animation).
- The bottom HUD panel hides whenever it would cover the climber.
- Technique badges (tech_moves.py) follow the climber above the head.
"""
import argparse
import json
import shutil
from pathlib import Path

from procutil import run as sub_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TITLE_DEFAULT = "攀岩动作技术分析"
BASE_W, BASE_H = 1080, 1920
GSAP_CDN = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"


def probe_display_size(video_path):
    """Display-space (rotation-applied) frame size, via OpenCV first frame."""
    import cv2  # deferred: keeps unit tests import-light
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        h, w = frame.shape[:2]
        return int(w), int(h)
    return BASE_W, BASE_H


def probe_video_encoder():
    """Prefer NVENC (h264_nvenc, ~10x faster on 4K segment cuts); fall back
    to libx264 when the GPU/driver lacks it (measured: driver 595.79 has
    nvenc API 13.0 while ffmpeg 9.0 needs 13.1). Returns (codec, flags)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "testsrc2=duration=0.3:size=320x180",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30)
        if r.returncode == 0:
            return "h264_nvenc", ["-preset", "p6", "-rc", "vbr", "-cq", "18",
                                  "-b:v", "0"]
    except Exception:
        pass
    return "libx264", ["-preset", "fast", "-crf", "18"]


def pick_segment(frames, fps, max_s=0.0):
    """[0, end] where end follows the longest climbing run (+1.5s pad).

    The head of the video is NEVER trimmed: the preparation must play in full,
    just without analysis layers. Only the tail after the last climbing frame
    of the longest run is dropped. max_s > 0 caps total duration (trims the
    tail, still keeps frame 0).
    """
    best = None
    i = 0
    N = len(frames)
    while i < N:
        if frames[i]["climbing"]:
            j = i
            gap = 0
            while j + 1 < N:
                if frames[j + 1]["climbing"]:
                    gap = 0
                else:
                    gap += 1
                    if gap > int(2.5 * fps):  # tolerate short dropouts
                        break
                j += 1
            # trim trailing non-climbing frames
            end = j
            while end > i and not frames[end]["climbing"]:
                end -= 1
            if best is None or (end - i) > (best[1] - best[0]):
                best = (i, end)
            i = j + 1
        else:
            i += 1
    if best is None:
        b = min(N - 1, int((max_s or 10.0) * fps) - 1)
        return 0, b
    a, b = 0, min(N - 1, best[1] + int(1.5 * fps))
    if max_s and (b - a + 1) / fps > max_s:
        b = a + int(max_s * fps) - 1
    return a, b


def fmt_events(seg_frames, events, fps, off_i, cap=60, min_cap_dur=0.5):
    """Events inside the segment as [(event, t0_rel, t1_rel)], queued not dropped.

    Climbing is mostly chained 3pt->3pt transfers, so captions queue
    back-to-back: when two events are close, the earlier caption window is
    shortened (never below min_cap_dur + fades) instead of being skipped. A
    clip spans [t0-0.35, t0+cap_dur+1.25] and the fade-out must complete
    inside it, so consecutive starts need >= 1.65s + min_cap_dur apart; closer
    than that and the smaller-displacement one yields. Each returned event
    dict carries the clamped "cap_dur".
    """
    t_off = off_i / fps
    seg_dur = len(seg_frames) / fps
    cand = []
    for e in events:
        t0, t1 = e["t0"] - t_off, e["t1"] - t_off
        if t1 < 4.6 or t0 < 0.5 or t0 > seg_dur - 1.0:
            continue  # hidden by the title panel, or clipped by the segment
        cand.append((e, t0, t1))
    if len(cand) > cap:  # extreme density: keep the largest displacements
        cand.sort(key=lambda x: -x[0]["disp"])
        cand = cand[:cap]
    cand.sort(key=lambda x: x[1])

    # clip i ends at t0_i + cap_dur + 1.25 and must not touch clip i+1
    # (same-track clips may not overlap; fade-out completes inside the clip)
    min_gap = 1.65 + min_cap_dur
    kept = []
    for e, t0, t1 in cand:
        if kept and t0 - kept[-1][1] < min_gap:
            if e["disp"] > kept[-1][0]["disp"]:
                kept.pop()
                if kept and t0 - kept[-1][1] < min_gap:
                    continue  # doesn't fit in the freed slot either
            else:
                continue
        kept.append((e, t0, t1))

    out = []
    for k, (e, t0, t1) in enumerate(kept):
        cap_dur = min(e["dur"], 3.2)
        if k + 1 < len(kept):  # hand off cleanly before the next caption
            cap_dur = min(cap_dur, kept[k + 1][1] - t0 - 1.65)
        out.append((dict(e, cap_dur=round(max(cap_dur, min_cap_dur), 2)), t0, t1))
    out.sort(key=lambda x: x[1])
    return out


def hide_windows(seg_frames, fps, panel_top, dur, panel_x1=None):
    """Time windows where the climber's body enters the bottom-panel zone.

    Occlusion = any keypoint below panel_top and left of panel_x1 (canvas
    px). panel_x1 matters in landscape, where the panel spans only part of
    the width; portrait passes None (full width). Hysteresis: hide after
    0.4s occluded, show again after 0.6s clear; windows padded 0.3s so
    draw()'s 0.35s fade ramps read as motion, not flicker.
    """
    occ = []
    for fr in seg_frames:
        pts = [p for p in (fr.get("kpts") or []) if p]
        occ.append(bool(pts) and any(p[1] > panel_top and
                                     (panel_x1 is None or p[0] <= panel_x1)
                                     for p in pts))
    return _hysteresis_windows(occ, fps, dur, on_s=0.4, off_s=0.6, pad=0.3)


def _seg_rect_hit(p, q, x0, y0, x1, y1):
    """Does segment p-q cross axis-aligned rect (x0,y0,x1,y1)? Cohen-style."""
    # quick reject on segment bbox
    if max(p[0], q[0]) < x0 or min(p[0], q[0]) > x1 or \
       max(p[1], q[1]) < y0 or min(p[1], q[1]) > y1:
        return False
    for a, b in ((p, q),):
        # either endpoint inside?
        if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
            return True
        if x0 <= b[0] <= x1 and y0 <= b[1] <= y1:
            return True
    # segment vs the 4 rect edges
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    def seg2(px, py, qx, qy, rx, ry, sx, sy):
        return ccw(px, py, rx, ry, qx, qy) != ccw(px, py, sx, sy, qx, qy) and \
               ccw(rx, ry, px, py, sx, sy) != ccw(rx, ry, qx, qy, sx, sy)
    for ex0, ey0, ex1, ey1 in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                               (x1, y1, x0, y1), (x0, y1, x0, y0)):
        if seg2(p[0], p[1], q[0], q[1], ex0, ey0, ex1, ey1):
            return True
    return False


# COCO-17 bone pairs used for skeleton-vs-subtitle collision
_SKEL_BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
               (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def cap_zone_windows(seg_frames, fps, dur, scale, vw=1080,
                     top_base=200.0, low_base=1280.0, band_h=160.0,
                     half_w=500.0, on_s=1.0, dwell_s=6.0, pad=0.2):
    """Windows while the subtitle sits in the LOW slot.

    Stability-first v3, driven by TRUE skeleton collision (user-confirmed):
    a bone segment or keypoint of the climber intersecting the subtitle
    rectangle at the CURRENT slot, sustained on_s seconds. Additional
    anti-oscillation guards:
      - never moves into a slot that is ALSO currently occupied;
      - at most one switch per dwell_s seconds (min dwell in each slot);
    so a long climb through both zones produces at most a couple of moves,
    never threshold-hover bouncing. Frames without keypoints carry state.
    """
    def slot_hit(fr, slot_base):
        pts = [p for p in (fr.get("kpts") or []) if p]
        if not pts:
            return None
        y0 = slot_base * scale
        y1 = (slot_base + band_h) * scale
        x0 = vw / 2 - half_w * scale
        x1 = vw / 2 + half_w * scale
        for p in pts:  # any keypoint inside the rect
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return True
        K = fr.get("kpts")
        for a, b in _SKEL_BONES:  # any bone segment crossing the rect
            pa, pb = K[a], K[b]
            if pa and pb and _seg_rect_hit(pa, pb, x0, y0, x1, y1):
                return True
        return False

    on_need = max(1, int(on_s * fps))
    dwell_f = int(dwell_s * fps)
    wins = []
    slot = top_base          # current slot: starts (and ends) at the top
    low = False              # inside a low-slot window?
    cnt = 0
    since_switch = dwell_f   # allow the first switch immediately
    start = 0.0
    for i, fr in enumerate(seg_frames):
        since_switch += 1
        occ = slot_hit(fr, slot)
        if occ is None:
            continue         # detection dropout: keep state and count
        if occ:
            cnt += 1
            if cnt >= on_need and since_switch >= dwell_f:
                other = low_base if slot == top_base else top_base
                if not slot_hit(fr, other):
                    # sustained true occlusion -> move exactly once
                    if slot == top_base:
                        slot, low = low_base, True
                        start = max(0.0, (i - on_need) / fps - pad)
                    else:
                        wins.append([round(start, 2),
                                     round((i - on_need) / fps + pad, 2)])
                        slot, low = top_base, False
                    cnt, since_switch = 0, 0
        else:
            cnt = 0
    if low:
        wins.append([round(start, 2), round(dur, 2)])
    return [[a, b] for a, b in wins if b > a]


def _hysteresis_windows(occ, fps, dur, on_s, off_s, pad):
    """Boolean/None stream -> merged [t0, t1] windows (None = keep state)."""
    on_need, off_need = max(1, int(on_s * fps)), max(1, int(off_s * fps))
    wins, state, cnt, start = [], False, 0, 0.0
    for i, o in enumerate(occ):
        if o is None:
            continue
        if state:
            if not o:
                cnt += 1
                if cnt >= off_need:
                    wins.append((start, (i - off_need) / fps))
                    state, cnt = False, 0
            else:
                cnt = 0
        else:
            if o:
                cnt += 1
                if cnt >= on_need:
                    state, start, cnt = True, (i - on_need) / fps, 0
            else:
                cnt = 0
    if state:
        wins.append((start, dur))
    return [[round(max(0.0, a - pad), 2), round(min(dur, b + pad), 2)]
            for a, b in wins if b > a]


def build_outputs(d, a, b, video_name, title_sub, template_html,
                  vw=BASE_W, vh=BASE_H, tech=None, captions=None):
    """Pure composition builder -> (html, data_js, timings)."""
    frames, events, stats = d["frames"], d["events"], d["stats"]
    fps = stats["fps"]
    dur = (b - a + 1) / fps
    seg_frames = frames[a:b + 1]
    seg_events = fmt_events(seg_frames, events, fps, a)
    # HUD is laid out on the 1080x1920 base grid and scaled to fit BOTH
    # canvas axes: portrait keeps W/1080, landscape (e.g. 3840x2160) scales
    # by height instead so the panels never swallow the frame.
    scale = min(vw / BASE_W, vh / BASE_H)

    # Analysis layers mount 0.8s before the first climbing frame and fade in
    # exactly when climbing starts: before that the footage plays clean, with
    # no skeleton or HUD. Top HUD follows the same gate as the bottom panel.
    pose_t0 = next((i / fps for i, fr in enumerate(seg_frames)
                    if fr.get("climbing")), 4.3)
    pose_pre = max(0.0, pose_t0 - 0.8)
    hud_top_start = max(4.3, pose_pre)
    hud_bot_start = max(4.3, pose_pre)
    sum_t0 = max(4.4, dur - 4.8)
    ov_end = sum_t0 - 0.15
    hud_top_dur = ov_end - hud_top_start
    hud_bot_dur = ov_end - hud_bot_start
    pose_dur = max(0.1, ov_end + 0.2 - pose_pre)

    # technique badges, re-based to segment time
    t_off = a / fps
    seg_tech = {"phases": [], "events": []}
    if tech:
        seg_tech = {
            "phases": [{**p, "t0": round(p["t0"] - t_off, 2),
                        "t1": round(p["t1"] - t_off, 2)}
                       for p in tech.get("phases", [])
                       if p["t1"] > t_off and p["t0"] < t_off + dur],
            "events": [{**e, "t": round(e["t"] - t_off, 2)}
                       for e in tech.get("events", [])
                       if t_off <= e["t"] <= t_off + dur],
        }

    # caps = collision zones for captions; clamped pairwise so chained
    # captions hand off without overlapping zones
    caps = []
    for k, (e, t0, t1) in enumerate(seg_events):
        c0, c1 = t0 - 0.35, t0 + e["cap_dur"] + 1.3
        if k + 1 < len(seg_events):
            c1 = min(c1, seg_events[k + 1][1] - 0.36)
        caps.append([round(c0, 2), round(max(c1, c0 + 0.5), 2)])

    data = {
        "fps": fps, "n": len(seg_frames), "vstart": round(t_off, 3),
        "pose_t0": round(pose_t0, 2),
        "stats": stats, "frames": seg_frames,
        "tech": seg_tech,
        "caps": caps,
        "hud_hide": hide_windows(seg_frames, fps, vh - 348 * scale, dur,
                                 panel_x1=1080 * scale),
        # subtitles switch slots only when the band truly covers the
        # climber at the current slot (toggle, heavy hysteresis) - the
        # position otherwise stays put, no threshold-hover bouncing
        "cap_low": cap_zone_windows(seg_frames, fps, dur, scale, vw=vw),
        "ov_end": round(ov_end, 2),
    }

    clips, tweens, ev_kills = [], [], []
    for k, (e, t0, t1) in enumerate(seg_events):
        moved = "、".join(e["moved"][:2]) + ("等" if len(e["moved"]) > 2 else "") if e["moved"] else "肢体"
        label = ("三点平衡转移" if e.get("s0") == "3pt" and e.get("s1") == "3pt"
                 else "重心转移")
        cap_dur = e["cap_dur"]  # clamped so chained captions queue, not skip
        dirx = "左移" if e["dx"] < -0.15 else ("右移" if e["dx"] > 0.15 else "")
        diry = "上升" if e["dy"] < -0.15 else ("下降" if e["dy"] > 0.15 else "")
        dirs = "·".join(x for x in (dirx, diry) if x) or "重心调整"
        clips.append(
            f'<div id="ev{k}" class="clip evcap" data-start="{max(0, t0-0.35):.2f}" '
            f'data-duration="{cap_dur+1.6:.2f}" data-track-index="7">'
            f'<div class="ev-scale"><div id="evi{k}" class="ev-inner"><div class="ev-mark"></div>'
            f'<div class="ev-txt">{label} · {moved}移动</div>'
            f'<div class="ev-sub">{dirs} {e["disp"]:.2f} 身位 · {e["dur"]:.1f}s</div>'
            f"</div></div></div>")
        tweens.append(
            f'tl.fromTo("#evi{k}", {{opacity:0, y:26, scale:0.94}}, '
            f'{{opacity:1, y:0, scale:1, duration:0.45, ease:"back.out(1.6)"}}, {t0:.2f});')
        tweens.append(
            f'tl.to("#evi{k}", {{opacity:0, y:-12, duration:0.35, ease:"power2.in"}}, {t0+cap_dur+0.7:.2f});')
        # hard kill after the fade: non-linear seeks must not leave stale
        # state (emitted after clip_starts is known - see below for the
        # boundary-snap pass)
        ev_kills.append((k, t0 + cap_dur + 1.15))

    s3 = stats["state_s"].get("3pt", 0)
    n_transfers = stats.get("transfers", len(seg_events))
    n_tech = len(seg_tech["phases"]) + len(seg_tech["events"])
    countups = (
        f'tl.to("#sv-3pt", {{innerText: {s3:.0f}, snap:{{innerText:1}}, duration: 1.1, ease:"power1.out"}}, {sum_t0+0.7:.2f});\n'
        f'      tl.to("#sv-ev", {{innerText: {n_transfers}, snap:{{innerText:1}}, duration: 1.0, ease:"power1.out"}}, {sum_t0+0.8:.2f});\n'
        f'      tl.to("#sv-tech", {{innerText: {n_tech}, snap:{{innerText:1}}, duration: 1.0, ease:"power1.out"}}, {sum_t0+0.9:.2f});\n'
        f'      tl.to("#sv-bd", {{innerText: {stats["max_barndoor"]:.0f}, snap:{{innerText:1}}, duration: 1.2, ease:"power1.out"}}, {sum_t0+1.0:.2f});')

    # speech captions: strictly sequential windows (one on screen at a time),
    # rebased to segment time, ending before the summary card. They live in
    # the top zone (below the top HUD); while the title panel is up they sit
    # below it and slide up after it fades (template handles positioning).
    cap_clips, cap_tweens, cap_starts, cap_ends = [], [], [], []
    if captions:
        from gen_tutorial import caption_html  # lazy: avoid import cycle
        prev_end = 0.0
        for k, c in enumerate(captions):
            cs = max(c["start"] - t_off - 0.15, prev_end + 0.05)
            ce = max(c["end"] - t_off, cs + 0.4)
            if ce > ov_end - 0.3:  # never draw over the summary card
                ce = min(ce, ov_end - 0.3)
                if ce - cs < 0.4:
                    continue
            if cs > dur - 0.2:
                continue
            win = ce - cs + 0.45   # fade-out completes strictly inside the clip
            prev_end = cs + win + 0.05
            cap_starts.append(round(cs, 2))
            cap_ends.append(round(ce, 2))
            cap_clips.append(
                f'<div id="cap{k}" class="clip cap" data-start="{cs:.2f}" '
                f'data-duration="{win:.2f}" data-track-index="10">'
                f'<div class="cap-scale"><div id="capi{k}" class="cap-txt">'
                f'{caption_html(c["text"], c.get("terms", []))}'
                f"</div></div></div>")
            cap_tweens.append(
                f'tl.fromTo("#capi{k}", {{opacity:0, y:14}}, '
                f'{{opacity:1, y:0, duration:0.3, ease:"power3.out"}}, {cs+0.05:.2f});')
            cap_tweens.append(
                f'tl.to("#capi{k}", {{opacity:0, duration:0.25, ease:"power2.in"}}, {ce+0.05:.2f});')
            cap_tweens.append(
                f'tl.set("#capi{k}", {{opacity:0}}, {ce+0.40:.2f});')

    # linter rule gsap_exit_missing_hard_kill: when an exit sequence (fade
    # AND its hard-kill set) crosses a later clip's start boundary (usually
    # a speech-caption clip), a tl.set must also land exactly on that
    # boundary so non-linear seeks never leave stale visibility past it
    clip_starts = sorted(set(round(max(0.0, t0 - 0.35), 2) for _, t0, _ in seg_events)
                         | set(cap_starts) | {round(sum_t0, 2)})
    for k, kill_t in ev_kills:
        # a hard kill landing EXACTLY on another clip's start boundary trips
        # gsap_exit_missing_hard_kill: snap it just before the boundary
        if any(abs(kill_t - cs) < 0.05 for cs in clip_starts):
            kill_t -= 0.06
        tweens.append(
            f'tl.set("#evi{k}", {{opacity:0}}, {kill_t:.2f});')
    for k, (e, t0, t1) in enumerate(seg_events):
        kill_end = t0 + e["cap_dur"] + 1.15
        crossed = [s for s in clip_starts if t0 + 0.4 < s < kill_end]
        for cb in crossed:   # EVERY crossed boundary needs its own kill
            tweens.append(
                f'tl.set("#evi{k}", {{opacity:0}}, {cb:.2f});')
    for k, (cs_val, ce_val) in enumerate(zip(cap_starts, cap_ends)):
        kill_end = ce_val + 0.40
        crossed = [s for s in clip_starts if cs_val + 0.4 < s < kill_end]
        for cb in crossed:
            cap_tweens.append(
                f'tl.set("#capi{k}", {{opacity:0}}, {cb:.2f});')

    html = (template_html
            .replace("{{DURATION}}", f"{dur:.2f}")
            .replace("{{VIDEO}}", video_name)
            .replace("{{TITLE_SUB}}", title_sub)
            .replace("{{VW}}", str(vw)).replace("{{VH}}", str(vh))
            .replace("{{SCALE}}", f"{scale:g}")
            .replace("{{EV_BOT}}", str(round(348 * scale)))
            .replace("{{POSE_PRE}}", f"{pose_pre:.2f}")
            .replace("{{POSE_DUR}}", f"{pose_dur:.2f}")
            .replace("{{EVENT_CLIPS}}", "\n      ".join(clips))
            .replace("{{EVENT_TWEENS}}", "\n      ".join(tweens))
            .replace("{{CAPTION_CLIPS}}", "\n      ".join(cap_clips))
            .replace("{{CAPTION_TWEENS}}", "\n      ".join(cap_tweens))
            .replace("{{SUM_T0}}", f"{sum_t0:.2f}")
            .replace("{{SUM_DUR}}", f"{dur - sum_t0:.2f}")
            .replace("{{HUD_TOP_START}}", f"{hud_top_start:.2f}")
            .replace("{{HUD_TOP_DUR}}", f"{hud_top_dur:.2f}")
            .replace("{{HUD_BOT_START}}", f"{hud_bot_start:.2f}")
            .replace("{{HUD_BOT_DUR}}", f"{hud_bot_dur:.2f}")
            .replace("{{SUM_COUNTUPS}}", countups))
    timings = {
        "dur": dur, "a": a, "b": b, "sum_t0": sum_t0, "ov_end": ov_end,
        "pose_t0": pose_t0, "pose_pre": pose_pre,
        "hud_top_start": hud_top_start, "hud_bot_start": hud_bot_start,
        "hud_top_dur": hud_top_dur, "hud_bot_dur": hud_bot_dur,
        "n_events": len(seg_events), "n_tech": n_tech,
    }
    return html, data, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--video", required=True, help="source video of the analysis")
    ap.add_argument("--out-dir", default=str(ROOT / "overlay"))
    ap.add_argument("--seg", default=None, help="start_frame:end_frame override")
    ap.add_argument("--max-seg-s", type=float, default=0.0,
                    help="duration cap in seconds (0 = no cap); frame 0 is "
                         "always kept so the preparation plays uncut")
    ap.add_argument("--tech", default=None,
                    help="tech_moves.py json; badges follow the climber")
    ap.add_argument("--captions", default=None,
                    help="caption_fix.py json; speech subtitles, top zone")
    ap.add_argument("--title", default=None, help="title panel subtitle")
    args = ap.parse_args()

    d = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    frames, events, stats = d["frames"], d["events"], d["stats"]
    fps = stats["fps"]

    if args.seg:
        a, b = (int(x) for x in args.seg.split(":"))
    else:
        a, b = pick_segment(frames, fps, args.max_seg_s)
    dur = (b - a + 1) / fps
    print(f"segment frames {a}:{b}  t {a/fps:.1f}s..{b/fps:.1f}s  dur {dur:.1f}s"
          f"  (head kept: preparation plays uncut)")

    vw, vh = probe_display_size(args.video)
    print(f"canvas {vw}x{vh} (native source resolution, no rescale)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cut segment video (re-encode for exact start + rotation baked + audio
    # kept; NO scale filter - the segment keeps the source resolution).
    # Dense keyframes (-g 30): hyperframes seeks the DOM video per captured
    # frame; sparse GOPs (x264 default 250f) stall 4K captures indefinitely.
    seg_mp4 = out_dir / "segment.mp4"
    vcodec, vflags = probe_video_encoder()
    print(f"segment encode: {vcodec}")
    sub_run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{a/fps:.3f}", "-i", str(args.video),
        "-t", f"{dur:.3f}", "-c:v", vcodec, *vflags,
        "-g", "30", "-keyint_min", "30",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", str(seg_mp4)])

    # gsap is bundled locally when available (offline / proxied networks)
    gsap_src = ROOT / "workspace" / "gsap.min.js"
    if gsap_src.exists():
        shutil.copyfile(gsap_src, out_dir / "gsap.min.js")

    tech = (json.loads(Path(args.tech).read_text(encoding="utf-8"))
            if args.tech else None)
    captions = (json.loads(Path(args.captions).read_text(encoding="utf-8"))
                if args.captions else None)
    title_sub = args.title or f"动作技术分析 · 三点平衡 / 重心转移 / 支撑稳定"
    template_html = (HERE / "overlay_template.html").read_text(encoding="utf-8")
    html, data, timings = build_outputs(d, a, b, "segment.mp4", title_sub,
                                        template_html, vw=vw, vh=vh,
                                        tech=tech, captions=captions)
    if not gsap_src.exists():
        html = html.replace('src="gsap.min.js"', f'src="{GSAP_CDN}"')

    (out_dir / "data.js").write_text(
        "window.__CLIMB_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";",
        encoding="utf-8")
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {out_dir}/index.html, data.js, segment.mp4; "
          f"events={timings['n_events']} tech={timings['n_tech']} "
          f"pose_t0={timings['pose_t0']:.2f}s hud_hide={len(data['hud_hide'])}win")


if __name__ == "__main__":
    main()
