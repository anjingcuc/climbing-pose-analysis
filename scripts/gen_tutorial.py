"""Generate the hyperframes tutorial composition: captions + pose overlay.

Inputs: analysis.json (full video), captions.json (from caption_fix),
video. Pose overlay starts at --pose-start seconds (fade-in), captions run
over the whole video. No joint-angle layer.
"""
import argparse
import html as H
import json
from pathlib import Path

from procutil import run as sub_run

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_overlay import fmt_events

HERE = Path(__file__).resolve().parent


def caption_html(text, terms):
    """Escape caption text and wrap highlight terms in spans."""
    spans = sorted((t["off"], t["off"] + len(t["term"])) for t in terms)
    out, pos = [], 0
    for a, b in spans:
        out.append(H.escape(text[pos:a]))
        out.append(f'<span class="term">{H.escape(text[a:b])}</span>')
        pos = b
    out.append(H.escape(text[pos:]))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("captions")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pose-start", type=float, default=39.0)
    ap.add_argument("--title-main", default="仰角 V2 · 侧身技术讲解")
    ap.add_argument("--title-sub", default="姿态追踪 + 重心 / 支撑分析 · 讲解字幕")
    ap.add_argument("--tech", default=None, help="techniques.json from tech_moves")
    args = ap.parse_args()

    d = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    caps = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    tech = (json.loads(Path(args.tech).read_text(encoding="utf-8"))
            if args.tech else {"phases": [], "events": []})
    frames, events, stats = d["frames"], d["events"], d["stats"]
    fps = stats["fps"]
    N = len(frames)
    dur = N / fps

    pose_i = int(round(args.pose_start * fps))
    pose_i = min(max(0, pose_i), N - 1)
    pose_t0 = pose_i / fps
    pose_frames = frames[pose_i:]

    # summary stats restricted to the pose window
    s3 = sum(1 for f in pose_frames if f["state"] == "3pt") / fps
    seg_events = fmt_events(pose_frames, events, fps, pose_i)
    bd = max((f["bd"] or 0) for f in pose_frames)

    sum_t0 = max(4.4, dur - 4.8)
    ov_start = pose_t0 - 0.8
    ov_dur = sum_t0 - 0.15 - ov_start

    # ---- cut segment (whole video, re-encode, rotation baked, audio kept) ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(HERE.parent / "workspace" / "gsap.min.js", out / "gsap.min.js")
    seg_mp4 = out / "segment.mp4"
    sub_run([
        "ffmpeg", "-y", "-v", "error", "-i", str(args.video),
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", str(seg_mp4)])

    data = {
        "fps": fps, "n": len(pose_frames), "pose_t0": round(pose_t0, 2),
        "stats": stats, "frames": pose_frames, "tech": tech,
        "caps": [[t0 - 0.35, t0 + min(e["dur"], 3.2) + 1.3]
                 for e, t0, t1 in seg_events],
    }
    # caps stay segment-relative like frames[] (draw() works in pose time)
    (out / "data.js").write_text(
        "window.__CLIMB_DATA = " + json.dumps(data, ensure_ascii=False,
                                              separators=(",", ":")) + ";",
        encoding="utf-8")

    # ---- caption clips (sequential: window holds text + fade-out inside) ----
    cap_clips, cap_tweens = [], []
    prev_end = 0.0
    for i, c in enumerate(caps):
        cs = max(c["start"] - 0.15, prev_end + 0.05)
        ce = max(c["end"], cs + 0.4)
        win = ce - cs + 0.45          # fade-out completes strictly inside
        prev_end = cs + win + 0.05
        cap_clips.append(
            f'<div id="cap{i}" class="clip cap" data-start="{cs:.2f}" '
            f'data-duration="{win:.2f}" data-track-index="4">'
            f'<div id="capi{i}" class="cap-txt">{caption_html(c["text"], c["terms"])}</div>'
            f"</div>")
        cap_tweens.append(
            f'tl.fromTo("#capi{i}", {{opacity:0, y:18}}, '
            f'{{opacity:1, y:0, duration:0.35, ease:"power3.out"}}, {cs+0.05:.2f});')
        cap_tweens.append(
            f'tl.to("#capi{i}", {{opacity:0, duration:0.25, ease:"power2.in"}}, {ce+0.05:.2f});')

    # ---- transfer badges (pose window only; fmt_events returns segment-
    # relative times, so re-offset to absolute composition time = +pose_t0)
    ev_clips, ev_tweens = [], []
    for k, (e, t0, t1) in enumerate(seg_events):
        t0, t1 = t0 + pose_t0, t1 + pose_t0
        moved = "、".join(e["moved"][:2]) + ("等" if len(e["moved"]) > 2 else "") if e["moved"] else "肢体"
        cap_dur = min(e["dur"], 3.2)
        dirx = "左移" if e["dx"] < -0.15 else ("右移" if e["dx"] > 0.15 else "")
        diry = "上升" if e["dy"] < -0.15 else ("下降" if e["dy"] > 0.15 else "")
        dirs = "·".join(x for x in (dirx, diry) if x) or "重心调整"
        ev_clips.append(
            f'<div id="ev{k}" class="clip evcap" data-start="{max(0, t0-0.35):.2f}" '
            f'data-duration="{cap_dur+1.6:.2f}" data-track-index="7">'
            f'<div id="evi{k}" class="ev-inner"><div class="ev-mark"></div>'
            f'<div class="ev-txt">重心转移 · {moved}移动</div>'
            f'<div class="ev-sub">{dirs} {e["disp"]:.2f} 身位 · {e["dur"]:.1f}s</div>'
            f"</div></div>")
        ev_tweens.append(
            f'tl.fromTo("#evi{k}", {{opacity:0, y:-16, scale:0.94}}, '
            f'{{opacity:1, y:0, scale:1, duration:0.45, ease:"back.out(1.6)"}}, {t0:.2f});')
        ev_tweens.append(
            f'tl.to("#evi{k}", {{opacity:0, y:-10, duration:0.35, ease:"power2.in"}}, {t0+cap_dur+0.7:.2f});')

    # technique summary: counts + variety (more informative than raw 3pt
    # stats for a tutorial whose demo is mostly 1pt/2pt movement)
    from collections import Counter
    TECH_CN = {"side_on": "侧身", "heel_hook": "挂脚", "flagging": "旗式",
               "cross_feet": "交叉脚", "match_hands": "并手",
               "straight_arm": "直臂休息", "high_step": "高脚",
               "foot_swap": "换脚", "cross_hands": "交叉手",
               "rock_over": "压重心"}
    cnt = Counter()
    for ph in tech["phases"]:
        cnt[TECH_CN.get(ph["name"], ph["name"])] += 1
    for ev in tech["events"]:
        cnt[TECH_CN.get(ev["name"], ev["name"])] += 1
    n_tech = sum(cnt.values())
    countups = (
        f'tl.to("#sv-3pt", {{innerText: {n_tech}, snap:{{innerText:1}}, duration: 1.1, ease:"power1.out"}}, {sum_t0+0.7:.2f});\n'
        f'      tl.to("#sv-ev", {{innerText: {len(cnt)}, snap:{{innerText:1}}, duration: 1.0, ease:"power1.out"}}, {sum_t0+0.8:.2f});\n'
        f'      tl.to("#sv-bd", {{innerText: {len(seg_events)}, snap:{{innerText:1}}, duration: 1.2, ease:"power1.out"}}, {sum_t0+0.9:.2f});')

    tpl = (HERE / "tutorial_template.html").read_text(encoding="utf-8")
    html = (tpl
            .replace("{{DURATION}}", f"{dur:.2f}")
            .replace("{{AUDIO_DUR}}", f"{dur - 0.3:.2f}")
            .replace("{{VIDEO}}", "segment.mp4")
            .replace("{{TITLE_MAIN}}", args.title_main)
            .replace("{{TITLE_SUB}}", args.title_sub)
            .replace("{{POSE_T0_PRE}}", f"{ov_start:.2f}")
            .replace("{{POSE_DUR}}", f"{ov_dur:.2f}")
            .replace("{{SUM_T0}}", f"{sum_t0:.2f}")
            .replace("{{SUM_DUR}}", f"{dur - sum_t0:.2f}")
            .replace("{{CAPTION_CLIPS}}", "\n      ".join(cap_clips))
            .replace("{{EVENT_CLIPS}}", "\n      ".join(ev_clips))
            .replace("{{CAPTION_TWEENS}}", "\n      ".join(cap_tweens))
            .replace("{{EVENT_TWEENS}}", "\n      ".join(ev_tweens))
            .replace("{{SUM_COUNTUPS}}", countups))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {out}/index.html; captions={len(caps)} badges={len(seg_events)} "
          f"pose_t0={pose_t0}s dur={dur:.1f}s")


if __name__ == "__main__":
    main()
