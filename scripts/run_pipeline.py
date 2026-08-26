"""One-shot driver with quality gates:
pose extraction -> validate -> biomechanics -> validate -> overlay project.

Usage:
  python run_pipeline.py <video> [--seg a:b] [--model yolo11x-pose.pt]
                                 [--title "..."] [--max-seg-s 75] [--imgsz 1280]
"""
import argparse
import sys
from pathlib import Path

from procutil import run as sub_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    sub_run([str(c) for c in cmd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11x-pose.pt")
    ap.add_argument("--seg", default=None)
    ap.add_argument("--max-seg-s", type=float, default=0.0,
                    help="duration cap in seconds (0 = no cap); frame 0 is "
                         "always kept so the preparation plays uncut")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="pose inference size (raise for >1080p sources, e.g. 1536)")
    ap.add_argument("--on-t", type=float, default=0.09,
                    help="contact-on threshold; raise (~1.4x still-limb noise "
                         "floor) when the climber is small in frame")
    ap.add_argument("--tech", default=None, help="tech_moves json for badges")
    ap.add_argument("--captions", default=None,
                    help="caption_fix json for speech subtitles")
    ap.add_argument("--title", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--pose", default=None, help="reuse existing pose json")
    ap.add_argument("--work-dir", default=None, help="dir for pose/analysis json")
    ap.add_argument("--out-dir", default=None, help="hyperframes project dir")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    here = Path(__file__).resolve().parent
    root = here.parent
    work = Path(args.work_dir) if args.work_dir else root / "workspace"
    out_dir = Path(args.out_dir) if args.out_dir else root / "overlay"
    work.mkdir(parents=True, exist_ok=True)
    pose_json = Path(args.pose) if args.pose else work / f"pose_{args.tag}.json"
    ana_json = work / f"analysis_{args.tag}.json"

    from gen_overlay import probe_display_size
    vw, vh = probe_display_size(video)
    print(f"video display size {vw}x{vh} (canvas follows source resolution)")

    if not args.pose:
        extract = [PY, here / "extract_pose.py", video, "-o", pose_json,
                   "--model", args.model]
        if args.imgsz:
            extract += ["--imgsz", args.imgsz]
        run(extract)
    run([PY, here / "validate.py", "pose", pose_json, "--min-detect", "0.75",
         "--w", vw, "--h", vh])
    run([PY, here / "biomech.py", pose_json, ana_json, "--on-t", args.on_t])
    run([PY, here / "validate.py", "analysis", ana_json, "--min-com", "0.85",
         "--w", vw, "--h", vh])
    gen_cmd = [PY, here / "gen_overlay.py", ana_json, "--video", video,
               "--out-dir", out_dir, "--max-seg-s", args.max_seg_s]
    if args.seg:
        gen_cmd += ["--seg", args.seg]
    if args.tech:
        gen_cmd += ["--tech", args.tech]
    if args.captions:
        gen_cmd += ["--captions", args.captions]
    if args.title:
        gen_cmd += ["--title", args.title]
    run(gen_cmd)
    print("pipeline done ->", out_dir / "index.html")


if __name__ == "__main__":
    main()
