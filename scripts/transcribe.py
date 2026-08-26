"""Transcribe a climbing tutorial video with whisperX (GPU) -> word-level JSON.

Usage: python transcribe.py <video> -o words.json [--model large-v2]
Set HF_ENDPOINT=https://hf-mirror.com for China network before running.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from procutil import run as sub_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="large-v3",
                    help="whisperX model (large-v3 recommended for Chinese)")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--initial-prompt", default=None,
                    help="domain vocabulary hint for the whisper decoder, "
                         "e.g. terms/names from DICT.md - improves recall of "
                         "climbing jargon and people's names")
    ap.add_argument("--dict", dest="dict_path", default=None,
                    help="workspace DICT.md - feeds the decoder prompt "
                         "automatically (## 转录 initial-prompt + 术语 sections)")
    args = ap.parse_args()

    if args.dict_path:
        from caption_fix import load_dict  # shared DICT.md parser
        dd = load_dict(args.dict_path)
        dict_prompt = " ".join(filter(None, [
            dd["prompt"], "、".join(sorted(dd["highlight"]))]))
        args.initial_prompt = " ".join(
            filter(None, [args.initial_prompt, dict_prompt])).strip()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    import whisperx  # deferred: heavy import

    wav = Path(tempfile.gettempdir()) / "climb_tx.wav"
    sub_run(["ffmpeg", "-y", "-v", "error", "-i", str(args.video),
             "-vn", "-ac", "1", "-ar", "16000", str(wav)])

    audio = whisperx.load_audio(str(wav))
    # whisperX's batched pipeline takes decoder hints via asr_options, not
    # transcribe kwargs (initial_prompt seeds the vocabulary, e.g. DICT terms)
    load_kw = {"compute_type": "float16", "language": args.language}
    if args.initial_prompt:
        load_kw["asr_options"] = {"initial_prompt": args.initial_prompt}
    model = whisperx.load_model(args.model, "cuda", **load_kw)
    result = model.transcribe(audio, batch_size=16, language=args.language)
    print(f"segments={len(result['segments'])}")

    # alignment for word-level timestamps
    align_model, meta = whisperx.load_align_model(
        language_code=result.get("language", args.language), device="cuda")
    result = whisperx.align(result["segments"], align_model, meta, audio,
                            "cuda", return_char_alignments=False)

    segs = []
    for s in result["segments"]:
        segs.append({"start": s["start"], "end": s["end"],
                     "text": s.get("text", ""),
                     "words": s.get("words") or []})
    Path(args.out).write_text(json.dumps(
        {"segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_words = sum(len(s["words"]) for s in segs)
    print(f"wrote {args.out}: {len(segs)} segments, {n_words} words")


if __name__ == "__main__":
    main()
