"""Transcribe a climbing video -> word-level JSON (caption pipeline S1).

Engine v2 (2026-08, adapted from the course-subtitles pipeline):
- DEFAULT funasr: paraformer-zh + fsmn-vad + ct-punc. Chinese CER beats
  whisper, ~100x faster, ships REAL punctuation, and takes hotwords (from
  DICT.md) as a hard bias - much stronger than whisper's initial_prompt.
  Its weak spot: word timestamps drift ~-3s/min on continuous speech
  (measured -17s @ 285s), so a whisper pass runs as the TIMING REFERENCE
  and funasr times are warped onto it via text anchors (text from funasr,
  timing from whisper - both engines do what they are best at).
- --engine whisper: pure whisperX path (large-v3 + align), the old default.

Usage: python transcribe.py <video> -o words.json [--dict DICT.md]
                              [--engine funasr|whisper] [--model large-v3]
Set HF_ENDPOINT=https://hf-mirror.com for China network before running.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

from procutil import run as sub_run


def extract_wav(video, out_wav=None):
    wav = Path(out_wav or (Path(tempfile.gettempdir()) / "climb_tx.wav"))
    sub_run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    return wav


def whisper_words(wav, language, model_name, initial_prompt=None):
    """whisperX large-v3 + word alignment (trusted timeline)."""
    import whisperx  # deferred: heavy import

    load_kw = {"compute_type": "float16", "language": language}
    if initial_prompt:
        # whisperX takes decoder hints via asr_options, not transcribe kwargs
        load_kw["asr_options"] = {"initial_prompt": initial_prompt}
    model = whisperx.load_model(model_name, "cuda", **load_kw)
    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=16, language=language)
    align_model, meta = whisperx.load_align_model(
        language_code=result.get("language", language), device="cuda")
    result = whisperx.align(result["segments"], align_model, meta, audio,
                            "cuda", return_char_alignments=False)
    words = []
    for s in result["segments"]:
        for w in s.get("words") or []:
            t = str(w.get("word", "")).strip()
            if t:
                words.append({"word": t, "start": w["start"], "end": w["end"]})
    return words


def _merge_micro_sentences(sents):
    """funasr sentence_info often splits fast speech into 4-6 char micro
    fragments; ct-punc then punctuates every fragment tail, planting
    MID-WORD commas/periods (这应。|该是, 只，|是 - measured on a real
    video). Merge a fragment into its neighbour (core <= 5 chars or
    boundary gap < 0.2s), stripping the fragment's trailing punctuation:
    the merged sentence keeps only ct-punc commas that are real. Unit/ts
    counts are additive so the zip invariant survives the merge."""
    import re
    unit_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#'/-]*|[\u4e00-\u9fff]")
    merged = []
    for s in sents:
        text = s.get("text", "")
        ts = s.get("timestamp") or []
        if merged and (len(unit_re.findall(merged[-1]["text"])) <= 5 or
                       len(unit_re.findall(text)) <= 5 or
                       (ts and merged[-1]["timestamp"] and
                        ts[0][0] - merged[-1]["timestamp"][-1][1] < 200)):
            prev = merged[-1]
            # punctuation-retention (course-subtitles contract): a merge
            # means the sentence continues, so commas STAY (deleting them
            # stripped real clause punctuation from every merged boundary -
            # the bug behind "subtitles lost their punctuation"); sentence
            # tails downgrade to a comma; mid-word artifact tails are
            # removed later by strip_cross_boundary_punct at caption time
            pt = prev["text"].rstrip()
            if pt and pt[-1] in "。！？；;":
                pt = pt[:-1] + "，"
            prev["text"] = pt + text
            prev["timestamp"] = prev["timestamp"] + ts
        else:
            merged.append({"text": text, "timestamp": list(ts)})
    return merged


def funasr_words(wav, hotword=""):
    """paraformer-zh + fsmn-vad + ct-punc -> words with REAL punctuation.

    Mapping rules (course-subtitles proven): latin/digit runs are one
    timestamped unit, CJK chars are units; count mismatch falls back to
    proportional spread inside the sentence span; micro-sentence fragments
    merge first (see _merge_micro_sentences); units regroup into jieba
    words with punctuation tails kept on the word.
    """
    from funasr import AutoModel
    from asr_align import group_units_to_words, sentence_units

    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                      punc_model="ct-punc", disable_update=True)
    kwargs = {"sentence_timestamp": True}
    if hotword:
        kwargs["hotword"] = hotword
    res = model.generate(input=str(wav), batch_size_s=60, **kwargs)
    words = []
    for r in res:
        for sent in _merge_micro_sentences(r.get("sentence_info") or []):
            units = sentence_units(sent["text"], sent["timestamp"])
            if units:
                words.extend(group_units_to_words(units, sent["text"]))
    words.sort(key=lambda w: (w["start"], w["end"]))
    cjk = [w for w in words if w["word"] and not w["word"][0].isascii()]
    if cjk:
        single = sum(1 for w in cjk if len(w["word"].rstrip("，。！？、；：")) == 1)
        if single / len(cjk) > 0.40:
            print("WARNING: %.0f%% single-char CJK words - grouping broken?"
                  % (100 * single / len(cjk)))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--engine", choices=["funasr", "whisper"], default="funasr")
    ap.add_argument("--model", default="large-v3",
                    help="whisperX model (timing reference / whisper engine)")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--initial-prompt", default=None)
    ap.add_argument("--dict", dest="dict_path", default=None,
                    help="workspace DICT.md - hotwords for funasr, decoder "
                         "prompt for whisper (see caption_fix.load_dict)")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    hotword = prompt = ""
    if args.dict_path:
        from caption_fix import load_dict
        dd = load_dict(args.dict_path)
        import re
        hotword = " ".join(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+",
                                      "、".join(sorted(dd["highlight"]))))[:400]
        prompt = dd["prompt"] or None
    args.initial_prompt = args.initial_prompt or prompt

    wav = extract_wav(args.video)

    if args.engine == "whisper":
        words = whisper_words(wav, args.language, args.model,
                              args.initial_prompt)
        segments = _to_segments(words)
    else:
        fwords = funasr_words(wav, hotword)
        if not fwords:
            raise SystemExit("funasr produced no words - check the audio track")
        # timing reference: whisper on the same wav; warp funasr times onto it
        wwords = whisper_words(wav, args.language, args.model,
                               args.initial_prompt)
        # cache both raw streams: warp tuning never needs the GPUs again
        Path(args.out + ".raw.json").write_text(json.dumps(
            {"funasr": fwords, "whisper": wwords}, ensure_ascii=False),
            encoding="utf-8")
        from asr_align import anchor_pairs, warp_times
        if wwords:
            pairs, s_text, r_text = anchor_pairs(fwords, wwords)
            anchors = len([p for p in pairs if p != (0.0, 0.0)]) - 1
            drift = (wwords[-1]["end"] - fwords[-1]["end"]) if fwords else 0
            print(f"anchors={anchors} end-drift={drift:+.1f}s "
                  f"(funasr text, whisper timing)")
            if anchors < 3:
                print("WARNING: <3 text anchors - warping unreliable, "
                      "check engines' languages match")
            fwords = warp_times(fwords, pairs)
        segments = _to_segments(fwords, split_on_punct=True)

    Path(args.out).write_text(json.dumps(
        {"segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(s["words"]) for s in segments)
    print(f"wrote {args.out}: {len(segments)} segments, {n} words "
          f"(engine={args.engine})")


def _to_segments(words, split_on_punct=False):
    """Group the flat word list into segments. funasr words carry real
    punctuation: sentence-final tails (。！？) end a segment; whisper words
    (no punctuation) group by <=0.9s gaps."""
    segs, cur = [], []
    prev_end = None

    def flush():
        if cur:
            segs.append({"start": round(cur[0]["start"], 3),
                         "end": round(cur[-1]["end"], 3),
                         "text": "".join(w["word"] for w in cur),
                         "words": [{"word": w["word"],
                                    "start": round(w["start"], 3),
                                    "end": round(w["end"], 3)} for w in cur]})
            cur.clear()

    for w in words:
        if split_on_punct:
            if prev_end is not None and w["start"] - prev_end > 0.9:
                flush()
            cur.append(w)
            if any(c in w["word"] for c in "。！？!?"):
                flush()
        else:
            if prev_end is not None and w["start"] - prev_end > 0.9 and cur:
                flush()
            cur.append(w)
        prev_end = w["end"]
    flush()
    return segs


if __name__ == "__main__":
    main()
