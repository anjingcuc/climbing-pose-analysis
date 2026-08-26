"""WhisperX transcript post-processing for climbing tutorials.

Applies a climbing-domain homophone dictionary, term normalization and
punctuation cleanup to word-level whisperX output, then emits:
  - corrected SRT (sentence-level)
  - JSON [{start, end, text, terms:[{word, start, end}]}] with highlighted
    domain terms for overlay animation.

Usage:
  python caption_fix.py <whisper_words.json> -o captions.srt --json captions.json
whisper_words.json: whisperX "segments" list with words [{word,start,end}].
"""
import argparse
import json
import re
from pathlib import Path

# homophone / mishearing -> canonical (ordered: longer patterns first)
TERM_FIX = {
    # 中文术语同音纠错
    "摇点": "岩点", "要点岩": "岩点", "岩点子": "岩点",
    "羊角": "仰角", "养脚": "仰角", "阳角": "仰角", "仰交": "仰角",
    "侧申": "侧身", "测身": "侧身", "测拉": "侧拉", "侧啦": "侧拉",
    "刮脚": "挂脚", "勾脚尖": "勾脚尖", "沟脚": "勾脚",
    "中心": "重心",  # climbing context: 重心 almost always intended
    "严闭": "岩壁", "言壁": "岩壁", "岩石壁": "岩壁",
    "岩管": "岩馆", "宝石": "抱石", "宝时": "抱石",
    "美分": "镁粉", "梅粉": "镁粉", "镁粉袋": "镁粉袋",
    "枝点": "支点", "之点": "支点",
    "纸洞": "指洞", "指力条": "指条",
    "交点": "脚点", "脚踝点": "脚点",
    "勺坡": "sloper", "斯洛坡": "sloper", "开放点": "sloper",
    "拼起": "pinch", "品池": "pinch", "捏点": "pinch",
    "克林普": "crimp", "科林普": "crimp", "小条": "crimp",
    "加格": "jug", "大把手": "jug",
    "沃乐": "volume", "窝鲁": "volume", "造型": "volume",
    "地诺": "Dyno", "迪诺": "Dyno", "dinu": "Dyno",
    "希尔胡克": "heel hook", "后跟挂": "heel hook", "脚后跟勾": "heel hook",
    "脚尖勾": "toe hook", "图hook": "toe hook",
    "恐龙膝": "drop knee", "内扣膝": "drop knee", "埃及人": "Egyptian",
    "弗莱格": "flagging", "旗帜": "flagging", "甩腿": "flagging",
    "贝塔": "beta", "bata": "beta",
    "三脚架": "三脚架",
    "吐克力": "tension", "张力": "张力",
    "开脚": "开脚", "外侧重心": "外侧重心",
    "顶哏": "顶胯", "顶跟": "顶胯", "推哏": "推胯",
    "绷直脚": "绷脚", "脚尖发力": "脚尖发力",
    "静音": "静音",
}
# English casing normalization
CASE_FIX = {
    "dyno": "Dyno", "DYNO": "Dyno", "sloper": "sloper", "SLOPER": "sloper",
    "pinch": "pinch", "crimp": "crimp", "CRIMP": "crimp", "jug": "jug",
    "volume": "volume", "VOLUME": "volume", "flagging": "flagging",
    "beta": "beta", "BETA": "beta", "heel hook": "heel hook",
    "toe hook": "toe hook", "drop knee": "drop knee",
    "match": "match", "mantle": "mantle", "smear": "smear",
    "edging": "edging", "heel-hook": "heel hook", "toe-hook": "toe hook",
}
# terms to highlight in captions (display layer)
HIGHLIGHT = {"重心", "三点", "侧身", "侧拉", "仰角", "挂脚", "勾脚", "flagging",
             "heel hook", "toe hook", "Dyno", "drop knee", "开脚", "顶胯",
             "推胯", "绷脚", "重心转移", "支撑", "岩点", "脚点", "手点",
             "sloper", "pinch", "crimp", "volume", "beta",
             "交叉脚", "并手", "摆荡", "交叉手", "换手", "高脚", "换脚"}


BOUNDARY = "的了呢吧吗啊嘛哟哦啦嘛对我说你你我他这那就是很挺不太还就也都又再"
_PUNCT_TAIL = ("，", "。", "！", "？", ",", ".", "!", "?")


def _cut_at_boundary(words):
    """Word-level backtrack: prefer closing a line after a word that ends
    with a natural particle. NEVER splits inside a word token (words are
    the atomic unit of whisperX alignment - slicing characters garbles
    Chinese subtitles)."""
    n = len(words)
    for k in range(1, min(5, n)):
        tok = words[n - k - 1]["word"]
        if tok and tok[-1] in BOUNDARY:
            return n - k
    return n


def _punctuate(chunk, trailing):
    """Join word tokens, inserting a comma at speech pauses."""
    text, last_end = "", None
    for w in chunk:
        if last_end is not None and w["start"] - last_end >= 0.30 and \
                not text.endswith(_PUNCT_TAIL):
            text += "，"
        text += w["word"]
        last_end = w["end"]
    if not text.endswith(_PUNCT_TAIL):
        text += trailing
    return text


def resegment(words, max_chars=18, gap_s=0.6, sentence_gap=0.55):
    """Re-chunk word timestamps into readable, punctuated caption lines.

    Gates (user-confirmed):
    - lines only break on WORD boundaries (a word token is never split
      across two captions, however long it is);
    - inter-word pauses >= 0.30s insert a comma; a line followed by a
      >= sentence_gap pause (or the end of the stream) ends with a full
      stop, otherwise a comma (the sentence continues next line).
    """
    chunks, cur, prev_end = [], [], None
    for w in words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        w = {"word": text, "start": float(w["start"]), "end": float(w["end"])}
        if cur and prev_end is not None and w["start"] - prev_end > gap_s:
            chunks.append(cur)            # hard speech pause -> new chunk
            cur = []
        cur.append(w)
        prev_end = w["end"]
        if sum(len(x["word"]) for x in cur) >= max_chars:
            cut = _cut_at_boundary(cur)
            chunks.append(cur[:cut])
            cur = cur[cut:]
            if cur:
                prev_end = cur[-1]["end"]
    if cur:
        chunks.append(cur)

    lines = []
    for i, chunk in enumerate(chunks):
        nxt_gap = None
        if i + 1 < len(chunks):
            nxt_gap = chunks[i + 1][0]["start"] - chunk[-1]["end"]
        trailing = "。" if (nxt_gap is None or nxt_gap >= sentence_gap) else "，"
        text = _punctuate(chunk, trailing)
        if text.strip("，。"):
            lines.append({"start": chunk[0]["start"], "end": chunk[-1]["end"],
                          "text": text})
    return lines


def load_dict(path):
    """Parse the workspace DICT.md into correction/highlight/prompt layers.

    Sections (## headers, terms separated by middle dots or newlines):
      ## 纠错映射   lines "错词 => 正词"  -> per-project deterministic fixes
      ## 口播高频术语 / ## 术语  terms    -> auto-extend the highlight set
      ## 转录 initial-prompt  paragraph  -> fed to transcribe.py as decoder
                                          vocabulary hint (see transcribe)
    """
    d = {"fixes": {}, "highlight": set(), "prompt": ""}
    section = None
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s.startswith("##"):
            section = s.lstrip("# ").strip()
            continue
        if "=>" in s and section and "纠错" in section:
            k, _, v = s.partition("=>")
            if k.strip() and v.strip():
                d["fixes"][k.strip()] = v.strip()
        elif section and ("术语" in section):
            for t in re.split(r"[·•|、，,\s]+", s):
                if 2 <= len(t) <= 12 and not t.isdigit():
                    d["highlight"].add(t)
        elif section and "initial-prompt" in section and s:
            d["prompt"] += s
    return d


def fix_text(text, extra_fixes=None):
    fixes = dict(TERM_FIX)
    if extra_fixes:
        fixes.update(extra_fixes)
    for k, v in sorted(fixes.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(k, v)
    for k, v in CASE_FIX.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.I)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"^[，。,.]+", "", text)
    return text


def find_terms(text, extra=None):
    """Spans of highlight terms for karaoke-style emphasis."""
    terms = set(HIGHLIGHT) | set(extra or ())
    spans = []
    for term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(re.escape(term), text):
            if not any(a <= m.start() < b for a, b, _ in spans):
                spans.append((m.start(), m.end(), term))
    return sorted(spans)


def fmt_srt_t(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("whisper_json")
    ap.add_argument("-o", "--srt", required=True)
    ap.add_argument("--json", dest="json_out", required=True)
    ap.add_argument("--dict", dest="dict_path", default=None,
                    help="workspace DICT.md: correction mappings, extra "
                         "highlight terms (and the transcribe prompt source)")
    args = ap.parse_args()

    dd = load_dict(args.dict_path) if args.dict_path else None
    extra_fixes = dd["fixes"] if dd else None
    extra_hl = dd["highlight"] if dd else None

    raw = json.loads(Path(args.whisper_json).read_text(encoding="utf-8"))
    segments = raw["segments"] if isinstance(raw, dict) else raw

    caps, srt_lines = [], []
    idx = 0
    for seg in segments:
        words = seg.get("words") or []
        if words:
            # re-chunk long whisper segments into readable caption lines
            for line in resegment(words):
                text = fix_text(line["text"], extra_fixes)
                if not text or text == "。":
                    continue
                spans = find_terms(text, extra_hl)
                caps.append({"start": round(line["start"], 2),
                             "end": round(line["end"], 2), "text": text,
                             "terms": [{"term": t, "off": a} for a, b, t in spans]})
                idx += 1
                srt_lines += [str(idx),
                              f"{fmt_srt_t(line['start'])} --> {fmt_srt_t(line['end'])}",
                              text, ""]
        else:
            text = fix_text(seg.get("text", ""), extra_fixes)
            if not text or text == "。":
                continue
            caps.append({"start": round(seg["start"], 2),
                         "end": round(seg["end"], 2), "text": text, "terms": []})
            idx += 1
            srt_lines += [str(idx),
                          f"{fmt_srt_t(seg['start'])} --> {fmt_srt_t(seg['end'])}",
                          text, ""]

    Path(args.srt).write_text("\n".join(srt_lines), encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(caps, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    n_terms = sum(len(c["terms"]) for c in caps)
    print(f"captions={len(caps)} highlighted_terms={n_terms}")


if __name__ == "__main__":
    main()
