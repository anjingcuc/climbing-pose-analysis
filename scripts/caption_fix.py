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
_SENT_TAIL = "。！？!?；;"
_COMMA_TAIL = "，、,;"
_ALL_TAIL = _PUNCT_TAIL + ("、", ";", "；")


def _cut_at_boundary(words):
    """Word-level backtrack: prefer closing a line after a word that ends
    with a natural particle. NEVER splits inside a word token (words are
    the atomic unit of ASR alignment - slicing characters garbles Chinese)."""
    n = len(words)
    for k in range(1, min(5, n)):
        tok = words[n - k - 1]["word"]
        if tok and tok[-1] in BOUNDARY:
            return n - k
    return n


def disp_w(text):
    """Display width: CJK/fullwidth = 1, ASCII = 0.5 (course-subtitles rule)."""
    return sum(1.0 if ord(c) > 0x2E7F else 0.5 for c in text)


def merge_latin_fragments(words):
    """Glue single-character latin fragments back onto an adjacent latin word
    (ASR often splits one term into 'v'+'olume'; a split latin word would
    break cues mid-term). Only merges fragments of length 1; never merges
    two multi-char latin words (that would invent terms)."""
    out = []
    for w in words:
        t = w["word"]
        if out:
            prev = out[-1]
            pt = prev["word"]
            prev_latin_end = (pt and pt[-1].isascii() and pt[-1].isalpha()
                              and pt[-1] not in _PUNCT_TAIL)
            # trailing single-char fragment glues onto the previous latin word
            if (len(t) == 1 and t.isascii() and t.isalpha()
                    and prev_latin_end):
                prev = dict(prev)
                prev["word"] = pt + t
                prev["end"] = w["end"]
                out[-1] = prev
                continue
            # leading single-char fragment glues onto the next latin word
            if (len(pt) == 1 and pt.isascii() and pt.isalpha()
                    and t and t[0].isascii() and t[0].isalpha()):
                merged = dict(w)
                merged["word"] = pt + t
                merged["start"] = prev["start"]
                out[-1] = merged
                continue
        if out and t and t[0].isascii() and t[0].isalpha() and \
                out[-1]["word"] and out[-1]["word"][-1].isascii() and \
                out[-1]["word"][-1].isalpha() and \
                out[-1]["word"][-1] not in _PUNCT_TAIL:
            w = dict(w)                     # adjacent latin words: keep the
            w["word"] = " " + t             # space glued so assembly shows it
        out.append(w)
    return out


def _join_chunk(chunk):
    """Assemble cue text: latin runs separated by single spaces."""
    text = ""
    for w in chunk:
        t = w["word"]
        if text and text[-1].isascii() and text[-1].isalpha() and \
                t and t[0].isascii() and t[0].isalpha():
            text += " "
        text += t.lstrip() if t.startswith(" ") and not text else t
    return text


def _punctuate(chunk, trailing):
    """Join word tokens, inserting a comma at speech pauses (fallback when
    the ASR did not punctuate; never drops ASR punctuation)."""
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


def strip_cross_boundary_punct(words):
    """funasr micro-fragment tails punctuate MID-WORD boundaries: word A
    ends with a tail and word B starts with a char that TOGETHER form one
    jieba word (应+该, 力+竭, 起+步, 只+是, 右+手 - all measured on real
    videos). The tail punctuation is a ct-punc fragment artifact, not a
    sentence end: strip it."""
    import jieba
    out = []
    for w in words:
        if out:
            prev = out[-1]
            core = prev["word"].rstrip("，。！？、；：")
            tail = prev["word"][len(core):] if core != prev["word"] else ""
            if tail and core and w["word"]:
                pair = core[-1] + w["word"][0]
                # comma artifacts strip via the pair test alone; sentence
                # tails (。！？) only when dangling on a SINGLE char (应。) -
                # real sentence ends like 结束了。+然后 must survive (了然
                # is a dictionary word - measured false positive)
                if len(pair) == 2 and                         list(jieba.cut(pair, HMM=False)) == [pair] and                         (tail[0] in "，、" or
                         pair[0] not in "了的着呢吧吗啊嘛呀哦"):
                    prev = dict(prev)
                    prev["word"] = core     # artifact tail removed
                    out[-1] = prev
        out.append(w)
    return out


def collapse_stutter(words, min_repeat=3):
    """ASR repetition artifact: the same multi-char token repeated >=3 times
    back-to-back (有两个两个两个) collapses to one. A doubled token is kept -
    real speech does repeat twice for emphasis (对吧对吧)."""
    out, run = [], []
    for w in words:
        if run and w["word"] == run[0]["word"]:
            run.append(w)
            continue
        if len(run) >= min_repeat and len(run[0]["word"]) >= 2:
            keep = dict(run[0])
            keep["end"] = run[-1]["end"]
            out.append(keep)               # one occurrence, full span
        else:
            out.extend(run)
        run = [w]
    if len(run) >= min_repeat and len(run[0]["word"]) >= 2:
        keep = dict(run[0])
        keep["end"] = run[-1]["end"]
        out.append(keep)
    else:
        out.extend(run)
    return out


def resegment(words, max_w=20.0, gap_s=0.9, dur_max=7.0, sentence_gap=0.55):
    """Re-chunk word timestamps into readable, punctuated caption lines.

    Gates (user-confirmed + course-subtitles contract):
    - PUNCTUATION RETENTION: funasr words carry real ct-punc tails - they
      are kept verbatim and DRIVE the segmentation (sentence tails 。！？ =
      hard break; comma tails break under length pressure: width >= 10 and
      (real gap >= 0.25s or width >= 20)). Pause-synthesized punctuation is
      a fallback that only runs when the stream carries none (whisper).
    - Lines break only on WORD boundaries; hard cap width/duration at word
      boundaries; a word token is never split however long.
    - Adjacent latin words are space-separated in the assembled text.
    """
    words = strip_cross_boundary_punct(collapse_stutter(merge_latin_fragments(
        [{"word": str(w.get("word", "")).strip(), "start": float(w["start"]),
          "end": float(w["end"])} for w in words
         if str(w.get("word", "")).strip()])))
    has_asr_punct = any(w["word"] and w["word"][-1] in _ALL_TAIL
                        for w in words)

    chunks, cur, prev_end = [], [], None

    def close():
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = []

    for w in words:
        gap = None if prev_end is None else w["start"] - prev_end
        tail = w["word"][-1] if w["word"] else ""
        if has_asr_punct:
            if cur and tail in _SENT_TAIL:
                cur.append(w)
                close()                     # trust ct-punc sentence breaks
                prev_end = w["end"]
                continue
            if cur and gap is not None and gap > gap_s:
                close()                     # real speech pause
        else:
            if cur and gap is not None and gap > 0.6:
                close()
        cur.append(w)
        prev_end = w["end"]
        width = disp_w("".join(x["word"] for x in cur))
        dur = cur[-1]["end"] - cur[0]["start"]
        if has_asr_punct and cur and tail in _COMMA_TAIL and \
                width >= 10 and (gap is not None and gap >= 0.25 or width >= 20):
            close()                         # comma under length pressure
        elif width > max_w or dur > dur_max:
            if has_asr_punct:
                # forced break: prefer the most recent comma-tailed word
                cut = max((i + 1 for i, x in enumerate(cur[:-1])
                           if x["word"] and x["word"][-1] in _COMMA_TAIL),
                          default=0)
                if cut < 3:
                    cut = len(cur) - 1
            else:
                cut = _cut_at_boundary(cur)
            if cut <= 0:
                cut = len(cur)              # single overlong word: keep whole
            chunks.append(cur[:cut])
            cur = cur[cut:]
    close()

    # post-chunk merge: a hard-cap/gap cut can land INSIDE a word pair
    # (时|候, 倒|过来) or shatter dramatic pauses into one-word lines
    # (我们/上去/看看/左). Merge neighbours when the boundary chars form
    # one jieba word, or when both sides are tiny and no sentence tail
    # stands between them.
    def _core(t):
        return t.rstrip("，。！？、；：")

    import jieba as _jb
    merged = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            ptxt = _core(_join_chunk(prev))
            ctxt = _join_chunk(chunk).lstrip()
            forms_word = False
            if ptxt and ctxt:
                pair = ptxt[-1] + ctxt[0]
                if len(pair) == 2 and list(_jb.cut(pair, HMM=False)) == [pair]:
                    forms_word = True
            small = disp_w(ptxt) + disp_w(_core(ctxt)) <= max_w
            raw_prev = _join_chunk(prev).rstrip()
            no_sent = not raw_prev.endswith(("。", "！", "？"))
            gap_here = (chunk[0]["start"] - prev[-1]["end"]) if prev and chunk                 else 9.9
            if ((forms_word and gap_here <= 0.9) or
                    (small and no_sent and gap_here <= 0.9
                     and ctxt[:1] not in "。！？")):
                merged[-1] = prev + chunk
                continue
        merged.append(chunk)
    chunks = merged

    lines = []
    for i, chunk in enumerate(chunks):
        nxt_gap = None
        if i + 1 < len(chunks):
            nxt_gap = chunks[i + 1][0]["start"] - chunk[-1]["end"]
        if has_asr_punct:
            text = _join_chunk(chunk)
            if not text.endswith(_PUNCT_TAIL):
                # every line closes with punctuation (user 2026-08-29:
                # bare continuation ends read as lost punctuation);
                # sentence end -> 。, any continuation -> ，
                if nxt_gap is None or nxt_gap >= sentence_gap:
                    text += "。"
                else:
                    text += "，"
        else:
            trailing = "。" if (nxt_gap is None or nxt_gap >= sentence_gap) else "，"
            text = _punctuate(chunk, trailing)
        if text.strip("，。 "):
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
    # ONE flat word stream: funasr segment boundaries are decoder fragments,
    # not sentence boundaries - chunking across them lets the punctuation
    # contract (and the cross-boundary artifact strip) see every boundary
    words_all = [w for seg in segments for w in seg.get("words") or []]
    lines = resegment(words_all) if words_all else []
    for line in lines:
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
    # wordless segments (engine fallback): emit verbatim as captions
    for seg in segments:
        if not seg.get("words"):
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
