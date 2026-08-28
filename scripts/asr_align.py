"""Pure helpers for the FunASR caption engine (no heavy imports; testable).

Adapted from the course-subtitles pipeline (battle-tested mapping rules):
- funasr sentence units: latin/digit runs are ONE timestamped unit, each CJK
  char is one unit; when counts disagree, spread proportionally inside the
  sentence span (never emit a whole-sentence giant word);
- regroup timed units into jieba words (latin/digit runs pass through as
  atomic words); a token's trailing punctuation stays attached to the word
  (punctuation-retention contract: ASR punctuation is real and is kept);
- warp funasr word times onto a whisper reference timeline via difflib text
  anchors (funasr timestamps live in "pure speech stream" coordinates and
  drift ~ -3s/min on long continuous audio - measured -17s @ 285s).
"""
import bisect
import difflib
import re

PUNCT = "，。！？、；：,.!?;:…—“”\"'（）()<>《》«»·"
_UNIT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#'/-]*|[\u4e00-\u9fff]")


def sentence_units(seg_text, ts_ms):
    """Sentence text + funasr char timestamps -> [(unit, start_s, end_s)].

    Units whose count matches the timestamps zip 1:1 (latin runs are single
    units); a mismatch falls back to proportional spread across the sentence
    span so every unit still gets a plausible, monotonic time.
    """
    units = _UNIT_RE.findall(seg_text)
    if not units or not ts_ms:
        return []
    if len(units) == len(ts_ms):
        return [(u, s / 1000.0, e / 1000.0) for u, (s, e) in zip(units, ts_ms)]
    s0, e0 = ts_ms[0][0] / 1000.0, ts_ms[-1][1] / 1000.0
    total = sum(len(u) for u in units) or 1
    t, span = s0, max(e0 - s0, 0.012 * len(units))
    out = []
    for u in units:
        d = span * len(u) / total
        out.append((u, t, t + d))
        t += d
    return out


def group_units_to_words(timed_units, seg_text):
    """Timed units + sentence text -> words along jieba boundaries.

    Latin/digit units are atomic words; CJK stretches regroup into jieba
    tokens; a token's trailing punctuation chars attach to the word (kept,
    never dropped); single-token garbage spans clamp to 2.5s.
    """
    import jieba  # deferred: light, pure-python

    # char-index -> unit-index (units are maximal same-class runs)
    by_char = {}
    i = ui = 0
    while i < len(seg_text) and ui < len(timed_units):
        u = timed_units[ui][0]
        if seg_text.startswith(u, i):
            for off in range(len(u)):
                by_char[i + off] = ui
            i += len(u)
            ui += 1
        else:
            i += 1          # punctuation / whitespace: no unit

    def unit_span(tok, tpos):
        uis, seen = [], set()
        for off in range(len(tok)):
            u = by_char.get(tpos + off)
            if u is not None and u not in seen:
                seen.add(u)
                uis.append(u)
        return uis

    def tail_from(p):
        """Consume the punctuation run at p ONCE (positions recorded so the
        later punct-only jieba token cannot attach it a second time -
        doubled 。。，， was a real output bug). Consecutive identical
        punctuation collapses to one (ct-punc occasionally doubles)."""
        t = ""
        while p < len(seg_text) and seg_text[p] in PUNCT and p not in consumed:
            if not t or t[-1] != seg_text[p]:
                t += seg_text[p]
            consumed.add(p)
            p += 1
        return t

    def strip_tok_punct(tok, tpos):
        """jieba can keep trailing punctuation INSIDE a token (要点。。) -
        strip it and mark consumed, so it flows through tail_from once."""
        core = tok.rstrip("".join(c for c in PUNCT))
        for i2 in range(tpos + len(core), tpos + len(tok)):
            consumed.discard(i2)            # not yet attached; tail_from adds
        return core

    consumed = set()
    out, used = [], set()
    pos = 0
    for tok in jieba.cut(seg_text, HMM=False):
        tpos = pos
        pos += len(tok)
        if not tok.strip():
            continue
        if all(c in PUNCT for c in tok):
            live = "".join(c for i2, c in enumerate(tok, tpos)
                           if i2 not in consumed)
            if out and live:     # standalone punct not yet consumed: tail it
                out[-1]["word"] += live
                for i2 in range(tpos, tpos + len(tok)):
                    consumed.add(i2)
            continue
        core = strip_tok_punct(tok, tpos)
        if not core:
            continue
        uis = [u for u in unit_span(core, tpos) if u not in used]
        if not uis:
            continue        # token overlaps already-consumed units (safety)
        used.update(uis)
        grp = [timed_units[u] for u in uis]
        word = {"word": core + tail_from(tpos + len(core)),
                "start": grp[0][1], "end": grp[-1][2]}
        if word["word"][0].isascii():
            word["end"] = min(word["end"], word["start"] + 2.5)
        out.append(word)
    for u in range(len(timed_units)):     # unconsumed safety net
        if u not in used:
            out.append({"word": timed_units[u][0],
                        "start": timed_units[u][1], "end": timed_units[u][2]})
    out.sort(key=lambda w: w["start"])
    return out


def char_stream(words):
    """[{word,start,end}] -> (chars, char_time) with intra-word spread."""
    chars, times = [], []
    for w in words:
        t = str(w["word"])
        n = len(t) or 1
        d = (w["end"] - w["start"]) / n
        for k, c in enumerate(t):
            chars.append(c)
            times.append(w["start"] + d * k)
    return "".join(chars), times


def anchor_pairs(src_words, ref_words, min_block=4, tail_tol=5.0):
    """Difflib matching blocks between two word streams' char texts ->
    [(src_time, ref_time)] anchors.

    Tail anchors are only appended when the two streams end within tail_tol
    of each other: whisper hallucination tails (looped text after the last
    real speech, sometimes +10s) would otherwise stretch the whole warp.
    When the tails diverge, extrapolate from the last two real anchors
    instead of trusting the raw stream end."""
    s_text, s_times = char_stream(src_words)
    r_text, r_times = char_stream(ref_words)
    sm = difflib.SequenceMatcher(a=s_text, b=r_text, autojunk=False)
    pairs = [(0.0, 0.0)]
    last_block = None
    for b in sm.get_matching_blocks():
        if b.size >= min_block:
            pairs.append((s_times[b.a], r_times[b.b]))
            last_block = b
    pairs = sorted(set(pairs))
    s_end, r_end = s_times[-1], r_times[-1]
    # Tail anchor is trusted only when the last matching block reaches BOTH
    # stream ends (content-aligned tails). A whisper hallucination tail
    # (extra looped text after the last real speech) leaves the block short
    # on the ref side - then extrapolate the real trend instead, capped at
    # the ref end.
    src_covered = last_block is not None and \
        last_block.a + last_block.size >= len(s_text) - 2
    ref_covered = last_block is not None and \
        last_block.b + last_block.size >= len(r_text) - 2
    if src_covered and ref_covered:
        pairs.append((s_end, r_end))
    elif len(pairs) >= 2:
        (x0, y0), (x1, y1) = pairs[-2], pairs[-1]
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 1.0
        pairs.append((s_end, min(r_end, y1 + (s_end - x1) * slope)))
    return pairs, s_text, r_text


def warp_times(words, pairs):
    """Piecewise-linear map of word times through anchor pairs
    (src=funasr compressed -> ref=whisper true)."""
    pairs = sorted(pairs)
    xs = [p[0] for p in pairs]
    out = []
    for w in words:
        s2 = _interp(w["start"], xs, pairs)
        e2 = _interp(w["end"], xs, pairs)
        if e2 <= s2:
            e2 = s2 + 0.08
        out.append({**w, "start": round(s2, 3), "end": round(e2, 3)})
    return out


def _interp(t, xs, pairs):
    if t <= xs[0]:
        return pairs[0][1] + (t - xs[0])
    if t >= xs[-1]:
        return pairs[-1][1] + (t - xs[-1])
    i = bisect.bisect_right(xs, t) - 1
    x0, y0 = pairs[i]
    x1, y1 = pairs[i + 1]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
