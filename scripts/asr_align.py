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

    KEY INVARIANT: "".join(unit strings) == seg_text stripped of punctuation
    and whitespace ("core text"). So jieba runs on the CLEAN core text (never
    on the raw sentence: spaces/punct inside it shatter jieba into single
    chars - half the words came out 1-char and punctuation attached one word
    early; measured 247/457 single-char words before this fix). Tokens map
    back to units by core-offset, punctuation tails by original position.
    """
    import jieba  # deferred: light, pure-python

    # core text + maps: core position -> unit index / original index
    core_chars, core2unit, core2orig = [], [], []
    unit_start = {}                     # unit idx -> its core start offset
    oi = 0
    for ui, (u, _, _) in enumerate(timed_units):
        unit_start[ui] = len(core_chars)
        for off, c in enumerate(u):
            # advance the original index to this unit's position
            while oi < len(seg_text) and not seg_text.startswith(u, oi):
                oi += 1
            core_chars.append(c)
            core2unit.append(ui)
            core2orig.append(oi + off)
        oi += len(u)
    core_text = "".join(core_chars)

    consumed = set()                    # original positions already tailed

    def tail_from(orig_pos):
        """Punctuation run at an ORIGINAL position, consumed once; identical
        consecutive punctuation collapses (ct-punc occasionally doubles)."""
        t = ""
        p = orig_pos
        while p < len(seg_text) and seg_text[p] in PUNCT and p not in consumed:
            if not t or t[-1] != seg_text[p]:
                t += seg_text[p]
            consumed.add(p)
            p += 1
        return t

    out, used = [], set()
    pos = 0
    for tok in jieba.cut(core_text, HMM=False):
        tpos = pos
        pos += len(tok)
        if not tok.strip():
            continue
        uis, seen = [], set()
        for off in range(len(tok)):
            u = core2unit[tpos + off]
            if u not in seen:
                seen.add(u)
                if u not in used:
                    uis.append(u)
        if not uis:
            continue                    # safety: token fully consumed already
        used.update(uis)
        grp = [timed_units[u] for u in uis]
        if tpos + len(tok) < len(core2orig):
            nxt_orig = core2orig[tpos + len(tok)]
        else:                              # token ends the core: punct run
            nxt_orig = core2orig[-1] + 1   # starts right after its last char
        word = {"word": tok + tail_from(nxt_orig),
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


def anchor_pairs(src_words, ref_words, min_block=3):
    """Difflib matching blocks between two word streams' char texts ->
    [(src_time, ref_time)] anchors. Both block ENDS anchor too (twice the
    constraints: piecewise-linear interpolation across sparse-anchor gaps
    missed local drift by seconds with start-only anchors - measured -3.3s
    mid-film before this fix).

    Tail anchors are only appended when the last matching block reaches BOTH
    stream ends (content-aligned tails). A whisper hallucination tail
    (extra looped text after the last real speech) leaves the block short
    on the ref side - then extrapolate the real trend instead of trusting
    the raw stream end."""
    s_text, s_times = char_stream(src_words)
    r_text, r_times = char_stream(ref_words)
    sm = difflib.SequenceMatcher(a=s_text, b=r_text, autojunk=False)
    pairs = [(0.0, 0.0)]
    last_block = None
    for b in sm.get_matching_blocks():
        if b.size >= min_block:
            pairs.append((s_times[b.a], r_times[b.b]))
            pairs.append((s_times[b.a + b.size - 1],
                          r_times[b.b + b.size - 1]))
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


def _filter_pairs(pairs, lo=0.55, hi=1.5, span=3.0):
    """Drop anchors that bend the warp: real speech-time drift is bounded
    (funasr compresses by silence-dropping; local slope stays ~0.85-1.1).
    An anchor whose local slope vs the last kept anchor falls outside
    [lo, hi] over a >= span window is a spurious match - whisper's
    mid-stream hallucination loops repeat text, and difflib can bind a
    funasr phrase to the WRONG copy, pulling the warp by seconds."""
    if len(pairs) < 3:
        return pairs
    keep = [pairs[0]]
    for p in pairs[1:]:
        x0, y0 = keep[-1]
        if p[0] - x0 >= span:
            slope = (p[1] - y0) / (p[0] - x0)
            if not (lo <= slope <= hi):
                continue          # suspicious anchor: skip it
        keep.append(p)
    return keep


def warp_times(words, pairs):
    """Piecewise-linear map of word times through anchor pairs
    (src=funasr compressed -> ref=whisper true)."""
    pairs = _filter_pairs(sorted(pairs))
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
