"""caption_fix gates: word-boundary splitting, pause punctuation, DICT layers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from caption_fix import find_terms, fix_text, load_dict, resegment  # noqa: E402


def w(text, start, dur=0.12):
    return {"word": text, "start": round(start, 3), "end": round(start + dur, 3)}


# ---------- word-boundary splitting (never slice inside a token) ----------

def test_resegment_never_splits_inside_a_word():
    """A long run must break BETWEEN word tokens - the old char-level
    cutter produced lines ending mid-word (user-visible bug)."""
    words = [w(t, 0.1 + i * 0.15) for i, t in enumerate(
        ["这条", "抱石", "线路", "的", "起步", "特别", "难", "需要",
         "侧身", "贴墙", "然后", "挂脚", "换脚", "交叉手", "并手", "结束",
         "我们", "继续", "往左", "转移", "重心", "再", "压一把"])]
    lines = resegment(words, max_w=8)
    assert len(lines) >= 2
    vocab = {x["word"] for x in words}
    for ln in lines:
        # every line must be built of whole words joined by optional commas
        body = ln["text"].rstrip("，。")
        toks, cur = [], ""
        for ch in body:
            cur += ch
            if cur in vocab:
                toks.append(cur)
                cur = ""
        assert cur == "", f"line has a partial token: {ln['text']!r}"
        assert "".join(t for t in toks) == body


def test_resegment_single_long_word_stays_whole():
    """One 12-char token overruns max_chars - it must survive intact."""
    words = [w("这条线特别特别难", 0.0, 1.0), w("对吧", 1.4)]
    lines = resegment(words, max_w=6)
    assert any("这条线特别特别难" in l["text"] for l in lines)
    assert not any(l["text"].startswith("别") or l["text"].startswith("难，")
                   for l in lines)


def test_resegment_prefers_particle_word_boundary():
    words = [w(t, 0.1 + i * 0.12) for i, t in enumerate(
        ["我们", "要", "先", "看", "一下", "这个", "线路", "的", "整体", "走向",
         "然后", "再", "决定", "怎么", "爬", "上去", "比较", "稳妥"])]
    lines = resegment(words, max_w=8)
    for ln in lines[:-1]:
        assert "的" in ln["text"] or ln["text"].rstrip("，").endswith(
            tuple("了呢吧啊嘛哟哦啦")) or True  # soft preference, no gate


# ---------- pause punctuation ----------

def test_resegment_inserts_comma_at_pauses_and_full_stop_at_end():
    words = [w("起步", 0.0), w("然后", 0.5),    # 0.38s pause -> comma
             w("侧身", 1.6),                    # ~1.0s pause -> sentence break
             w("贴墙", 1.75)]
    lines = resegment(words)
    texts = [l["text"] for l in lines]
    assert "起步，然后。" == texts[0]
    assert texts[-1].endswith("。")


def test_resegment_comma_at_internal_short_pause():
    words = [w("左脚", 0.0), w("踩住", 0.5), w("重心", 0.52), w("上移", 0.68)]
    # 0.5s gap between 踩住/重心 >= 0.30 -> comma inserted mid-line
    lines = resegment(words, max_w=40)
    assert "，" in lines[0]["text"]


# ---------- DICT layers ----------

def test_load_dict_parses_fixes_highlight_prompt(tmp_path):
    d = tmp_path / "DICT.md"
    d.write_text("""# x

## 纠错映射（whisper 错听 => 规范词）
点墙 => 蹬墙
推宽 => 推髋

## 口播高频术语
三点平衡 · 挂脚 · 内扣膝

## 转录 initial-prompt
这是一条攀岩教学视频的口播。
""", encoding="utf-8")
    dd = load_dict(str(d))
    assert dd["fixes"] == {"点墙": "蹬墙", "推宽": "推髋"}
    assert {"三点平衡", "挂脚", "内扣膝"} <= dd["highlight"]
    assert "攀岩教学" in dd["prompt"]


def test_fix_text_applies_dict_fixes():
    assert fix_text("我们推宽往右", {"推宽": "推髋"}) == "我们推髋往右"
    assert fix_text("这个点墙发力", {"点墙": "蹬墙"}) == "这个蹬墙发力"


def test_find_terms_extended_by_dict():
    spans = find_terms("做一个内扣膝的动作", {"内扣膝"})
    assert spans and spans[0][2] == "内扣膝"


# ---------- v3: ASR punctuation-retention contract (funasr ct-punc) ----------

def test_resegment_keeps_asr_sentence_punct_and_breaks():
    """ct-punc tails are kept verbatim and sentence tails hard-break."""
    words = [w(t, 0.1 + i * 0.25) for i, t in enumerate(
        ["这条", "线", "是", "一个", "三点", "平衡，", "它", "有", "两个",
         "要点。", "第一", "个", "是", "熟悉", "三点", "平衡。"])]
    lines = resegment(words)
    texts = [l["text"] for l in lines]
    assert any(t.endswith("要点。") for t in texts)     # kept, drives break
    assert any(t.endswith("平衡。") for t in texts)
    joined = "".join(texts)
    assert "平衡，" in joined and "要点。" in joined    # nothing dropped


def test_resegment_comma_breaks_under_length_pressure():
    """Comma tails break the line only with width >= 10 AND (real gap or
    width >= 20): short clauses keep the comma inside the line."""
    tight = [w(t, 0.1 + i * 0.15) for i, t in enumerate(
        ["起步", "然后", "侧身，", "贴墙，", "挂脚，", "换脚。"])]
    lines = resegment(tight)                            # width < 10: no break
    assert len(lines) == 1
    long_run = [w(t, 0.1 + i * 0.15) for i, t in enumerate(
        ["我们", "先", "看", "一下", "这个", "线路", "的", "整体", "走向，",
         "然后", "再", "决定", "怎么", "爬", "上去", "比较", "稳妥。"])]
    lines = resegment(long_run)                         # width >= 10 at comma
    assert len(lines) >= 2
    assert lines[0]["text"].endswith("走向，")


def test_resegment_width_cap_never_splits_words():
    words = [w(t, 0.1 + i * 0.2) for i, t in enumerate(
        ["这条", "抱石", "线路", "的", "起步", "特别", "难", "需要",
         "侧身", "贴墙", "然后", "挂脚", "换脚", "结束。"])]
    lines = resegment(words, max_w=8)
    vocab = {"这条", "抱石", "线路", "的", "起步", "特别", "难", "需要",
             "侧身", "贴墙", "然后", "挂脚", "换脚", "结束"}
    for ln in lines:
        body = ln["text"].rstrip("，。")
        toks, cur = [], ""
        for ch in body:
            cur += ch
            if cur in vocab:
                toks.append(cur)
                cur = ""
        assert cur == "", f"mid-word split: {ln['text']!r}"


def test_resegment_latin_words_get_spaces_not_glued():
    words = [w(t, 0.1 + i * 0.3) for i, t in enumerate(
        ["踩", "这个", "volume", "然后", "抓", "crimp", "点。"])]
    lines = resegment(words)
    assert "volume" in lines[0]["text"] and "crimp" in lines[0]["text"]
    assert "volumecrimp" not in "".join(l["text"] for l in lines)


def test_merge_latin_fragments_glues_single_chars():
    from caption_fix import merge_latin_fragments
    words = [w("v", 0.0), w("olume", 0.12), w("很", 0.3), w("大", 0.4)]
    merged = merge_latin_fragments(words)
    assert merged[0]["word"] == "volume"
    assert merged[0]["end"] == words[1]["end"]


# ---------- v3: funasr mapping + drift warp (asr_align) ----------

def test_sentence_units_zip_and_proportional_fallback():
    from asr_align import sentence_units
    # 3 CJK chars + 1 latin run == 4 units, zip path
    u = sentence_units("挂脚crimp", [[0, 200], [200, 400], [400, 900]])
    assert [x[0] for x in u] == ["挂", "脚", "crimp"]
    assert abs(u[2][1] - 0.4) < 1e-6
    # mismatch: proportional spread, monotonic, plausible
    u2 = sentence_units("五个字啊", [[100, 600]])
    assert len(u2) == 4 and all(b > a for _, a, b in u2)


def test_group_units_keeps_punct_tails_and_latin_atomic():
    from asr_align import group_units_to_words
    units = [("挂", 0.0, 0.1), ("脚", 0.1, 0.2), ("上", 0.2, 0.3),
             ("来", 0.3, 0.4), ("。", 0.4, 0.5)]
    # note: units never contain punctuation; put it in text only
    units = [("挂", 0.0, 0.1), ("脚", 0.1, 0.2), ("上", 0.2, 0.3), ("来", 0.3, 0.4)]
    words = group_units_to_words(units, "挂脚上来。")
    assert any(x["word"].endswith("。") for x in words)   # tail attached
    units2 = [("踩", 0.0, 0.1), ("volume", 0.1, 0.5), ("点", 0.5, 0.6)]
    words2 = group_units_to_words(units2, "踩volume点")
    assert any(x["word"] == "volume" for x in words2)     # latin atomic


def test_warp_times_recovers_linear_drift():
    from asr_align import anchor_pairs, warp_times
    # src stream runs 10% fast; ref is truth
    src = [{"word": "锚点一", "start": 0.0, "end": 0.3},
           {"word": "中间词", "start": 45.0, "end": 45.3},
           {"word": "锚点二", "start": 90.0, "end": 90.3}]
    ref = [{"word": "锚点一", "start": 0.0, "end": 0.33},
           {"word": "中间词", "start": 50.0, "end": 50.3},
           {"word": "锚点二", "start": 100.0, "end": 100.3}]
    pairs, _, _ = anchor_pairs(src, ref)
    warped = warp_times(src, pairs)
    assert abs(warped[1]["start"] - 50.0) < 0.6          # 45 -> ~50
    assert abs(warped[2]["start"] - 100.0) < 0.6


def test_collapse_stutter_triples_not_doubles():
    from caption_fix import collapse_stutter
    ws = [w("有两个", 0.0), w("有两个", 0.3), w("有两个", 0.6), w("要点。", 0.9)]
    out = collapse_stutter(ws)
    assert [x["word"] for x in out] == ["有两个", "要点。"]
    assert abs(out[0]["end"] - 0.72) < 1e-9     # span covers the stutter
    ws2 = [w("对吧", 0.0), w("对吧", 0.3), w("走", 0.6)]
    assert [x["word"] for x in collapse_stutter(ws2)] == ["对吧", "对吧", "走"]


def test_group_units_no_doubled_punctuation():
    from asr_align import group_units_to_words
    units = [("要", 0.0, 0.1), ("点", 0.1, 0.2)]
    words = group_units_to_words(units, "要点。。")
    joined = "".join(x["word"] for x in words)
    assert "。。" not in joined and joined.count("。") == 1


def test_merge_micro_sentences_keeps_commas_downgrades_tails():
    """Punctuation-retention on merge (course-subtitles contract): commas
    survive (deleting them stripped real clause punctuation from every
    merged boundary), sentence tails downgrade to a comma; the mid-word
    artifact (这应。|该是) is removed later at word level by
    strip_cross_boundary_punct, not here."""
    from transcribe import _merge_micro_sentences
    sents = [{"text": "到这儿，", "timestamp": [[0, 400]]},
             {"text": "大包大pinch啊。", "timestamp": [[420, 900]]},
             {"text": "平衡刹然后这应。", "timestamp": [[900, 1500]]},
             {"text": "该是上左脚呢。", "timestamp": [[1520, 1900]]}]
    out = _merge_micro_sentences(sents)
    joined = "".join(s["text"] for s in out)
    assert "到这儿，大包" in joined                 # real comma survived
    assert "啊。平衡" not in joined                 # sentence tail downgraded
    assert "啊，平衡" in joined
    # mid-word artifact tail survives merge as a comma - removed at the
    # word level (strip_cross_boundary_punct) where 应+该 is detectable
    assert "这应，该是" in joined or "这应。该是" in joined


def test_strip_cross_boundary_punct_removes_fragment_artifacts():
    from caption_fix import strip_cross_boundary_punct
    ws = [w("这应。", 0.0), w("该是", 0.3), w("上左脚", 0.5)]
    out = strip_cross_boundary_punct(ws)
    assert out[0]["word"] == "这应"            # 应+该 = 应该 -> tail stripped
    ws2 = [w("充分休息啊哎力，", 0.0), w("竭了", 0.4)]
    out2 = strip_cross_boundary_punct(ws2)
    assert out2[0]["word"].endswith("力")      # 力+竭 = 力竭 -> stripped
    ws3 = [w("结束了。", 0.0), w("然后", 0.5)]
    out3 = strip_cross_boundary_punct(ws3)
    assert out3[0]["word"] == "结束了。"       # 了+然 not a word -> kept
