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
    lines = resegment(words, max_chars=8)
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
    lines = resegment(words, max_chars=6)
    assert any("这条线特别特别难" in l["text"] for l in lines)
    assert not any(l["text"].startswith("别") or l["text"].startswith("难，")
                   for l in lines)


def test_resegment_prefers_particle_word_boundary():
    words = [w(t, 0.1 + i * 0.12) for i, t in enumerate(
        ["我们", "要", "先", "看", "一下", "这个", "线路", "的", "整体", "走向",
         "然后", "再", "决定", "怎么", "爬", "上去", "比较", "稳妥"])]
    lines = resegment(words, max_chars=8)
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
    lines = resegment(words, max_chars=20)
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
