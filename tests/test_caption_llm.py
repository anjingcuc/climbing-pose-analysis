import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caption_llm import apply_corrections


def cap(i, text, start=1.0, end=2.0):
    return {"start": start, "end": end, "text": text, "terms": []}


def test_apply_corrections_term_only():
    caps = [cap(0, "起步一个真身回来"), cap(1, "倒重心导过来")]
    out = apply_corrections(caps, [
        {"i": 0, "text": "起步一个转身回来"},
        {"i": 1, "text": "倒重心导过来"},   # unchanged -> no-op
    ])
    assert out[0]["text"] == "起步一个转身回来"
    assert out[1]["text"] == "倒重心导过来"
    # timing untouched -> audio/subtitle sync preserved
    assert out[0]["start"] == 1.0 and out[0]["end"] == 2.0


def test_apply_corrections_rejects_rewrites():
    caps = [cap(0, "我们看一下脚点")]
    out = apply_corrections(caps, [
        {"i": 0, "text": "让我们先观察一下脚上的支点"},  # rewrite, not a fix
        {"i": 0, "text": ""},                            # empty
        {"i": 9, "text": "越界"},                        # out of range
    ])
    assert out[0]["text"] == "我们看一下脚点"


def test_apply_corrections_recomputes_highlights():
    caps = [cap(0, "真身回来侧身")]
    out = apply_corrections(caps, [{"i": 0, "text": "转身回来侧身"}])
    terms = {t["term"] for t in out[0]["terms"]}
    assert "侧身" in terms               # HIGHLIGHT term re-found after fix
    assert all(t["off"] == out[0]["text"].index(t["term"]) for t in out[0]["terms"])
