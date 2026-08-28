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


# ---------- v3: five deterministic gates (course-subtitles contract) ----------

def _cap(text, s=0.0, e=3.0):
    return {"start": s, "end": e, "text": text, "terms": []}


def test_gates_reject_invented_latin_terms():
    from caption_llm import apply_corrections
    caps = [_cap("踩住这个 volume 很稳")]
    fixes = [{"i": 0, "text": "踩住这个 sloper 很稳"}]   # invented token
    out = apply_corrections(caps, fixes)
    assert out[0]["text"] == "踩住这个 volume 很稳"      # rejected


def test_gates_accept_whitelisted_terms():
    from caption_llm import apply_corrections
    caps = [_cap("踩住这个 volum 很稳")]
    fixes = [{"i": 0, "text": "踩住这个 volume 很稳"}]   # typo -> dict term
    out = apply_corrections(caps, fixes, whitelist=["volume"])
    assert out[0]["text"] == "踩住这个 volume 很稳"


def test_gates_reject_deletions_via_coverage():
    from caption_llm import apply_corrections
    caps = [_cap("我们要先看一下这个线路的整体走向再决定")]
    fixes = [{"i": 0, "text": "看下线路"}]               # massive deletion
    out = apply_corrections(caps, fixes)
    assert "整体走向" in out[0]["text"]                   # original kept


def test_gates_reject_speed_overflow():
    from caption_llm import apply_corrections
    caps = [_cap("起步", 0.0, 0.2)]                      # 2 chars in 0.2s
    fixes = [{"i": 0, "text": "起步然后侧身贴墙挂脚换脚"}]  # way too long
    out = apply_corrections(caps, fixes)
    assert out[0]["text"] == "起步"
