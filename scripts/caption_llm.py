"""LLM-based climbing-term correction for captions (runs after caption_fix).

Whisper + the rule dictionary still mis-hear domain jargon and names
(同音错听的人名/术语 -> 正确写法, 如 倒动星->倒重心, 遮膝->侧膝...). A fast chat model (GLM-4.7-FlashX)
fixes exactly those, under a strict contract:

- ONLY homophone / near-sound substitutions for climbing terms, route grades
  and names listed in the user dictionary (DICT.md) or standard climbing
  vocabulary. NOTHING else may change - no rewriting, no re-punctuation,
  no added or dropped content.
- Caption count, order and timing are untouched -> audio/subtitle sync is
  preserved by construction.
- Every returned fix is guarded by a similarity check; anything that rewrote
  the sentence (difflib ratio < 0.8) is rejected and the original kept.

Usage:
  python caption_llm.py captions.json DICT.md -o captions_llm.json \
      [--model glm-4.7-flashx] [--api-key <key>]
API key resolution: --api-key > $ZHIPUAI_API_KEY > $ZHIPU_API_KEY > $GLM_API_KEY.
"""
import argparse
import difflib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "glm-4.7-flashx"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

SYSTEM_PROMPT = """你是攀岩视频字幕的术语纠错器。用户给出按时间顺序排列的字幕列表和一个自定义词典。
你的唯一任务：修正与发音相同或相近的【攀岩术语、线路难度、人名】的错别字。
铁律：
1. 只做同音/近音字词替换（例如 倒动星→倒重心、遮膝→侧膝、真身→转身；人名按 DICT.md 名单纠同音）。
2. 除错别字外，一个字都不许改：不增、不删、不调语序、不改标点、不做通顺化。
3. 不确定是不是术语听错时，保持原样。
4. 只输出需要修改的条目，格式为 JSON 数组：[{"i":<字幕序号>,"text":"<修正后的整条字幕>"}]
5. 没有任何需要修改的条目时，输出 []。不要输出任何解释。"""


def load_dict(dict_path):
    """DICT.md -> header lines (grades/names/short entries), verbatim."""
    lines = [ln.strip() for ln in
             Path(dict_path).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    return lines


def apply_corrections(captions, fixes, min_ratio=0.8, min_cover=0.88,
                      max_cps=12.0, whitelist=None):
    """Pure: apply {"i": idx, "text": new} fixes through five deterministic
    gates (course-subtitles v1.7 contract). Timing fields are never touched;
    highlights (terms) are recomputed for changed lines.

    Gates per fix:
    1. similarity  - difflib ratio >= min_ratio (no sentence rewrites)
    2. coverage    - anti-deletion: shrinkage <=20%, >=75% chars kept
                     (word-by-word proofreading changes chars, never deletes)
    3. whitelist   - latin tokens in the new text must come from the original
                     line or the dictionary (the model may not invent terms)
    4. speed       - <= max_cps chars/sec over the caption window (a wildly
                     longer line cannot fit its slot)
    5. index       - fixes map onto existing caption indices only
    """
    import re as _re
    from caption_fix import find_terms
    whitelist = set(whitelist or ())
    out = [dict(c) for c in captions]
    for fx in fixes:
        i, new = fx.get("i"), fx.get("text")
        if not isinstance(i, int) or not (0 <= i < len(out)) or not new:
            continue                                   # gate 5: index
        old = out[i]["text"]
        new = new.strip()
        if new == old:
            continue
        sm = difflib.SequenceMatcher(None, old, new)
        if sm.ratio() < min_ratio:                     # gate 1: similarity
            continue
        kept = sum(b.size for b in sm.get_matching_blocks())
        # gate 2: coverage = anti-deletion. Substitution proofreading keeps
        # the length; single-char fixes in short lines must pass. Reject
        # shrinkage >20% or lines losing >25% of their chars.
        if len(new) < len(old) * 0.8 or kept / max(len(old), 1) < 0.75:
            continue
        tok_re = r"[A-Za-z][A-Za-z0-9.+#'-]*"
        allowed = whitelist | set(_re.findall(tok_re, old))
        if any(tok not in allowed for tok in _re.findall(tok_re, new)):
            continue                                   # gate 3: whitelist
        dur = max(out[i]["end"] - out[i]["start"], 0.1)
        if len(new) / dur > max_cps:                   # gate 4: speed
            continue
        out[i]["text"] = new
        out[i]["terms"] = [{"term": t, "off": a} for a, b, t in find_terms(new)]
    return out


def call_glm(captions, dict_entries, model, api_key, base_url, timeout=60):
    """One chat-completion round; returns the parsed fixes list."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "自定义词典（线路难度/人名/术语）": dict_entries,
                "字幕列表": [{"i": i, "text": c["text"]}
                             for i, c in enumerate(captions)],
            }, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    content = resp["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", content, re.S)  # tolerate markdown fences
    if not m:
        raise ValueError(f"model returned no JSON array: {content[:200]}")
    return json.loads(m.group(0))


def _load_env_file():
    """Auto-load ZHIPUAI_API_KEY from .env (cwd, then skill root) if the
    environment doesn't carry it. Placeholder values are ignored."""
    if os.environ.get("ZHIPUAI_API_KEY"):
        return
    for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
        env = base / ".env"
        if not env.exists():
            continue
        for ln in env.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith("ZHIPUAI_API_KEY="):
                val = ln.partition("=")[2].strip().strip('"').strip("'")
                if val and "填" not in val and "placeholder" not in val.lower():
                    os.environ["ZHIPUAI_API_KEY"] = val
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captions")
    ap.add_argument("dict_md")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args()

    _load_env_file()
    api_key = (args.api_key or os.environ.get("ZHIPUAI_API_KEY")
               or os.environ.get("ZHIPU_API_KEY")
               or os.environ.get("GLM_API_KEY"))
    if not api_key:
        print("ERROR: no API key. Pass --api-key or set ZHIPUAI_API_KEY "
              "(https://open.bigmodel.cn). Corrections were NOT applied.",
              file=sys.stderr)
        sys.exit(2)

    captions = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    fixes = call_glm(captions, load_dict(args.dict_md), args.model,
                     api_key, args.base_url)
    fixed = apply_corrections(captions, fixes)
    Path(args.out).write_text(json.dumps(fixed, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    n = sum(1 for a, b in zip(captions, fixed) if a["text"] != b["text"])
    print(f"corrected {n}/{len(fixed)} captions -> {args.out}")
    for a, b in zip(captions, fixed):
        if a["text"] != b["text"]:
            print(f"  [{a['text']}] -> [{b['text']}]")


if __name__ == "__main__":
    main()
