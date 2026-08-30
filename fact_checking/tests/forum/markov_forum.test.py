"""
MarkovForum 测试：
1. 用 ExtractClaimAgent 处理声明（默认单条，必要时少量子声明）；
2. 对每条输出声明并行跑一轮 Markov 辩论；
3. 将完整结果写入 tests/forum/example/markov_forum_result.json。

python -B tests/forum/markov_forum.test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "tests" / "forum" / "example"
OUTPUT_JSON = OUTPUT_DIR / "markov_forum_result.json"
MAX_WORKERS = 4

from agents.claim.extract_claim_agent import ExtractClaimAgent
from config import load_env
from forum.markov_forum import MarkovForum
from utils.thread_pool import run_in_thread_pool


def _run_debate_for_sub_claim(
    payload: Tuple[int, str, float, int, int, str],
) -> Dict[str, Any]:
    """
    单个子声明的辩论（在线程池中执行，各自 new MarkovForum）。
    payload: (index, claim_text, weight, min_debate_rounds, max_debate_rounds, model)
    """
    idx, claim_text, weight, min_r, max_r, model = payload
    forum = MarkovForum(
        min_debate_rounds=min_r,
        max_debate_rounds=max_r,
        model=model,
    )
    debate = forum.verify(claim=claim_text)
    return {
        "index": idx,
        "claim": claim_text,
        "weight": weight,
        "debate": debate,
    }


def run_markov_forum_test() -> Dict[str, Any]:
    load_env()

    complex_claim = (
        "76-85% of people with severe mental disorder receive no treatment in low and middle income countries."
    )

    min_debate_rounds = 1
    max_debate_rounds = 2
    model = "deepseek-chat"

    print("[1/2] 分解声明 …")
    extract_agent = ExtractClaimAgent(max_tokens=800)
    extraction = extract_agent.extract(complex_claim)

    items: List[Dict[str, Any]] = list(extraction.get("items") or [])
    if not items:
        claims = extraction.get("claims") or []
        weights = extraction.get("weights") or []
        if claims:
            items = [
                {"claim": c, "weight": weights[i] if i < len(weights) else 1.0 / len(claims)}
                for i, c in enumerate(claims)
            ]
        else:
            items = [{"claim": complex_claim, "weight": 1.0}]

    n = len(items)
    workers = min(MAX_WORKERS, n)
    print(f"[2/2] 并行辩论：{n} 条子声明，线程数={workers}（上限 {MAX_WORKERS}）")

    payloads: List[Tuple[int, str, float, int, int, str]] = []
    for i, it in enumerate(items):
        c = str(it.get("claim", "")).strip()
        w = float(it.get("weight", 0.0))
        if not c:
            continue
        payloads.append((i, c, w, min_debate_rounds, max_debate_rounds, model))

    if not payloads:
        payloads = [(0, complex_claim.strip(), 1.0, min_debate_rounds, max_debate_rounds, model)]

    sub_results: List[Dict[str, Any]] = run_in_thread_pool(
        _run_debate_for_sub_claim,
        payloads,
        max_workers=MAX_WORKERS,
    )

    output: Dict[str, Any] = {
        "original_complex_claim": complex_claim,
        "extraction": extraction,
        "debate_settings": {
            "min_debate_rounds": min_debate_rounds,
            "max_debate_rounds": max_debate_rounds,
            "model": model,
            "max_workers": min(MAX_WORKERS, len(payloads)),
        },
        "sub_claim_debates": sub_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"已写入: {OUTPUT_JSON}")

    return output


if __name__ == "__main__":
    run_markov_forum_test()
