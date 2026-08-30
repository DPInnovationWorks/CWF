"""
批量运行 claims_51 的 MarkovForum 讨论 + 构图，并导出每条声明的最终结论。

python -B tests/forum/claims_51_forum2graph_batch.test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_JSON = ROOT / "tests" / "data" / "claims_51.json"
OUTPUT_DIR = ROOT / "tests" / "forum" / "example"
OUTPUT_JSON = OUTPUT_DIR / "claims_51_final_claims.json"

from config import load_env
from forum.markov_forum import MarkovForum
from graph.comm_graph import CommGraph
from utils.forum2graph import markov_forum_result_to_comm_graph_dicts
from utils.thread_pool import run_in_thread_pool


def _build_single_markov_result(claim_text: str, debate: Dict[str, Any]) -> Dict[str, Any]:
    """将单条声明的辩论结果包装成 forum2graph 所需结构。"""
    return {
        "original_complex_claim": claim_text,
        "sub_claim_debates": [
            {
                "index": 0,
                "claim": claim_text,
                "weight": 1.0,
                "debate": debate,
            }
        ],
    }


def _process_one_item(payload: Tuple[Dict[str, Any], int, int, str]) -> Dict[str, Any]:
    item, min_rounds, max_rounds, model = payload
    claim_id = item.get("id")
    claim_text = str(item.get("claim") or "").strip()
    if not claim_text:
        return {
            "id": claim_id,
            "domain": item.get("domain"),
            "complexity": item.get("complexity"),
            "truth_label": item.get("truth_label"),
            "claim": claim_text,
            "error": "empty claim",
        }

    try:
        # 每个线程独立实例，避免共享 forum 状态导致并发问题
        forum = MarkovForum(
            min_debate_rounds=min_rounds,
            max_debate_rounds=max_rounds,
            model=model,
        )
        debate = forum.verify(claim=claim_text)

        one_result = _build_single_markov_result(claim_text, debate)
        graph_dicts = markov_forum_result_to_comm_graph_dicts(one_result)
        assert len(graph_dicts) == 2, "单条声明应得到 original + sub_claim 两张图"
        original_graph = CommGraph.from_dict(graph_dicts[0])
        claim_original_conf = original_graph.infer_claim_confidence()

        final = debate.get("final") or {}
        return {
            "id": claim_id,
            "domain": item.get("domain"),
            "complexity": item.get("complexity"),
            "truth_label": item.get("truth_label"),
            "claim": claim_text,
            "final_claim": {
                "opinion": final.get("opinion"),
                "stance": final.get("stance"),
                "factuality": final.get("factuality"),
                "total_rounds": debate.get("total_rounds"),
                "claim_original_conf": claim_original_conf,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": claim_id,
            "domain": item.get("domain"),
            "complexity": item.get("complexity"),
            "truth_label": item.get("truth_label"),
            "claim": claim_text,
            "error": str(exc),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行 claims_51 的讨论构图测试")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 条声明（用于快速调试）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="线程数（默认 4）",
    )
    return parser.parse_args()


def run_claims_51_batch_test(limit: int | None = None, max_workers: int = 4) -> Dict[str, Any]:
    load_env()

    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = list(payload.get("items") or [])
    assert items, f"未读取到声明项: {INPUT_JSON}"
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须是正整数")
        items = items[:limit]
    if max_workers <= 0:
        raise ValueError("--max-workers 必须是正整数")

    min_debate_rounds = 1
    max_debate_rounds = 2
    model = "deepseek-chat"
    total = len(items)
    workers = min(max_workers, total)
    print(f"开始批量处理: {total} 条声明，线程数={workers}")
    start_ts = time.time()

    payloads: List[Tuple[Dict[str, Any], int, int, str]] = [
        (item, min_debate_rounds, max_debate_rounds, model) for item in items
    ]
    records = run_in_thread_pool(
        _process_one_item,
        payloads,
        max_workers=workers,
        desc="处理进度",
        show_progress=True,
    )

    elapsed = time.time() - start_ts
    print(f"处理完成: {len(records)}/{total}，耗时 {elapsed:.1f}s")

    output = {
        "source": str(INPUT_JSON.relative_to(ROOT)),
        "total_items": len(items),
        "processed_items": len(records),
        "records": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写入: {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    args = _parse_args()
    run_claims_51_batch_test(limit=args.limit, max_workers=args.max_workers)
