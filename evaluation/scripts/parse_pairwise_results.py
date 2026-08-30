from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from _shared import load_tqdm, read_jsonl, should_disable_tqdm, write_jsonl


STRICT_CHOICE_PATTERN = re.compile(r"^[AB]$")
TOKEN_PATTERN = re.compile(r"\b([AB])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析 pairwise raw results。")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("artifacts/pairwise/results/pairwise_raw_results.jsonl"),
        help="pairwise raw results 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pairwise/results"),
        help="解析结果输出目录",
    )
    return parser.parse_args()


def normalize_choice(raw_text: str) -> tuple[str | None, str]:
    text = raw_text.strip()
    if STRICT_CHOICE_PATTERN.fullmatch(text):
        return text, "ok"
    candidate = text.rstrip(".!?:;，。！？；：").strip()
    if STRICT_CHOICE_PATTERN.fullmatch(candidate):
        return candidate, "ok"
    tokens = TOKEN_PATTERN.findall(text.upper())
    unique_tokens = sorted(set(tokens))
    if len(unique_tokens) == 1:
        return unique_tokens[0], "ok"
    return None, "invalid_choice"


def resolve_winner(row: dict[str, Any], winner_label: str) -> tuple[str, str]:
    left_baseline = row["left_baseline"]
    right_baseline = row["right_baseline"]
    if winner_label == "A":
        return left_baseline, right_baseline
    return right_baseline, left_baseline


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.results_file.exists():
        raise SystemExit(f"raw results 文件不存在: {args.results_file}")

    raw_rows = read_jsonl(args.results_file)
    parsed_rows: list[dict[str, Any]] = []
    progress = tqdm(
        raw_rows,
        desc="解析 pairwise results",
        unit="row",
        disable=should_disable_tqdm(),
    )
    for row in progress:
        request_status = row.get("request_status", "error")
        raw_output = str(row.get("raw_model_output", ""))
        if request_status != "ok":
            winner_label = None
            winner_baseline = None
            loser_baseline = None
            parse_status = "request_error"
        else:
            winner_label, parse_status = normalize_choice(raw_output)
            if winner_label is None:
                winner_baseline = None
                loser_baseline = None
            else:
                winner_baseline, loser_baseline = resolve_winner(row, winner_label)

        parsed_rows.append(
            {
                "comparison_id": row["comparison_id"],
                "domain": row["domain"],
                "audience": row["audience"],
                "index": row["index"],
                "topic_text": row["topic_text"],
                "baseline_a": row["baseline_a"],
                "baseline_b": row["baseline_b"],
                "left_baseline": row["left_baseline"],
                "right_baseline": row["right_baseline"],
                "left_source_file": row.get("left_source_file"),
                "right_source_file": row.get("right_source_file"),
                "winner_label": winner_label,
                "winner_baseline": winner_baseline,
                "loser_baseline": loser_baseline,
                "request_status": request_status,
                "parse_status": parse_status,
                "raw_model_output": raw_output,
                "error_message": row.get("error_message", ""),
                "retry_count": row.get("retry_count"),
                "latency_ms": row.get("latency_ms"),
                "model": row.get("model"),
                "judged_at": row.get("judged_at"),
            }
        )
    progress.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "pairwise_results.jsonl"
    csv_path = args.output_dir / "pairwise_results.csv"
    write_jsonl(jsonl_path, parsed_rows)
    write_csv(
        csv_path,
        parsed_rows,
        fieldnames=[
            "comparison_id",
            "domain",
            "audience",
            "index",
            "topic_text",
            "baseline_a",
            "baseline_b",
            "left_baseline",
            "right_baseline",
            "left_source_file",
            "right_source_file",
            "winner_label",
            "winner_baseline",
            "loser_baseline",
            "request_status",
            "parse_status",
            "raw_model_output",
            "error_message",
            "retry_count",
            "latency_ms",
            "model",
            "judged_at",
        ],
    )
    print(f"Pairwise 解析结果已写入: {jsonl_path}")
    print(f"Pairwise 解析表格已写入: {csv_path}")


if __name__ == "__main__":
    main()
