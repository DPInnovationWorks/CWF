from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from _shared import METRIC_ORDER, METRIC_SUMMARY_COLUMNS, load_tqdm, should_disable_tqdm


STRICT_ONE_DECIMAL_PATTERN = re.compile(r"^(?:[0-4]\.\d|5\.0)$")
NUMERIC_ONLY_PATTERN = re.compile(r"^(?:[0-4](?:\.\d+)?|5(?:\.0+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析 Batch 结果并汇总三项指标分数。")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("artifacts/batch_outputs/latest.output.jsonl"),
        help="poll_batch_job.py 下载得到的输出文件路径",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("artifacts/manifests/batch_manifest.jsonl"),
        help="build_metric_batch_jsonl.py 生成的 manifest 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="解析结果输出目录",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="将当前结果按 custom_id 合并进 output-dir 下的既有结果，而不是直接覆盖。",
    )
    return parser.parse_args()


METRIC_SCORE_FIELDNAMES = [
    "custom_id",
    "metric",
    "domain",
    "baseline",
    "audience",
    "index",
    "source_file",
    "topic_text",
    "source_prompt_field",
    "weight_id",
    "score",
    "parse_status",
    "raw_model_output",
    "batch_id",
]

SUMMARY_FIELDNAMES = [
    "domain",
    "baseline",
    "audience",
    "index",
    "source_file",
    "topic_text",
    "source_prompt_field",
    "cognitive_load_score",
    "personalization_alignment_score",
    "reader_attitude_score",
    "overall_personalization_score",
    "summary_status",
    "batch_id",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path} 第 {line_number} 行不是 JSON 对象")
            rows.append(row)
    return rows


def normalize_score(raw_text: str) -> tuple[str | None, str]:
    text = raw_text.strip()
    if STRICT_ONE_DECIMAL_PATTERN.fullmatch(text):
        return text, "ok"
    if NUMERIC_ONLY_PATTERN.fullmatch(text):
        try:
            decimal_value = Decimal(text)
        except InvalidOperation:
            return None, "invalid_score"
        normalized = decimal_value.quantize(Decimal("0.0"), rounding=ROUND_HALF_UP)
        if normalized < Decimal("0.0") or normalized > Decimal("5.0"):
            return None, "invalid_score"
        return format(normalized, ".1f"), "ok"
    return None, "invalid_score"


def extract_content(result_row: dict[str, Any]) -> tuple[str, str]:
    response = result_row.get("response")
    if not isinstance(response, dict):
        return "", "missing_response"
    status_code = response.get("status_code")
    if status_code != 200:
        return json.dumps(response, ensure_ascii=False), "request_error"
    body = response.get("body")
    if not isinstance(body, dict):
        return "", "missing_body"
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", "missing_choices"
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", "missing_choice_object"
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return "", "missing_message"
    content = message.get("content", "")
    if not isinstance(content, str):
        return "", "missing_content"
    return content, "ok"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_parsed_rows(
    output_dir: Path,
    parsed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_jsonl = output_dir / "metric_scores.jsonl"
    if not existing_jsonl.exists():
        return parsed_rows
    existing_rows = read_jsonl(existing_jsonl)
    merged_map = {row["custom_id"]: row for row in existing_rows}
    for row in parsed_rows:
        merged_map[row["custom_id"]] = row
    return [merged_map[custom_id] for custom_id in sorted(merged_map)]


def build_summary_rows(parsed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_map: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for parsed_row in parsed_rows:
        if not parsed_row.get("metric"):
            continue
        sample_key = (
            parsed_row["domain"],
            parsed_row["baseline"],
            parsed_row["audience"],
            parsed_row["index"],
            parsed_row["source_file"],
        )
        summary_row = summary_map.setdefault(
            sample_key,
            {
                "domain": parsed_row["domain"],
                "baseline": parsed_row["baseline"],
                "audience": parsed_row["audience"],
                "index": parsed_row["index"],
                "source_file": parsed_row["source_file"],
                "topic_text": parsed_row["topic_text"],
                "source_prompt_field": parsed_row["source_prompt_field"],
                "cognitive_load_score": "",
                "personalization_alignment_score": "",
                "reader_attitude_score": "",
                "overall_personalization_score": "",
                "summary_status": "ok",
                "batch_id": parsed_row["batch_id"],
            },
        )
        metric = parsed_row["metric"]
        metric_column = METRIC_SUMMARY_COLUMNS[metric]
        if parsed_row["parse_status"] == "ok" and parsed_row["score"] is not None:
            summary_row[metric_column] = parsed_row["score"]
        else:
            summary_row["summary_status"] = "incomplete"
        summary_row["batch_id"] = parsed_row["batch_id"]

    summary_rows: list[dict[str, Any]] = []
    for sample_key in sorted(summary_map):
        row = summary_map[sample_key]
        metric_values: list[Decimal] = []
        for metric in METRIC_ORDER:
            value = row[METRIC_SUMMARY_COLUMNS[metric]]
            if not value:
                row["summary_status"] = "incomplete"
                break
            metric_values.append(Decimal(value))
        if len(metric_values) == 3:
            average = sum(metric_values) / Decimal("3")
            row["overall_personalization_score"] = format(
                average.quantize(Decimal("0.0"), rounding=ROUND_HALF_UP),
                ".1f",
            )
        summary_rows.append(row)
    return summary_rows


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.results_file.exists():
        raise SystemExit(f"结果文件不存在: {args.results_file}")
    if not args.manifest_file.exists():
        raise SystemExit(f"Manifest 文件不存在: {args.manifest_file}")

    manifest_rows = read_jsonl(args.manifest_file)
    manifest_map = {row["custom_id"]: row for row in manifest_rows}
    result_rows = read_jsonl(args.results_file)
    if args.results_file.name.endswith(".output.jsonl"):
        batch_id = args.results_file.name.removesuffix(".output.jsonl")
    else:
        batch_id = args.results_file.stem

    parsed_rows: list[dict[str, Any]] = []

    progress = tqdm(
        result_rows,
        desc="解析 Batch 结果",
        unit="result",
        disable=should_disable_tqdm(),
    )
    for result_row in progress:
        custom_id = result_row.get("custom_id")
        manifest = manifest_map.get(custom_id)
        raw_output, response_status = extract_content(result_row)
        if manifest is None:
            score, parse_status = None, "missing_manifest"
        elif response_status == "ok":
            score, parse_status = normalize_score(raw_output)
        else:
            score, parse_status = None, response_status

        base_row = {
            "custom_id": custom_id,
            "metric": manifest.get("metric") if manifest else None,
            "domain": manifest.get("domain") if manifest else None,
            "baseline": manifest.get("baseline") if manifest else None,
            "audience": manifest.get("audience") if manifest else None,
            "index": manifest.get("index") if manifest else None,
            "source_file": manifest.get("source_file") if manifest else None,
            "topic_text": manifest.get("topic_text") if manifest else None,
            "source_prompt_field": manifest.get("source_prompt_field") if manifest else None,
            "weight_id": manifest.get("weight_id") if manifest else None,
            "score": score,
            "parse_status": parse_status,
            "raw_model_output": raw_output,
            "batch_id": batch_id,
        }
        parsed_rows.append(base_row)

    progress.close()

    if args.merge_existing:
        parsed_rows = merge_parsed_rows(args.output_dir, parsed_rows)
    summary_rows = build_summary_rows(parsed_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "metric_scores.jsonl"
    csv_path = args.output_dir / "metric_scores.csv"
    summary_path = args.output_dir / "sample_level_summary.csv"

    write_jsonl(jsonl_path, parsed_rows)
    write_csv(
        csv_path,
        parsed_rows,
        fieldnames=METRIC_SCORE_FIELDNAMES,
    )
    write_csv(
        summary_path,
        summary_rows,
        fieldnames=SUMMARY_FIELDNAMES,
    )

    print(f"已写入解析结果 JSONL: {jsonl_path}")
    print(f"已写入解析结果 CSV: {csv_path}")
    print(f"已写入样本汇总 CSV: {summary_path}")


if __name__ == "__main__":
    main()
