from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from _shared import (
    PairwiseComparisonRecord,
    build_pairwise_prompt,
    create_client_from_env,
    load_tqdm,
    read_jsonl,
    should_disable_tqdm,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多线程运行 qwen-plus 的 pairwise 判优。")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("artifacts/pairwise/manifests/pairwise_manifest.jsonl"),
        help="pairwise manifest 文件路径",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/pairwise/results/pairwise_raw_results.jsonl"),
        help="pairwise raw results 输出路径",
    )
    parser.add_argument(
        "--model",
        default="qwen-plus",
        help="直接调用的模型名，默认 qwen-plus",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="并发线程数，默认 6",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="单条 comparison 最大重试次数，默认 3",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="单次请求超时秒数，默认 120",
    )
    return parser.parse_args()


def row_to_record(row: dict[str, Any]) -> PairwiseComparisonRecord:
    return PairwiseComparisonRecord(
        comparison_id=row["comparison_id"],
        domain=row["domain"],
        audience=row["audience"],
        index=int(row["index"]),
        topic_text=row["topic_text"],
        baseline_a=row["baseline_a"],
        baseline_b=row["baseline_b"],
        left_baseline=row["left_baseline"],
        right_baseline=row["right_baseline"],
        left_text=row["left_text"],
        right_text=row["right_text"],
        left_source_file=row["left_source_file"],
        right_source_file=row["right_source_file"],
    )


def load_existing_results(output_file: Path) -> dict[str, dict[str, Any]]:
    if not output_file.exists():
        return {}
    rows = read_jsonl(output_file)
    existing: dict[str, dict[str, Any]] = {}
    for row in rows:
        comparison_id = row.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise SystemExit(f"结果文件存在缺少 comparison_id 的记录: {output_file}")
        if comparison_id in existing:
            raise SystemExit(f"结果文件里存在重复 comparison_id: {comparison_id}")
        existing[comparison_id] = row
    return existing


def validate_resume_compatibility(
    manifest_rows: list[dict[str, Any]],
    existing_results: dict[str, dict[str, Any]],
) -> None:
    manifest_map = {row["comparison_id"]: row for row in manifest_rows}
    for comparison_id, existing in existing_results.items():
        current = manifest_map.get(comparison_id)
        if current is None:
            continue
        for field in ("left_baseline", "right_baseline"):
            if existing.get(field) != current.get(field):
                raise SystemExit(
                    f"comparison_id={comparison_id} 的 {field} 与现有结果不一致，"
                    "这通常意味着 seed 或展示顺序发生变化。为避免污染结果，请清理旧结果后重跑。"
                )


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_one(
    client: Any,
    record: PairwiseComparisonRecord,
    model: str,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_pairwise_prompt(record)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            response = client.with_options(timeout=timeout_seconds).chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            return {
                "comparison_id": record.comparison_id,
                "domain": record.domain,
                "audience": record.audience,
                "index": record.index,
                "topic_text": record.topic_text,
                "baseline_a": record.baseline_a,
                "baseline_b": record.baseline_b,
                "left_baseline": record.left_baseline,
                "right_baseline": record.right_baseline,
                "left_source_file": record.left_source_file,
                "right_source_file": record.right_source_file,
                "request_status": "ok",
                "parse_status": "unparsed",
                "raw_model_output": content.strip(),
                "error_message": "",
                "retry_count": attempt - 1,
                "latency_ms": latency_ms,
                "model": model,
                "judged_at": utc_now_iso(),
            }
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == max_retries:
                return {
                    "comparison_id": record.comparison_id,
                    "domain": record.domain,
                    "audience": record.audience,
                    "index": record.index,
                    "topic_text": record.topic_text,
                    "baseline_a": record.baseline_a,
                    "baseline_b": record.baseline_b,
                    "left_baseline": record.left_baseline,
                    "right_baseline": record.right_baseline,
                    "left_source_file": record.left_source_file,
                    "right_source_file": record.right_source_file,
                    "request_status": "error",
                    "parse_status": "request_error",
                    "raw_model_output": "",
                    "error_message": last_error,
                    "retry_count": attempt,
                    "latency_ms": latency_ms,
                    "model": model,
                    "judged_at": utc_now_iso(),
                }
            time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"未预期的重试终止: {record.comparison_id} / {last_error}")


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.manifest_file.exists():
        raise SystemExit(f"manifest 文件不存在: {args.manifest_file}")

    manifest_rows = read_jsonl(args.manifest_file)
    existing_results = load_existing_results(args.output_file)
    validate_resume_compatibility(manifest_rows, existing_results)

    to_run = [row for row in manifest_rows if row["comparison_id"] not in existing_results]
    skipped = len(manifest_rows) - len(to_run)
    client = create_client_from_env()

    progress = tqdm(
        total=len(manifest_rows),
        desc="总 comparison 完成进度",
        unit="cmp",
        disable=should_disable_tqdm(),
    )
    if skipped:
        progress.update(skipped)
        progress.set_postfix(done=0, skipped=skipped, failed=0)

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures: dict[Future[dict[str, Any]], str] = {}
        pending_rows = iter(to_run)

        def submit_next() -> bool:
            try:
                row = next(pending_rows)
            except StopIteration:
                return False
            record = row_to_record(row)
            future = executor.submit(
                judge_one,
                client,
                record,
                args.model,
                args.request_timeout,
                args.max_retries,
            )
            futures[future] = record.comparison_id
            return True

        initial = min(args.max_workers, len(to_run))
        for _ in range(initial):
            submit_next()

        while futures:
            completed, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in completed:
                comparison_id = futures.pop(future)
                row = future.result()
                append_jsonl_row(args.output_file, row)
                progress.update(1)
                if row["request_status"] == "ok":
                    done += 1
                else:
                    failed += 1
                progress.set_postfix(done=done, skipped=skipped, failed=failed)
                submit_next()

    progress.close()
    print(
        f"Pairwise judging 完成：总计 {len(manifest_rows)}，新增完成 {done}，跳过 {skipped}，失败 {failed}"
    )
    print(f"Raw results 已写入: {args.output_file}")


if __name__ == "__main__":
    main()
