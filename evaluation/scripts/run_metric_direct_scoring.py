from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from _shared import (
    METRIC_ORDER,
    SampleRecord,
    build_custom_id,
    build_direct_chat_kwargs,
    build_score_prompts,
    build_weight_id,
    collect_samples,
    create_client_for_model,
    ensure_model_supports_direct,
    load_tqdm,
    load_weight_store,
    normalize_model_slug,
    read_jsonl,
    should_disable_tqdm,
    utc_now_iso,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多线程直接调用模型，复用动态权重完成三指标打分。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据集目录，默认是 data",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=Path("artifacts/weights/metric_weights.v1.json"),
        help="冻结权重文件路径",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="direct scoring manifest 输出路径；默认按模型写入 artifacts/model_experiments/<model>/manifests/",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="direct scoring raw output 路径；默认按模型写入 artifacts/model_experiments/<model>/direct_outputs/",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="直接调用的模型名，例如 MiniMax-M2.5 或 gpt-4o-mini",
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
        help="单条请求最大重试次数，默认 3",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="单次请求超时秒数，默认 120",
    )
    parser.add_argument(
        "--allow-thinking-models",
        action="store_true",
        help="允许运行官方标注为仅思考模式的模型，例如 MiniMax-M2.5。",
    )
    return parser.parse_args()


def validate_weight_availability(samples: list[SampleRecord], weights: dict[str, Any]) -> None:
    missing: list[str] = []
    for sample in samples:
        for metric in METRIC_ORDER:
            weight_id = build_weight_id(metric, sample.domain, sample.audience, sample.topic_text)
            if weight_id not in weights:
                missing.append(
                    f"{weight_id} (source_file={sample.source_file}, index={sample.index})"
                )
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(
            "冻结权重文件不完整，以下权重键不存在：\n"
            f"{preview}\n"
            "请先运行 freeze_metric_weights.py 生成完整权重。"
        )


def resolve_default_paths(model: str) -> tuple[Path, Path]:
    model_slug = normalize_model_slug(model)
    root = Path("artifacts/model_experiments") / model_slug
    manifest_file = root / "manifests" / "metric_manifest.jsonl"
    output_file = root / "direct_outputs" / f"{model_slug}.output.jsonl"
    return manifest_file, output_file


def build_requests(
    samples: list[SampleRecord],
    weights: dict[str, Any],
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for sample in samples:
        for metric in METRIC_ORDER:
            weight_id = build_weight_id(metric, sample.domain, sample.audience, sample.topic_text)
            weight_record = weights[weight_id]
            system_prompt, user_prompt = build_score_prompts(
                metric=metric,
                sample=sample,
                weights=weight_record["weights"],
            )
            custom_id = build_custom_id(metric, sample)
            request_rows.append(
                {
                    "custom_id": custom_id,
                    "metric": metric,
                    "sample": sample,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "metric": metric,
                    "domain": sample.domain,
                    "baseline": sample.baseline,
                    "audience": sample.audience,
                    "index": sample.index,
                    "source_file": sample.source_file,
                    "topic_text": sample.topic_text,
                    "source_prompt_field": sample.source_prompt_field,
                    "final_instruction": sample.final_instruction,
                    "weight_id": weight_id,
                    "status": sample.status,
                    "model": model,
                    "built_at": utc_now_iso(),
                }
            )
    return request_rows, manifest_rows


def load_existing_results(output_file: Path) -> dict[str, dict[str, Any]]:
    if not output_file.exists():
        return {}
    rows = read_jsonl(output_file)
    existing: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise SystemExit(f"结果文件存在缺少 custom_id 的记录: {output_file}")
        if custom_id in existing:
            raise SystemExit(f"结果文件里存在重复 custom_id: {custom_id}")
        existing[custom_id] = row
    return existing


def is_successful_result(row: dict[str, Any]) -> bool:
    response = row.get("response")
    if not isinstance(response, dict):
        return False
    return response.get("status_code") == 200


def prepare_existing_results(output_file: Path) -> dict[str, dict[str, Any]]:
    existing = load_existing_results(output_file)
    if not existing:
        return {}
    successful = {
        custom_id: row
        for custom_id, row in existing.items()
        if is_successful_result(row)
    }
    if len(successful) != len(existing):
        write_jsonl(output_file, [successful[custom_id] for custom_id in sorted(successful)])
    return successful


def validate_resume_compatibility(
    request_rows: list[dict[str, Any]],
    existing_results: dict[str, dict[str, Any]],
    model: str,
) -> None:
    request_ids = {row["custom_id"] for row in request_rows}
    for custom_id, existing in existing_results.items():
        if custom_id not in request_ids:
            raise SystemExit(
                f"现有结果中的 custom_id={custom_id} 不在当前请求集合中。"
                "这通常意味着数据集、权重或模型路径发生变化。请切换输出路径或清理旧结果后重跑。"
            )
        response = existing.get("response")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                existing_model = body.get("model")
                if existing_model and existing_model != model:
                    raise SystemExit(
                        f"现有结果 custom_id={custom_id} 来自模型 {existing_model}，"
                        f"与当前模型 {model} 不一致。请切换输出路径。"
                    )


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_one(
    client: Any,
    request_row: dict[str, Any],
    model: str,
    timeout_seconds: int,
    max_retries: int,
    allow_forced_thinking: bool,
) -> dict[str, Any]:
    custom_id = request_row["custom_id"]
    last_error = ""
    kwargs = build_direct_chat_kwargs(
        model=model,
        system_prompt=request_row["system_prompt"],
        user_prompt=request_row["user_prompt"],
        max_tokens=8,
        allow_forced_thinking=allow_forced_thinking,
    )
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            response = client.with_options(timeout=timeout_seconds).chat.completions.create(**kwargs)
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": getattr(response, "id", None),
                    "body": {
                        "model": getattr(response, "model", model),
                        "choices": [
                            {
                                "message": {
                                    "content": content.strip(),
                                }
                            }
                        ],
                    },
                },
                "provider_mode": "direct",
                "retry_count": attempt - 1,
                "latency_ms": latency_ms,
                "judged_at": utc_now_iso(),
            }
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == max_retries:
                return {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 500,
                        "body": {
                            "error": {
                                "message": last_error,
                            }
                        },
                    },
                    "provider_mode": "direct",
                    "retry_count": attempt,
                    "latency_ms": latency_ms,
                    "judged_at": utc_now_iso(),
                }
            time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"未预期的重试终止: {custom_id} / {last_error}")


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    ensure_model_supports_direct(
        args.model,
        allow_forced_thinking=args.allow_thinking_models,
    )

    default_manifest, default_output = resolve_default_paths(args.model)
    manifest_file = args.manifest_file or default_manifest
    output_file = args.output_file or default_output

    samples = collect_samples(args.data_dir)
    weight_payload = load_weight_store(args.weights_file)
    weights = weight_payload["weights"]
    validate_weight_availability(samples, weights)

    request_rows, manifest_rows = build_requests(samples, weights, args.model)
    write_jsonl(manifest_file, manifest_rows)

    existing_results = prepare_existing_results(output_file)
    validate_resume_compatibility(request_rows, existing_results, args.model)
    to_run = [row for row in request_rows if row["custom_id"] not in existing_results]
    skipped = len(request_rows) - len(to_run)

    client = create_client_for_model(args.model)
    progress = tqdm(
        total=len(request_rows),
        desc="直接调用三指标打分",
        unit="request",
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
            future = executor.submit(
                score_one,
                client,
                row,
                args.model,
                args.request_timeout,
                args.max_retries,
                args.allow_thinking_models,
            )
            futures[future] = row["custom_id"]
            return True

        initial = min(args.max_workers, len(to_run))
        for _ in range(initial):
            submit_next()

        while futures:
            completed, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in completed:
                futures.pop(future)
                row = future.result()
                append_jsonl_row(output_file, row)
                progress.update(1)
                if row.get("response", {}).get("status_code") == 200:
                    done += 1
                else:
                    failed += 1
                progress.set_postfix(done=done, skipped=skipped, failed=failed)
                submit_next()

    progress.close()
    print(
        f"Direct scoring 完成：总计 {len(request_rows)}，新增完成 {done}，跳过 {skipped}，失败 {failed}"
    )
    print(f"Manifest 已写入: {manifest_file}")
    print(f"Raw output 已写入: {output_file}")


if __name__ == "__main__":
    main()
