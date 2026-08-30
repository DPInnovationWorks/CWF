from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _shared import (
    METRIC_ORDER,
    SampleRecord,
    build_batch_request_body,
    build_custom_id,
    build_score_prompts,
    build_weight_id,
    collect_samples,
    ensure_model_supports_batch,
    load_tqdm,
    load_weight_store,
    should_disable_tqdm,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把数据集转换为 qwen-plus batch JSONL。")
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
        "--output-file",
        type=Path,
        default=Path("batch_example.jsonl"),
        help="batch JSONL 输出路径",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("artifacts/manifests/batch_manifest.jsonl"),
        help="custom_id 与样本元信息的映射文件路径",
    )
    parser.add_argument(
        "--model",
        default="qwen-plus",
        help="用于 Batch 的模型名，默认 qwen-plus",
    )
    parser.add_argument(
        "--baseline-filter",
        default="",
        help="只处理指定 baseline，多个值用逗号分隔，例如 revised 或 base,Lora",
    )
    return parser.parse_args()


def parse_baseline_filters(raw_value: str) -> set[str] | None:
    values = {item.strip() for item in raw_value.split(",") if item.strip()}
    return values or None


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


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    ensure_model_supports_batch(args.model)
    samples = collect_samples(
        args.data_dir,
        baseline_filters=parse_baseline_filters(args.baseline_filter),
    )
    weight_payload = load_weight_store(args.weights_file)
    weights = weight_payload["weights"]
    validate_weight_availability(samples, weights)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)

    total_requests = len(samples) * len(METRIC_ORDER)
    with args.output_file.open("w", encoding="utf-8") as batch_fp, args.manifest_file.open(
        "w",
        encoding="utf-8",
    ) as manifest_fp:
        progress = tqdm(
            total=total_requests,
            desc="构建 Batch JSONL",
            unit="request",
            disable=should_disable_tqdm(),
        )
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
                request_line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_batch_request_body(
                        model=args.model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    ),
                }
                manifest_line = {
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
                    "built_at": utc_now_iso(),
                }
                batch_fp.write(json.dumps(request_line, ensure_ascii=False) + "\n")
                manifest_fp.write(json.dumps(manifest_line, ensure_ascii=False) + "\n")
                progress.update(1)
        progress.close()

    print(f"Batch JSONL 已写入: {args.output_file}")
    print(f"Manifest 已写入: {args.manifest_file}")
    print(f"总请求数: {total_requests}")


if __name__ == "__main__":
    main()
