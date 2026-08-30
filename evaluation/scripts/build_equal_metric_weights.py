from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _shared import (
    METRIC_DEFINITIONS,
    METRIC_ORDER,
    SampleRecord,
    build_weight_id,
    collect_samples,
    load_tqdm,
    should_disable_tqdm,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为三指标评估生成完全平均的维度权重文件。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据集目录，默认 data",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/weights/metric_weights.equal.json"),
        help="平均权重文件输出路径",
    )
    return parser.parse_args()


def normalize_equal_weights(dimensions: list[str]) -> dict[str, float]:
    base = round(1 / len(dimensions), 3)
    weights = {dimension: base for dimension in dimensions}
    diff = round(1.0 - sum(weights.values()), 3)
    weights[dimensions[-1]] = round(weights[dimensions[-1]] + diff, 3)
    return weights


def build_equal_weight_payload(samples: list[SampleRecord]) -> dict[str, Any]:
    weights: dict[str, Any] = {}
    progress = load_tqdm()(
        samples,
        desc="构建平均权重",
        unit="sample",
        disable=should_disable_tqdm(),
    )
    for sample in progress:
        for metric in METRIC_ORDER:
            weight_id = build_weight_id(metric, sample.domain, sample.audience, sample.topic_text)
            if weight_id in weights:
                existing_fields = set(weights[weight_id]["source_prompt_fields"])
                existing_fields.add(sample.source_prompt_field)
                weights[weight_id]["source_prompt_fields"] = sorted(existing_fields)
                continue
            dimensions = list(METRIC_DEFINITIONS[metric]["dimensions"].keys())
            equal_weights = normalize_equal_weights(dimensions)
            weights[weight_id] = {
                "weight_id": weight_id,
                "metric": metric,
                "metric_name": METRIC_DEFINITIONS[metric]["name"],
                "domain": sample.domain,
                "audience": sample.audience,
                "topic_hash": weight_id.rsplit("::", 1)[-1],
                "topic_text": sample.topic_text,
                "source_prompt_fields": [sample.source_prompt_field],
                "dimensions": dimensions,
                "weights": equal_weights,
                "raw_model_output": "equal_weight_baseline",
                "generated_at": utc_now_iso(),
                "model": "equal_weight_baseline",
            }
    progress.close()
    return {
        "version": "equal-v1",
        "source_protocol": "指标拆解v2_en.pdf",
        "weight_strategy": "equal_weight",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "equal_weight_baseline",
        "updated_at": utc_now_iso(),
        "weights": dict(sorted(weights.items())),
    }


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.data_dir)
    payload = build_equal_weight_payload(samples)
    write_json(args.output_file, payload)
    print(f"平均权重文件已写入: {args.output_file}")
    print(f"覆盖样本数: {len(samples)}")
    print(f"冻结权重键数: {len(payload['weights'])}")


if __name__ == "__main__":
    main()
