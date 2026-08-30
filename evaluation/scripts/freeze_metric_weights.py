from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _shared import (
    METRIC_DEFINITIONS,
    METRIC_ORDER,
    SampleRecord,
    build_weight_id,
    build_weight_prompt,
    collect_samples,
    create_client_from_env,
    load_tqdm,
    load_weight_store,
    should_disable_tqdm,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成并冻结三项指标的权重文件。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据集目录，默认是 data",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/weights/metric_weights.v1.json"),
        help="冻结权重输出路径",
    )
    parser.add_argument(
        "--model",
        default="qwen-plus",
        help="用于 Stage 1 权重生成的模型名，默认 qwen-plus",
    )
    parser.add_argument(
        "--allow-extend",
        action="store_true",
        help="当权重文件已存在且出现新 topic 时，仅补齐缺失权重，不覆盖旧权重。",
    )
    return parser.parse_args()


def build_weight_targets(samples: list[SampleRecord]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for metric in METRIC_ORDER:
            weight_id = build_weight_id(metric, sample.domain, sample.audience, sample.topic_text)
            target = targets.setdefault(
                weight_id,
                {
                    "weight_id": weight_id,
                    "metric": metric,
                    "metric_name": METRIC_DEFINITIONS[metric]["name"],
                    "domain": sample.domain,
                    "audience": sample.audience,
                    "topic_text": sample.topic_text,
                    "topic_hash": weight_id.rsplit("::", 1)[-1],
                    "source_prompt_fields": set(),
                },
            )
            target["source_prompt_fields"].add(sample.source_prompt_field)
    return targets


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"模型输出不是合法 JSON: {raw_text!r}") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"模型输出 JSON 不是对象: {payload!r}")
    return payload


def normalize_weights(metric: str, payload: dict[str, Any]) -> dict[str, float]:
    expected_dimensions = list(METRIC_DEFINITIONS[metric]["dimensions"].keys())
    actual_keys = list(payload.keys())
    if set(actual_keys) != set(expected_dimensions):
        raise ValueError(
            f"权重键不匹配，期望 {expected_dimensions}，实际 {actual_keys}"
        )
    numeric_weights: dict[str, float] = {}
    total = 0.0
    for dimension in expected_dimensions:
        value = payload[dimension]
        if not isinstance(value, (int, float)):
            raise ValueError(f"维度 {dimension} 的权重不是数字: {value!r}")
        numeric_value = float(value)
        if numeric_value < 0:
            raise ValueError(f"维度 {dimension} 的权重不能为负数: {numeric_value}")
        numeric_weights[dimension] = numeric_value
        total += numeric_value
    if total <= 0:
        raise ValueError(f"权重总和必须大于 0，当前是 {total}")

    normalized: dict[str, float] = {}
    running_sum = 0.0
    for dimension in expected_dimensions[:-1]:
        scaled = round(numeric_weights[dimension] / total, 3)
        normalized[dimension] = scaled
        running_sum += scaled
    last_dimension = expected_dimensions[-1]
    normalized[last_dimension] = round(max(0.0, 1.0 - running_sum), 3)
    final_total = round(sum(normalized.values()), 3)
    if final_total != 1.0:
        normalized[last_dimension] = round(
            normalized[last_dimension] + (1.0 - final_total),
            3,
        )
    return normalized


def generate_weight_record(client: Any, model: str, target: dict[str, Any]) -> dict[str, Any]:
    metric = target["metric"]
    system_prompt, user_prompt = build_weight_prompt(
        metric=metric,
        domain=target["domain"],
        audience=target["audience"],
        topic_text=target["topic_text"],
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_output = response.choices[0].message.content or ""
    parsed = extract_json_object(raw_output)
    normalized_weights = normalize_weights(metric, parsed)
    return {
        "weight_id": target["weight_id"],
        "metric": metric,
        "metric_name": target["metric_name"],
        "domain": target["domain"],
        "audience": target["audience"],
        "topic_hash": target["topic_hash"],
        "topic_text": target["topic_text"],
        "source_prompt_fields": sorted(target["source_prompt_fields"]),
        "dimensions": list(METRIC_DEFINITIONS[metric]["dimensions"].keys()),
        "weights": normalized_weights,
        "raw_model_output": raw_output.strip(),
        "generated_at": utc_now_iso(),
        "model": model,
    }


def build_output_payload(existing_payload: dict[str, Any], weights: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "version": existing_payload.get("version", "v1"),
        "source_protocol": "指标拆解v2_en.pdf",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": model,
        "updated_at": utc_now_iso(),
        "weights": dict(sorted(weights.items())),
    }


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    samples = collect_samples(args.data_dir)
    targets = build_weight_targets(samples)

    existing_payload: dict[str, Any] = {"version": "v1", "weights": {}}
    existing_weights: dict[str, Any] = {}
    if args.output_file.exists():
        existing_payload = load_weight_store(args.output_file)
        existing_weights = dict(existing_payload["weights"])

    missing_ids = [weight_id for weight_id in sorted(targets) if weight_id not in existing_weights]
    if args.output_file.exists() and missing_ids and not args.allow_extend:
        raise SystemExit(
            f"权重文件 {args.output_file} 已存在，但发现 {len(missing_ids)} 个新权重键尚未冻结。"
            "如需显式补齐，请重新运行并加上 --allow-extend。"
        )

    if not missing_ids:
        print(f"权重文件已完整，无需新增。共 {len(existing_weights)} 个冻结权重。")
        return

    client = create_client_from_env()
    progress = tqdm(
        missing_ids,
        desc="冻结权重",
        unit="weight",
        disable=should_disable_tqdm(),
    )
    for weight_id in progress:
        target = targets[weight_id]
        progress.set_postfix(
            metric=target["metric"],
            domain=target["domain"],
            audience=target["audience"],
        )
        existing_weights[weight_id] = generate_weight_record(
            client=client,
            model=args.model,
            target=target,
        )

    write_json(
        args.output_file,
        build_output_payload(
            existing_payload=existing_payload,
            weights=existing_weights,
            model=args.model,
        ),
    )
    print(
        f"已写入冻结权重文件: {args.output_file} "
        f"(新增 {len(missing_ids)}，总计 {len(existing_weights)})"
    )


if __name__ == "__main__":
    main()
