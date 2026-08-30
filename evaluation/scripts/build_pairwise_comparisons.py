from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from _shared import (
    DatasetError,
    PairwiseComparisonRecord,
    SampleRecord,
    build_pairwise_comparison_id,
    collect_samples,
    deterministic_left_right,
    iter_baseline_pairs,
    load_tqdm,
    should_disable_tqdm,
    utc_now_iso,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 pairwise comparison manifest。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="评测数据目录，默认 data",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("artifacts/pairwise/manifests/pairwise_manifest.jsonl"),
        help="pairwise manifest 输出路径",
    )
    parser.add_argument(
        "--invalid-cells-file",
        type=Path,
        default=Path("artifacts/pairwise/manifests/invalid_cells.jsonl"),
        help="无效 comparison cell 输出路径",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("artifacts/pairwise/manifests/build_summary.json"),
        help="构建摘要输出路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="决定 A/B 展示顺序的固定随机种子，默认 42",
    )
    return parser.parse_args()


def group_samples_by_cell(samples: list[SampleRecord]) -> dict[tuple[str, str, int], list[SampleRecord]]:
    cell_map: dict[tuple[str, str, int], list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        cell_map[(sample.domain, sample.audience, sample.index)].append(sample)
    return dict(sorted(cell_map.items()))


def normalize_cell(
    cell_key: tuple[str, str, int],
    cell_samples: list[SampleRecord],
) -> tuple[list[SampleRecord] | None, dict[str, Any] | None]:
    domain, audience, index = cell_key
    baselines = [sample.baseline for sample in cell_samples]
    if len(set(baselines)) != len(baselines):
        duplicate_baselines = sorted(
            baseline
            for baseline in set(baselines)
            if baselines.count(baseline) > 1
        )
        return None, {
            "domain": domain,
            "audience": audience,
            "index": index,
            "reason": "duplicate_baseline_entries",
            "baselines": sorted(baselines),
            "duplicate_baselines": duplicate_baselines,
            "source_files": sorted(sample.source_file for sample in cell_samples),
        }

    topic_texts = {sample.topic_text for sample in cell_samples}
    if len(topic_texts) != 1:
        return None, {
            "domain": domain,
            "audience": audience,
            "index": index,
            "reason": "mismatched_topic_text",
            "baselines": sorted(baselines),
            "topic_texts": sorted(topic_texts),
            "source_files": sorted(sample.source_file for sample in cell_samples),
        }
    return sorted(cell_samples, key=lambda item: item.baseline), None


def build_record(
    sample_a: SampleRecord,
    sample_b: SampleRecord,
    seed: int,
) -> PairwiseComparisonRecord:
    comparison_id = build_pairwise_comparison_id(
        domain=sample_a.domain,
        audience=sample_a.audience,
        index=sample_a.index,
        baseline_a=sample_a.baseline,
        baseline_b=sample_b.baseline,
    )
    left_baseline, right_baseline = deterministic_left_right(
        seed=seed,
        comparison_id=comparison_id,
        baseline_a=sample_a.baseline,
        baseline_b=sample_b.baseline,
    )
    baseline_map = {
        sample_a.baseline: sample_a,
        sample_b.baseline: sample_b,
    }
    left_sample = baseline_map[left_baseline]
    right_sample = baseline_map[right_baseline]
    return PairwiseComparisonRecord(
        comparison_id=comparison_id,
        domain=sample_a.domain,
        audience=sample_a.audience,
        index=sample_a.index,
        topic_text=sample_a.topic_text,
        baseline_a=min(sample_a.baseline, sample_b.baseline),
        baseline_b=max(sample_a.baseline, sample_b.baseline),
        left_baseline=left_sample.baseline,
        right_baseline=right_sample.baseline,
        left_text=left_sample.output,
        right_text=right_sample.output,
        left_source_file=left_sample.source_file,
        right_source_file=right_sample.source_file,
    )


def record_to_row(record: PairwiseComparisonRecord, seed: int) -> dict[str, Any]:
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
        "left_text": record.left_text,
        "right_text": record.right_text,
        "left_source_file": record.left_source_file,
        "right_source_file": record.right_source_file,
        "seed": seed,
        "built_at": utc_now_iso(),
    }


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()

    samples = collect_samples(args.data_dir)
    grouped_samples: dict[tuple[str, str, int], list[SampleRecord]] = defaultdict(list)
    progress_collect = tqdm(
        samples,
        desc="收集样本",
        unit="sample",
        disable=should_disable_tqdm(),
    )
    for sample in progress_collect:
        grouped_samples[(sample.domain, sample.audience, sample.index)].append(sample)
    progress_collect.close()

    invalid_cells: list[dict[str, Any]] = []
    valid_cells: dict[tuple[str, str, int], list[SampleRecord]] = {}
    progress_cells = tqdm(
        sorted(grouped_samples.items()),
        desc="构建 comparison cells",
        unit="cell",
        disable=should_disable_tqdm(),
    )
    for cell_key, cell_samples in progress_cells:
        normalized, invalid_payload = normalize_cell(cell_key, cell_samples)
        if invalid_payload:
            invalid_cells.append(invalid_payload)
            continue
        if normalized is None:
            raise DatasetError(f"无法标准化 cell: {cell_key}")
        valid_cells[cell_key] = normalized
    progress_cells.close()

    pairwise_rows: list[dict[str, Any]] = []
    progress_pairs = tqdm(
        total=sum(len(iter_baseline_pairs([sample.baseline for sample in cell_samples])) for cell_samples in valid_cells.values()),
        desc="展开 pairwise comparisons",
        unit="pair",
        disable=should_disable_tqdm(),
    )
    for cell_samples in valid_cells.values():
        baseline_map = {sample.baseline: sample for sample in cell_samples}
        for baseline_a, baseline_b in iter_baseline_pairs(list(baseline_map.keys())):
            record = build_record(
                sample_a=baseline_map[baseline_a],
                sample_b=baseline_map[baseline_b],
                seed=args.seed,
            )
            pairwise_rows.append(record_to_row(record, seed=args.seed))
            progress_pairs.update(1)
    progress_pairs.close()

    write_jsonl(args.manifest_file, pairwise_rows)
    write_jsonl(args.invalid_cells_file, invalid_cells)
    write_json(
        args.summary_file,
        {
            "built_at": utc_now_iso(),
            "seed": args.seed,
            "sample_count": len(samples),
            "comparison_cell_count": len(grouped_samples),
            "valid_cell_count": len(valid_cells),
            "invalid_cell_count": len(invalid_cells),
            "pairwise_request_count": len(pairwise_rows),
            "baselines": sorted({sample.baseline for sample in samples}),
            "domains": sorted({sample.domain for sample in samples}),
            "audiences": sorted({sample.audience for sample in samples}),
        },
    )

    print(f"Pairwise manifest 已写入: {args.manifest_file}")
    print(f"Invalid cells 已写入: {args.invalid_cells_file}")
    print(f"构建摘要已写入: {args.summary_file}")
    print(
        f"共 {len(samples)} 条样本，{len(valid_cells)} 个有效 cell，"
        f"{len(invalid_cells)} 个无效 cell，{len(pairwise_rows)} 个 pairwise comparisons"
    )


if __name__ == "__main__":
    main()
