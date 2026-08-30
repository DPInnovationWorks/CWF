from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_COLUMNS = [
    "cognitive_load_score",
    "personalization_alignment_score",
    "reader_attitude_score",
    "overall_personalization_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总不同 judge model 的三指标结果，生成模型对比分析。")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts/model_experiments"),
        help="按模型分目录保存结果的根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_comparison_analysis"),
        help="模型对比分析输出目录",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str) -> float:
    return float(value)


def discover_summary_files(results_root: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    if not results_root.exists():
        return discovered
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_file = model_dir / "results" / "sample_level_summary.csv"
        if summary_file.exists():
            discovered.append((model_dir.name, summary_file))
    return discovered


def aggregate_rows(
    rows: list[dict[str, str]],
    group_keys: list[str],
    metric_column: str,
) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("summary_status") != "ok":
            continue
        bucket[tuple(row[key] for key in group_keys)].append(safe_float(row[metric_column]))
    output: list[dict[str, Any]] = []
    for group_value, values in bucket.items():
        item = {key: value for key, value in zip(group_keys, group_value)}
        item["n"] = len(values)
        item["mean"] = round(mean(values), 3)
        output.append(item)
    output.sort(key=lambda x: (-x["mean"], tuple(x[key] for key in group_keys)))
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def build_baseline_model_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = sorted({row["baseline"] for row in rows})
    models = sorted({row["judge_model"] for row in rows})
    row_map = {(row["baseline"], row["judge_model"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        matrix_row: dict[str, Any] = {"baseline": baseline}
        for model in models:
            match = row_map.get((baseline, model))
            matrix_row[model] = match["mean"] if match else ""
        output.append(matrix_row)
    return output


def build_per_baseline_rankings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket[row["baseline"]].append(row)
    output: dict[str, list[dict[str, Any]]] = {}
    for baseline, values in bucket.items():
        ranked = sorted(values, key=lambda x: (-x["mean"], x["judge_model"]))
        for index, item in enumerate(ranked, start=1):
            item["rank_within_baseline"] = index
        output[baseline] = ranked
    return dict(sorted(output.items()))


def main() -> None:
    args = parse_args()
    summary_files = discover_summary_files(args.results_root)
    if not summary_files:
        raise SystemExit(f"在 {args.results_root} 下没有发现任何 sample_level_summary.csv")

    combined_rows: list[dict[str, str]] = []
    model_counts: list[dict[str, Any]] = []
    for model_name, summary_file in summary_files:
        rows = read_csv_rows(summary_file)
        valid_rows = [row for row in rows if row.get("summary_status") == "ok"]
        model_counts.append(
            {
                "model": model_name,
                "summary_file": str(summary_file),
                "valid_samples": len(valid_rows),
                "all_samples": len(rows),
            }
        )
        for row in valid_rows:
            enriched = dict(row)
            enriched["judge_model"] = model_name
            combined_rows.append(enriched)

    overall = aggregate_rows(combined_rows, ["judge_model"], "overall_personalization_score")
    cl_rows = aggregate_rows(combined_rows, ["judge_model"], "cognitive_load_score")
    pa_rows = aggregate_rows(combined_rows, ["judge_model"], "personalization_alignment_score")
    ra_rows = aggregate_rows(combined_rows, ["judge_model"], "reader_attitude_score")
    by_domain = aggregate_rows(combined_rows, ["judge_model", "domain"], "overall_personalization_score")
    by_audience = aggregate_rows(combined_rows, ["judge_model", "audience"], "overall_personalization_score")
    by_baseline = aggregate_rows(combined_rows, ["judge_model", "baseline"], "overall_personalization_score")
    baseline_overall = aggregate_rows(combined_rows, ["baseline"], "overall_personalization_score")
    by_model_baseline_domain = sorted(
        aggregate_rows(
            combined_rows,
            ["judge_model", "baseline", "domain"],
            "overall_personalization_score",
        ),
        key=lambda x: (x["baseline"], x["domain"], -x["mean"], x["judge_model"]),
    )
    by_model_baseline_audience = sorted(
        aggregate_rows(
            combined_rows,
            ["judge_model", "baseline", "audience"],
            "overall_personalization_score",
        ),
        key=lambda x: (x["baseline"], x["audience"], -x["mean"], x["judge_model"]),
    )
    baseline_model_matrix = build_baseline_model_matrix(by_baseline)
    per_baseline_rankings = build_per_baseline_rankings(by_baseline)

    metric_table_rows: list[dict[str, Any]] = []
    cl_map = {row["judge_model"]: row["mean"] for row in cl_rows}
    pa_map = {row["judge_model"]: row["mean"] for row in pa_rows}
    ra_map = {row["judge_model"]: row["mean"] for row in ra_rows}
    overall_map = {row["judge_model"]: row["mean"] for row in overall}
    sample_map = {row["judge_model"]: row["n"] for row in overall}
    for model_name in sorted(overall_map, key=lambda name: (-overall_map[name], name)):
        metric_table_rows.append(
            {
                "judge_model": model_name,
                "n": sample_map[model_name],
                "cognitive_load_score": cl_map.get(model_name, ""),
                "personalization_alignment_score": pa_map.get(model_name, ""),
                "reader_attitude_score": ra_map.get(model_name, ""),
                "overall_personalization_score": overall_map.get(model_name, ""),
            }
        )

    report_lines: list[str] = []
    report_lines.append("# Judge Model Comparison")
    report_lines.append("")
    report_lines.append("## 1. Result Coverage")
    report_lines.append(markdown_table(model_counts, ["model", "valid_samples", "all_samples", "summary_file"]))
    report_lines.append("")
    report_lines.append("## 2. Overall Ranking")
    report_lines.append(markdown_table(overall, ["judge_model", "n", "mean"]))
    report_lines.append("")
    report_lines.append("## 3. Three-Metric Comparison")
    report_lines.append(
        markdown_table(
            metric_table_rows,
            [
                "judge_model",
                "n",
                "cognitive_load_score",
                "personalization_alignment_score",
                "reader_attitude_score",
                "overall_personalization_score",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## 4. Domain Breakdown")
    report_lines.append(markdown_table(by_domain, ["judge_model", "domain", "n", "mean"]))
    report_lines.append("")
    report_lines.append("## 5. Audience Breakdown")
    report_lines.append(markdown_table(by_audience, ["judge_model", "audience", "n", "mean"]))
    report_lines.append("")
    report_lines.append("## 6. Baseline Overall")
    report_lines.append(markdown_table(baseline_overall, ["baseline", "n", "mean"]))
    report_lines.append("")
    report_lines.append("## 7. Judge Model × Baseline")
    report_lines.append(markdown_table(by_baseline, ["judge_model", "baseline", "n", "mean"]))
    report_lines.append("")
    report_lines.append("## 8. Baseline Matrix")
    if baseline_model_matrix:
        matrix_columns = list(baseline_model_matrix[0].keys())
        report_lines.append(markdown_table(baseline_model_matrix, matrix_columns))
    else:
        report_lines.append("_No data_")
    report_lines.append("")
    report_lines.append("## 9. Per-Baseline Judge Model Ranking")
    for baseline, rows in per_baseline_rankings.items():
        report_lines.append(f"### {baseline}")
        report_lines.append(markdown_table(rows, ["rank_within_baseline", "judge_model", "n", "mean"]))
        report_lines.append("")
    report_lines.append("## 10. Judge Model × Baseline × Domain")
    report_lines.append(
        markdown_table(
            by_model_baseline_domain,
            ["judge_model", "baseline", "domain", "n", "mean"],
        )
    )
    report_lines.append("")
    report_lines.append("## 11. Judge Model × Baseline × Audience")
    report_lines.append(
        markdown_table(
            by_model_baseline_audience,
            ["judge_model", "baseline", "audience", "n", "mean"],
        )
    )
    report_lines.append("")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "model_coverage.csv", model_counts)
    write_csv(args.output_dir / "model_overall.csv", overall)
    write_csv(args.output_dir / "model_metric_table.csv", metric_table_rows)
    write_csv(args.output_dir / "model_by_domain.csv", by_domain)
    write_csv(args.output_dir / "model_by_audience.csv", by_audience)
    write_csv(args.output_dir / "baseline_overall.csv", baseline_overall)
    write_csv(args.output_dir / "model_by_baseline.csv", by_baseline)
    write_csv(args.output_dir / "model_by_baseline_domain.csv", by_model_baseline_domain)
    write_csv(args.output_dir / "model_by_baseline_audience.csv", by_model_baseline_audience)
    write_csv(args.output_dir / "baseline_model_matrix.csv", baseline_model_matrix)
    per_baseline_rows: list[dict[str, Any]] = []
    for baseline, rows in per_baseline_rankings.items():
        for row in rows:
            per_baseline_rows.append(
                {
                    "baseline": baseline,
                    "rank_within_baseline": row["rank_within_baseline"],
                    "judge_model": row["judge_model"],
                    "n": row["n"],
                    "mean": row["mean"],
                }
            )
    write_csv(args.output_dir / "baseline_model_rankings.csv", per_baseline_rows)
    print(f"模型对比报告已写入: {args.output_dir / 'model_comparison.md'}")
    print(f"模型对比表格已写入目录: {args.output_dir}")


if __name__ == "__main__":
    main()
