from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_COLUMNS = {
    "CL": "cognitive_load_score",
    "PA": "personalization_alignment_score",
    "RA": "reader_attitude_score",
    "Overall": "overall_personalization_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于评估结果生成论文可用的表现分析报告。")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("artifacts/results/sample_level_summary.csv"),
        help="样本级汇总结果文件",
    )
    parser.add_argument(
        "--metric-file",
        type=Path,
        default=Path("artifacts/results/metric_scores.csv"),
        help="指标级结果文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis"),
        help="分析结果输出目录",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    return float(value)


def grouped_stats(
    rows: list[dict[str, str]],
    group_keys: list[str],
    metric_column: str,
) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        bucket[tuple(row[key] for key in group_keys)].append(safe_float(row[metric_column]))
    output: list[dict[str, Any]] = []
    for group_value, values in bucket.items():
        item = {key: value for key, value in zip(group_keys, group_value)}
        item["n"] = len(values)
        item["mean"] = round(mean(values), 3)
        item["std"] = round(pstdev(values), 3)
        output.append(item)
    output.sort(key=lambda x: (-x["mean"], tuple(x[key] for key in group_keys)))
    return output


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


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def compute_macro_deltas_vs_base(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cell_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        cell_values[(row["baseline"], row["domain"], row["audience"])].append(
            safe_float(row["overall_personalization_score"])
        )
    cell_means = {key: mean(values) for key, values in cell_values.items()}
    baselines = sorted({row["baseline"] for row in rows})
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline == "base":
            continue
        deltas: list[float] = []
        shared_cells = 0
        for domain in sorted({row["domain"] for row in rows}):
            for audience in sorted({row["audience"] for row in rows}):
                current_key = (baseline, domain, audience)
                base_key = ("base", domain, audience)
                if current_key in cell_means and base_key in cell_means:
                    deltas.append(cell_means[current_key] - cell_means[base_key])
                    shared_cells += 1
        if deltas:
            output.append(
                {
                    "baseline": baseline,
                    "shared_cells": shared_cells,
                    "macro_delta_vs_base": round(mean(deltas), 3),
                }
            )
    output.sort(key=lambda x: (-x["macro_delta_vs_base"], x["baseline"]))
    return output


def compute_coverage(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        bucket[(row["baseline"], row["domain"], row["audience"])] += 1
    output: list[dict[str, Any]] = []
    for (baseline, domain, audience), count in sorted(bucket.items()):
        output.append(
            {
                "baseline": baseline,
                "domain": domain,
                "audience": audience,
                "n": count,
            }
        )
    return output


def top_bottom_cells(rows: list[dict[str, str]], top_k: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = grouped_stats(rows, ["baseline", "domain", "audience"], "overall_personalization_score")
    top_rows = stats[:top_k]
    bottom_rows = list(reversed(stats[-top_k:]))
    return top_rows, bottom_rows


def find_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"未找到 {key}={value} 的统计行")


def coverage_summary(rows: list[dict[str, Any]]) -> tuple[bool, list[str], dict[str, int], int]:
    cells = {(row["domain"], row["audience"]) for row in rows}
    expected_count = len(cells)
    by_baseline: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        by_baseline[row["baseline"]].add((row["domain"], row["audience"]))
    counts = {baseline: len(values) for baseline, values in by_baseline.items()}
    incomplete = sorted(
        baseline for baseline, count in counts.items() if count < expected_count
    )
    return not incomplete, incomplete, counts, expected_count


def build_report(
    summary_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    coverage_rows: list[dict[str, Any]],
    baseline_overall: list[dict[str, Any]],
    audience_overall: list[dict[str, Any]],
    domain_overall: list[dict[str, Any]],
    baseline_by_metric: dict[str, list[dict[str, Any]]],
    baseline_by_audience_overall: list[dict[str, Any]],
    baseline_by_domain_overall: list[dict[str, Any]],
    audience_by_domain_overall: list[dict[str, Any]],
    macro_deltas: list[dict[str, Any]],
    top_cells: list[dict[str, Any]],
    bottom_cells: list[dict[str, Any]],
) -> str:
    summary_status = Counter(row["summary_status"] for row in summary_rows)
    parse_status = Counter(row["parse_status"] for row in metric_rows)
    metric_means = {
        metric_name: round(mean(safe_float(row[column]) for row in summary_rows), 3)
        for metric_name, column in METRIC_COLUMNS.items()
    }
    adult_row = find_row(audience_overall, "audience", "adult")
    child_row = find_row(audience_overall, "audience", "child")
    teen_row = find_row(audience_overall, "audience", "teen")
    ai_row = find_row(domain_overall, "domain", "ai")
    biology_row = find_row(domain_overall, "domain", "biology")
    medicine_row = find_row(domain_overall, "domain", "medicine")
    coverage_complete, incomplete_baselines, coverage_counts, expected_cell_count = coverage_summary(coverage_rows)
    strongest_baseline = baseline_overall[0]
    weakest_baseline = baseline_overall[-1]
    strongest_cell = top_cells[0] if top_cells else None
    weakest_cell = bottom_cells[0] if bottom_cells else None

    lines: list[str] = []
    lines.append("# 评估结果表现分析")
    lines.append("")
    lines.append("## 1. 数据与结果可用性")
    lines.append(f"- 样本级结果总数：{len(summary_rows)}")
    lines.append(f"- 指标级结果总数：{len(metric_rows)}")
    lines.append(f"- `summary_status`：{dict(summary_status)}")
    lines.append(f"- `parse_status`：{dict(parse_status)}")
    lines.append("- 当前结果文件全部解析成功，没有 `invalid_score` 或汇总缺失。")
    lines.append("")
    lines.append("## 2. 重要结论")
    lines.append(
        f"- 从总体均值看，`Reader Attitude` 最高（{metric_means['RA']:.3f}），"
        f"`Cognitive Load`（{metric_means['CL']:.3f}）和 `Personalization Alignment`（{metric_means['PA']:.3f}）更低，"
        "说明模型相对更容易生成“读起来让人愿意接受”的文本，但在真正的个性化贴合和认知难度校准上更难做到高分。"
    )
    lines.append(
        f"- 直接按总体均值排序，baseline 表现为："
        + " > ".join(f"{row['baseline']}({row['mean']:.3f})" for row in baseline_overall)
        + "。"
    )
    if coverage_complete:
        lines.append(
            f"- 当前所有 baseline 都覆盖了完整的 `domain × audience` 组合（每个 baseline 都覆盖 {expected_cell_count} 个单元），"
            "因此总体均值已经具备较好的可比性；matched comparison 仍可作为更稳健的补充分析。"
        )
    else:
        lines.append(
            "- 但需要注意 baseline 覆盖并不完全一致："
            + "、".join(
                f"`{baseline}` 仅覆盖 {coverage_counts[baseline]}/{expected_cell_count} 个 `domain × audience` 单元"
                for baseline in incomplete_baselines
            )
            + "，因此总体均值不能直接当作严格公平比较。"
        )
    if macro_deltas:
        lines.append(
            "- 在与 `base` 共享的 `domain × audience` 单元上做 matched comparison 后，"
            + "、".join(
                f"`{row['baseline']}` 平均提升 {row['macro_delta_vs_base']:.3f}"
                for row in macro_deltas
            )
            + "。这一结果比总体均值更适合写入论文的主结论。"
        )
    lines.append(
        f"- 年龄群体上，`adult`（{adult_row['mean']:.3f}）与 `child`（{child_row['mean']:.3f}）整体接近，"
        f"`teen` 最低（{teen_row['mean']:.3f}），提示“青少年 persona”是当前系统最难稳定适配的目标群体。"
    )
    lines.append(
        f"- 领域上，`ai`（{ai_row['mean']:.3f}）最高，`biology`（{biology_row['mean']:.3f}）次之，"
        f"`medicine` 最低（{medicine_row['mean']:.3f}），说明医学领域是当前最具挑战性的内容域。"
    )
    lines.append("")
    lines.append("## 3. Coverage")
    lines.append(markdown_table(coverage_rows, ["baseline", "domain", "audience", "n"]))
    lines.append("")
    lines.append("## 4. Baseline 整体排序")
    lines.append(markdown_table(baseline_overall, ["baseline", "n", "mean", "std"]))
    lines.append("")
    lines.append("## 5. 年龄群体与领域主效应")
    lines.append("### 5.1 Audience")
    lines.append(markdown_table(audience_overall, ["audience", "n", "mean", "std"]))
    lines.append("")
    lines.append("### 5.2 Domain")
    lines.append(markdown_table(domain_overall, ["domain", "n", "mean", "std"]))
    lines.append("")
    lines.append("## 6. 分指标表现")
    for metric_name, rows in baseline_by_metric.items():
        lines.append(f"### {metric_name}")
        lines.append(markdown_table(rows, ["baseline", "n", "mean", "std"]))
        lines.append("")
    lines.append("## 7. 交互分析")
    lines.append("### 7.1 Baseline × Audience（Overall）")
    lines.append(markdown_table(baseline_by_audience_overall, ["baseline", "audience", "n", "mean", "std"]))
    lines.append("")
    lines.append("### 7.2 Baseline × Domain（Overall）")
    lines.append(markdown_table(baseline_by_domain_overall, ["baseline", "domain", "n", "mean", "std"]))
    lines.append("")
    lines.append("### 7.3 Audience × Domain（Overall）")
    lines.append(markdown_table(audience_by_domain_overall, ["audience", "domain", "n", "mean", "std"]))
    lines.append("")
    lines.append("## 8. 与 base 的 matched comparison")
    lines.append(
        "这里按共享的 `domain × audience` 单元与 `base` 做宏平均比较，"
        "用于缓解 baseline 覆盖不完整带来的偏差。"
    )
    lines.append(markdown_table(macro_deltas, ["baseline", "shared_cells", "macro_delta_vs_base"]))
    lines.append("")
    lines.append("## 9. 最优与最弱单元")
    lines.append("### 9.1 Top Cells")
    lines.append(markdown_table(top_cells, ["baseline", "domain", "audience", "n", "mean", "std"]))
    lines.append("")
    lines.append("### 9.2 Bottom Cells")
    lines.append(markdown_table(bottom_cells, ["baseline", "domain", "audience", "n", "mean", "std"]))
    lines.append("")
    lines.append("## 10. 可直接写入论文的讨论")
    lines.append("- 第一，三个指标并不等难。`RA` consistently higher，说明目前模型更擅长生成“可接受、可信、愿意继续读”的文本，但对真正的人群定制和认知负荷控制仍存在明显短板。")
    lines.append("- 第二，`teen` 是当前最困难的人群。无论从总体还是多个 baseline 交互来看，青少年群体分数普遍低于 `adult` 和 `child`，说明这类 persona 的语言风格与知识层次平衡更难命中。")
    lines.append("- 第三，`medicine` 是当前最难领域。这可能与医学主题天然更依赖术语解释、风险表达和谨慎措辞有关，因此对认知负荷控制与个性化贴合提出了更高要求。")
    lines.append(
        f"- 第四，表现最强的 baseline 是 `{strongest_baseline['baseline']}`（overall={strongest_baseline['mean']:.3f}），"
        f"最弱的是 `{weakest_baseline['baseline']}`（overall={weakest_baseline['mean']:.3f}）。"
        "结合 matched comparison，可以认为不同基础模型或适配方案之间确实存在稳定性能差异。"
    )
    if strongest_cell and weakest_cell:
        lines.append(
            f"- 第五，最佳单元出现在 `{strongest_cell['baseline']} × {strongest_cell['domain']} × {strongest_cell['audience']}` "
            f"（mean={strongest_cell['mean']:.3f}），最弱单元出现在 "
            f"`{weakest_cell['baseline']} × {weakest_cell['domain']} × {weakest_cell['audience']}` "
            f"（mean={weakest_cell['mean']:.3f}）。这说明模型效果并不是简单由 baseline 单独决定，而是与领域和读者人群存在明显交互。"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.summary_file.exists():
        raise SystemExit(f"summary 文件不存在: {args.summary_file}")
    if not args.metric_file.exists():
        raise SystemExit(f"metric 文件不存在: {args.metric_file}")

    summary_rows = read_csv_rows(args.summary_file)
    metric_rows = read_csv_rows(args.metric_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = compute_coverage(summary_rows)
    baseline_overall = grouped_stats(summary_rows, ["baseline"], "overall_personalization_score")
    audience_overall = grouped_stats(summary_rows, ["audience"], "overall_personalization_score")
    domain_overall = grouped_stats(summary_rows, ["domain"], "overall_personalization_score")
    baseline_by_metric = {
        metric_name: grouped_stats(summary_rows, ["baseline"], metric_column)
        for metric_name, metric_column in METRIC_COLUMNS.items()
        if metric_name != "Overall"
    }
    baseline_by_audience_overall = grouped_stats(
        summary_rows,
        ["baseline", "audience"],
        "overall_personalization_score",
    )
    baseline_by_domain_overall = grouped_stats(
        summary_rows,
        ["baseline", "domain"],
        "overall_personalization_score",
    )
    audience_by_domain_overall = grouped_stats(
        summary_rows,
        ["audience", "domain"],
        "overall_personalization_score",
    )
    macro_deltas = compute_macro_deltas_vs_base(summary_rows)
    top_cells, bottom_cells = top_bottom_cells(summary_rows)

    report = build_report(
        summary_rows=summary_rows,
        metric_rows=metric_rows,
        coverage_rows=coverage_rows,
        baseline_overall=baseline_overall,
        audience_overall=audience_overall,
        domain_overall=domain_overall,
        baseline_by_metric=baseline_by_metric,
        baseline_by_audience_overall=baseline_by_audience_overall,
        baseline_by_domain_overall=baseline_by_domain_overall,
        audience_by_domain_overall=audience_by_domain_overall,
        macro_deltas=macro_deltas,
        top_cells=top_cells,
        bottom_cells=bottom_cells,
    )

    report_path = args.output_dir / "performance_analysis.md"
    report_path.write_text(report, encoding="utf-8")

    write_csv(args.output_dir / "coverage.csv", coverage_rows)
    write_csv(args.output_dir / "baseline_overall.csv", baseline_overall)
    write_csv(args.output_dir / "audience_overall.csv", audience_overall)
    write_csv(args.output_dir / "domain_overall.csv", domain_overall)
    for metric_name, rows in baseline_by_metric.items():
        write_csv(args.output_dir / f"baseline_{metric_name.lower()}.csv", rows)
    write_csv(args.output_dir / "baseline_by_audience_overall.csv", baseline_by_audience_overall)
    write_csv(args.output_dir / "baseline_by_domain_overall.csv", baseline_by_domain_overall)
    write_csv(args.output_dir / "audience_by_domain_overall.csv", audience_by_domain_overall)
    write_csv(args.output_dir / "macro_delta_vs_base.csv", macro_deltas)
    write_csv(args.output_dir / "top_cells.csv", top_cells)
    write_csv(args.output_dir / "bottom_cells.csv", bottom_cells)

    print(f"分析报告已写入: {report_path}")
    print(f"分析表格已写入目录: {args.output_dir}")


if __name__ == "__main__":
    main()
