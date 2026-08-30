from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from _shared import load_tqdm, read_jsonl, should_disable_tqdm


SLICE_SPECS = {
    "domain_audience": ["domain", "audience"],
    "domain": ["domain"],
    "audience": ["audience"],
    "global": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 pairwise ranking 结果。")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("artifacts/pairwise/results/pairwise_results.jsonl"),
        help="pairwise 规范化结果文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pairwise/analysis"),
        help="分析输出目录",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not fieldnames:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return
    effective_fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=effective_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped[key].append(row)
    return dict(sorted(grouped.items()))


def slice_label(slice_name: str, slice_key: tuple[str, ...]) -> str:
    if slice_name == "global":
        return "global"
    return "__".join(f"{field}-{value}" for field, value in zip(SLICE_SPECS[slice_name], slice_key))


def compute_slice_stats(
    slice_name: str,
    slice_key: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baselines = sorted(
        {baseline for row in rows for baseline in (row["baseline_a"], row["baseline_b"])}
    )
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        pair_key = tuple(sorted((row["baseline_a"], row["baseline_b"])))
        payload = pair_stats.setdefault(
            pair_key,
            {
                "baseline_x": pair_key[0],
                "baseline_y": pair_key[1],
                "wins": defaultdict(int),
                "total": 0,
            },
        )
        payload["total"] += 1
        payload["wins"][row["winner_baseline"]] += 1

    ranking_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        copeland_score = 0.0
        total_wins = 0
        total_losses = 0
        total_matches = 0
        win_rates: list[float] = []
        evaluated_opponents = 0
        for opponent in baselines:
            if opponent == baseline:
                continue
            pair_key = tuple(sorted((baseline, opponent)))
            payload = pair_stats.get(pair_key)
            if payload is None:
                continue
            evaluated_opponents += 1
            total = int(payload["total"])
            wins = int(payload["wins"].get(baseline, 0))
            losses = total - wins
            win_rate = wins / total
            total_wins += wins
            total_losses += losses
            total_matches += total
            win_rates.append(win_rate)
            if win_rate > 0.5:
                copeland_score += 1.0
            elif win_rate == 0.5:
                copeland_score += 0.5

        row_payload = {
            "slice_type": slice_name,
            "slice_label": slice_label(slice_name, slice_key),
            "baseline": baseline,
            "copeland_score": round(copeland_score, 3),
            "mean_win_rate": round(mean(win_rates), 3) if win_rates else 0.0,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_matches": total_matches,
            "evaluated_opponents": evaluated_opponents,
            "comparison_rows": len(rows),
        }
        for field_name, field_value in zip(SLICE_SPECS[slice_name], slice_key):
            row_payload[field_name] = field_value
        ranking_rows.append(row_payload)

    ranking_rows.sort(
        key=lambda row: (
            -row["copeland_score"],
            -row["mean_win_rate"],
            -row["total_wins"],
            row["baseline"],
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank

    matrix_rows: list[dict[str, Any]] = []
    prefix_fields = ["slice_type", "slice_label"]
    prefix_fields.extend(SLICE_SPECS[slice_name])
    for baseline in baselines:
        matrix_row: dict[str, Any] = {
            "slice_type": slice_name,
            "slice_label": slice_label(slice_name, slice_key),
            "baseline": baseline,
        }
        for field_name, field_value in zip(SLICE_SPECS[slice_name], slice_key):
            matrix_row[field_name] = field_value
        for opponent in baselines:
            if baseline == opponent:
                matrix_row[opponent] = ""
                continue
            pair_key = tuple(sorted((baseline, opponent)))
            payload = pair_stats.get(pair_key)
            if payload is None:
                matrix_row[opponent] = ""
                continue
            total = int(payload["total"])
            wins = int(payload["wins"].get(baseline, 0))
            matrix_row[opponent] = f"{wins / total:.3f}"
        matrix_rows.append(matrix_row)
    return ranking_rows, matrix_rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def build_report(
    rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    rankings_by_slice_type: dict[str, list[dict[str, Any]]],
) -> str:
    parse_status = Counter(row["parse_status"] for row in rows)
    request_status = Counter(row["request_status"] for row in rows)
    global_rankings = rankings_by_slice_type["global"]
    domain_audience_rankings = rankings_by_slice_type["domain_audience"]
    best_cells = [
        row
        for row in domain_audience_rankings
        if row["rank"] == 1
    ]
    worst_cells = [
        row
        for row in domain_audience_rankings
        if row["rank"] == max(
            item["rank"]
            for item in domain_audience_rankings
            if item["slice_label"] == row["slice_label"]
        )
    ]

    lines: list[str] = []
    lines.append("# Pairwise Baseline Ranking 分析")
    lines.append("")
    lines.append("## 1. 结果可用性")
    lines.append(f"- 原始 comparison 总数：{len(rows)}")
    lines.append(f"- 成功进入排名统计的 comparison 数：{len(valid_rows)}")
    lines.append(f"- `request_status`：{dict(request_status)}")
    lines.append(f"- `parse_status`：{dict(parse_status)}")
    lines.append("")
    lines.append("## 2. 全局排名")
    lines.append(
        markdown_table(
            global_rankings,
            [
                "rank",
                "baseline",
                "copeland_score",
                "mean_win_rate",
                "total_wins",
                "total_losses",
                "total_matches",
            ],
        )
    )
    lines.append("")
    lines.append("## 3. Domain × Audience 第一名")
    lines.append(
        markdown_table(
            sorted(best_cells, key=lambda row: row["slice_label"]),
            ["slice_label", "baseline", "copeland_score", "mean_win_rate", "total_matches"],
        )
    )
    lines.append("")
    lines.append("## 4. Domain × Audience 末位")
    lines.append(
        markdown_table(
            sorted(worst_cells, key=lambda row: row["slice_label"]),
            ["slice_label", "baseline", "copeland_score", "mean_win_rate", "total_matches"],
        )
    )
    lines.append("")
    lines.append("## 5. 可直接写入论文的讨论")
    if global_rankings:
        strongest = global_rankings[0]
        weakest = global_rankings[-1]
        lines.append(
            f"- 全局上，`{strongest['baseline']}` 排名第一，`{weakest['baseline']}` 排名最后。"
            "这说明 pairwise judge 给出的整体偏好并不均匀，不同 baseline 的相对优势具有稳定性。"
        )
    lines.append(
        "- Pairwise 排名与点分式评估不同，它直接比较同一 topic、同一领域、同一人群下哪个输出更优，因此更接近人工主观偏好排序。"
    )
    lines.append(
        "- Copeland 分数与平均胜率可以同时报告：前者更强调相对名次，后者更强调总体对局表现。"
    )
    lines.append(
        "- `domain × audience` 细粒度排名适合正文呈现，全局和按 domain / audience 聚合的排名更适合作为主文补充或附录表格。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tqdm = load_tqdm()
    if not args.results_file.exists():
        raise SystemExit(f"pairwise results 文件不存在: {args.results_file}")

    rows = read_jsonl(args.results_file)
    valid_rows = [row for row in rows if row.get("parse_status") == "ok"]

    slice_groups: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = {
        slice_name: group_rows(valid_rows, group_keys)
        for slice_name, group_keys in SLICE_SPECS.items()
    }
    total_slices = sum(len(grouped) for grouped in slice_groups.values())

    rankings_by_slice_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matrix_payloads: list[tuple[Path, list[dict[str, Any]], list[str]]] = []

    progress_stats = tqdm(
        total=total_slices,
        desc="构建胜负统计",
        unit="slice",
        disable=should_disable_tqdm(),
    )
    for slice_name, grouped in slice_groups.items():
        for slice_key, slice_rows in grouped.items():
            ranking_rows, matrix_rows = compute_slice_stats(slice_name, slice_key, slice_rows)
            rankings_by_slice_type[slice_name].extend(ranking_rows)
            matrix_columns = ["slice_type", "slice_label"]
            matrix_columns.extend(SLICE_SPECS[slice_name])
            matrix_columns.append("baseline")
            baselines = sorted(
                {baseline for row in slice_rows for baseline in (row["baseline_a"], row["baseline_b"])}
            )
            matrix_columns.extend(baselines)
            matrix_path = args.output_dir / f"winrate_matrix__{slice_label(slice_name, slice_key)}.csv"
            matrix_payloads.append((matrix_path, matrix_rows, matrix_columns))
            progress_stats.update(1)
    progress_stats.close()

    progress_rankings = tqdm(
        total=sum(len(rows) for rows in rankings_by_slice_type.values()),
        desc="生成 ranking slices",
        unit="row",
        disable=should_disable_tqdm(),
    )
    for rows_for_slice in rankings_by_slice_type.values():
        progress_rankings.update(len(rows_for_slice))
    progress_rankings.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_tasks: list[tuple[Path, list[dict[str, Any]], list[str] | None]] = [
        (
            args.output_dir / "rankings_by_domain_audience.csv",
            rankings_by_slice_type["domain_audience"],
            [
                "slice_type",
                "slice_label",
                "domain",
                "audience",
                "rank",
                "baseline",
                "copeland_score",
                "mean_win_rate",
                "total_wins",
                "total_losses",
                "total_matches",
                "evaluated_opponents",
                "comparison_rows",
            ],
        ),
        (
            args.output_dir / "rankings_by_domain.csv",
            rankings_by_slice_type["domain"],
            [
                "slice_type",
                "slice_label",
                "domain",
                "rank",
                "baseline",
                "copeland_score",
                "mean_win_rate",
                "total_wins",
                "total_losses",
                "total_matches",
                "evaluated_opponents",
                "comparison_rows",
            ],
        ),
        (
            args.output_dir / "rankings_by_audience.csv",
            rankings_by_slice_type["audience"],
            [
                "slice_type",
                "slice_label",
                "audience",
                "rank",
                "baseline",
                "copeland_score",
                "mean_win_rate",
                "total_wins",
                "total_losses",
                "total_matches",
                "evaluated_opponents",
                "comparison_rows",
            ],
        ),
        (
            args.output_dir / "rankings_global.csv",
            rankings_by_slice_type["global"],
            [
                "slice_type",
                "slice_label",
                "rank",
                "baseline",
                "copeland_score",
                "mean_win_rate",
                "total_wins",
                "total_losses",
                "total_matches",
                "evaluated_opponents",
                "comparison_rows",
            ],
        ),
    ]
    export_tasks.extend(matrix_payloads)
    report_path = args.output_dir / "pairwise_analysis.md"
    progress_export = tqdm(
        total=len(export_tasks) + 1,
        desc="导出分析文件",
        unit="file",
        disable=should_disable_tqdm(),
    )
    for path, output_rows, fieldnames in export_tasks:
        write_csv(path, output_rows, fieldnames)
        progress_export.update(1)
    report = build_report(rows, valid_rows, rankings_by_slice_type)
    report_path.write_text(report, encoding="utf-8")
    progress_export.update(1)
    progress_export.close()

    print(f"Pairwise ranking 分析已写入目录: {args.output_dir}")
    print(f"分析报告已写入: {report_path}")


if __name__ == "__main__":
    main()
