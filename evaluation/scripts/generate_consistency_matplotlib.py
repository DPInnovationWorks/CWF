from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "一致性"

BASELINE_ORDER = ["Base", "LoRA", "MoE", "Qwen2.5-14B", "CWF", "CWF-R"]

FALLBACK_DYNAMIC = {
    "Base": 2.965,
    "LoRA": 2.984,
    "MoE": 3.059,
    "Qwen2.5-14B": 3.209,
    "CWF": 3.451,
    "CWF-R": 3.302,
}

FALLBACK_EQUAL = {
    "Base": 2.276,
    "LoRA": 2.602,
    "MoE": 2.672,
    "Qwen2.5-14B": 2.863,
    "CWF": 3.119,
    "CWF-R": 2.914,
}

HUMAN_SCORES = {
    "Base": 3.510,
    "LoRA": 3.275,
    "MoE": 3.449,
    "Qwen2.5-14B": 3.705,
    "CWF": 3.794,
    "CWF-R": 3.736,
}

ONE_STEP_LLM_SCORES = {
    "Base": 2.183,
    "LoRA": 2.576,
    "MoE": 2.462,
    "Qwen2.5-14B": 2.763,
    "CWF": 3.011,
    "CWF-R": 2.828,
}

COLORS = {
    "Human Evaluation": "#7c3aed",
    "Dynamic Weights": "#0f766e",
    "Equal Weights": "#2563eb",
    "One-Step LLM Scoring": "#c2410c",
}


def ordered_values(series_map: dict[str, float]) -> list[float]:
    return [series_map[name] for name in BASELINE_ORDER]


def build_y_axis(values: list[float]) -> tuple[float, float, list[float]]:
    min_value = min(values)
    max_value = max(values)

    lower = min(2.0, math.floor((min_value - 0.06) * 10) / 10)
    upper = math.ceil((max_value + 0.08) * 10) / 10
    if upper - lower < 1.0:
        upper = lower + 1.0

    tick_count = int(round((upper - lower) / 0.2)) + 1
    ticks = [round(lower + 0.2 * idx, 1) for idx in range(tick_count)]
    return lower, upper, ticks


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    x = list(range(len(BASELINE_ORDER)))

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    ax.set_facecolor("white")

    series = [
        ("Human Evaluation", ordered_values(HUMAN_SCORES)),
        ("Dynamic Weights", ordered_values(FALLBACK_DYNAMIC)),
        ("Equal Weights", ordered_values(FALLBACK_EQUAL)),
        ("One-Step LLM Scoring", ordered_values(ONE_STEP_LLM_SCORES)),
    ]

    all_values = [value for _, values in series for value in values]
    y_min, y_max, y_ticks = build_y_axis(all_values)

    for label, values in series:
        ax.plot(
            x,
            values,
            marker="o",
            markersize=7.0,
            linewidth=3.0,
            color=COLORS[label],
            label=label,
        )

    ax.set_xlim(-0.25, len(BASELINE_ORDER) - 0.75)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.set_ylabel("Mean Score", fontsize=16, fontweight="bold", labelpad=14)

    ax.set_xticks(x)
    ax.set_xticklabels(BASELINE_ORDER, rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", color="#d7dbe3", linestyle=(0, (3, 6)), linewidth=1.0)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.06),
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#e5e7eb",
        fontsize=13,
        borderpad=0.65,
        labelspacing=0.9,
        handlelength=2.2,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    for suffix in ("png", "pdf", "svg"):
        output_path = ANALYSIS_DIR / f"consistency_line_chart_matplotlib.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(fig)


if __name__ == "__main__":
    main()
