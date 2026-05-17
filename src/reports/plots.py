"""Helpers for rendering conclusion diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker


@dataclass(frozen=True)
class ConclusionPlotConfig:
    """Configuration for comparison bar charts in conclusion block."""

    filename: str = "results_conclusion_comparison.png"
    figsize: Tuple[float, float] = (16, 6)
    dpi: int = 220
    width: float = 0.28
    title_fontsize: int = 16
    label_fontsize: int = 12
    legend_fontsize: int = 11
    annotation_fontsize: int = 9
    xlabel_rotation: int = 35
    x_ticks: int = 6
    tick_margin_ratio: float = 0.20
    tick_min: float = 1e-4
    family_figsize: Tuple[float, float] = (16, 8)
    delta_figsize: Tuple[float, float] = (14, 6)


def _to_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric copy of a metric column; invalid values become NaN."""

    return pd.to_numeric(df[column], errors="coerce")


def _collect_metric_values(plot_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute y-limits for metric charts based on observed values."""

    y_columns = [
        "HitRate@10 (все)",
        "HitRate@10 (холодные)",
        "NDCG@10 (все)",
        "NDCG@10 (холодные)",
    ]

    values = plot_df[y_columns].to_numpy(dtype=float)
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))

    margin = max((vmax - vmin) * 0.2, 1e-4)
    y_min = max(vmin - margin, 0.0)
    y_max = min(vmax + margin, 1.0)

    if y_max - y_min < 1e-4:
        # tiny spread or constant series; keep visible bars without collapsing axis
        center = (y_min + y_max) / 2
        y_min = max(center - 5e-4, 0.0)
        y_max = min(center + 5e-4, 1.0)

    return y_min, y_max


def _annotate_bars(ax, bars, precision: int = 3, offset: int = 6) -> None:
    """Add value labels above bars if values are finite."""

    fmt = f"{{:.{precision}f}}"
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(fmt.format(height), xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)


def _best_numeric_row(df: pd.DataFrame, metric_col: str) -> pd.Series:
    valid = df.copy()
    valid[metric_col] = pd.to_numeric(valid[metric_col], errors="coerce")
    if valid[metric_col].notna().any():
        return valid.loc[valid[metric_col].idxmax()]
    return valid.iloc[0]


def _family_from_experiment_id(experiment_id: str) -> str:
    exp = str(experiment_id).upper()
    if exp in {"E0", "E00", "E11"}:
        return "Baseline"
    if exp in {"E1", "E2"}:
        return "Linear/MLP"
    if exp in {"E3", "E3S", "E4", "E5", "E6", "E7", "E8", "E10", "E12", "E13", "E14", "E15", "E16"}:
        return "PairwiseAlignment"
    if exp in {"E9"}:
        return "Ablation"
    return "Other"


def plot_conclusion_comparison(results_summary: pd.DataFrame, config: ConclusionPlotConfig | None = None,
                              output_path: str = "results_conclusion_comparison.png") -> tuple:
    """Build and render a readable bar chart with adaptive y-scale for conclusion metrics."""

    cfg = config or ConclusionPlotConfig()

    if results_summary is None:
        raise ValueError("results_summary is required to build conclusion comparison plot")

    plot_df = results_summary.copy()
    plot_df["HitRate@10 (все)"] = _to_numeric_column(plot_df, "HitRate@10 (все)")
    plot_df["NDCG@10 (все)"] = _to_numeric_column(plot_df, "NDCG@10 (все)")
    plot_df["HitRate@10 (холодные)"] = _to_numeric_column(plot_df, "HitRate@10 (холодные)")
    plot_df["NDCG@10 (холодные)"] = _to_numeric_column(plot_df, "NDCG@10 (холодные)")

    x = np.arange(len(plot_df))
    y_min, y_max = _collect_metric_values(plot_df)
    tick_step = max((y_max - y_min) / cfg.x_ticks, cfg.tick_min)

    fig, axes = plt.subplots(1, 2, figsize=cfg.figsize, dpi=cfg.dpi, constrained_layout=True)

    bars_hit_all = axes[0].bar(x - cfg.width / 2, plot_df["HitRate@10 (все)"], cfg.width,
                               label="HitRate@10 (все)", alpha=0.9)
    bars_hit_cold = axes[0].bar(x + cfg.width / 2, plot_df["HitRate@10 (холодные)"], cfg.width,
                                label="HitRate@10 (холодные)", alpha=0.8)
    axes[0].set_title("Сравнение HitRate", fontsize=16)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["Метод"], rotation=cfg.xlabel_rotation, ha="right", fontsize=cfg.label_fontsize)
    axes[0].set_ylim(y_min, y_max)
    axes[0].yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0].set_xlabel("Метод", fontsize=cfg.label_fontsize)
    axes[0].set_ylabel("Метрика", fontsize=cfg.label_fontsize)
    axes[0].legend(fontsize=cfg.legend_fontsize)
    axes[0].grid(True, alpha=0.3)

    bars_ndcg_all = axes[1].bar(x - cfg.width / 2, plot_df["NDCG@10 (все)"], cfg.width,
                                label="NDCG@10 (все)", alpha=0.9)
    bars_ndcg_cold = axes[1].bar(x + cfg.width / 2, plot_df["NDCG@10 (холодные)"], cfg.width,
                                 label="NDCG@10 (холодные)", alpha=0.8)
    axes[1].set_title("Сравнение NDCG", fontsize=16)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["Метод"], rotation=cfg.xlabel_rotation, ha="right", fontsize=cfg.label_fontsize)
    axes[1].set_ylim(y_min, y_max)
    axes[1].yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].set_xlabel("Метод", fontsize=cfg.label_fontsize)
    axes[1].set_ylabel("Метрика", fontsize=cfg.label_fontsize)
    axes[1].legend(fontsize=cfg.legend_fontsize)
    axes[1].grid(True, alpha=0.3)

    _annotate_bars(axes[0], bars_hit_all, precision=3)
    _annotate_bars(axes[0], bars_hit_cold, precision=3)
    _annotate_bars(axes[1], bars_ndcg_all, precision=3)
    _annotate_bars(axes[1], bars_ndcg_cold, precision=3)

    fig.suptitle("Сравнение метрик рекомендаций", fontsize=18, y=1.02)
    fig.tight_layout()

    out_path = Path(output_path)
    plt.savefig(str(out_path), dpi=cfg.dpi, bbox_inches="tight")
    plt.show()

    return fig, axes


def plot_family_panels(
    experiment_results: pd.DataFrame,
    metric_all: str = "NDCG@10 (все)",
    metric_cold: str = "NDCG@10 (холодные)",
    config: ConclusionPlotConfig | None = None,
    output_path: str = "results_family_panels.png",
):
    """Plot per-family best experiment bars for all/cold metrics."""
    if experiment_results is None or experiment_results.empty:
        raise ValueError("experiment_results is required for family panels.")

    cfg = config or ConclusionPlotConfig()
    df = experiment_results.copy()
    if "experiment_id" not in df.columns:
        raise KeyError("experiment_results must contain experiment_id column.")
    if "method_name" not in df.columns:
        raise KeyError("experiment_results must contain method_name column.")
    if metric_all not in df.columns or metric_cold not in df.columns:
        raise KeyError(f"Missing metric columns: {metric_all}, {metric_cold}")

    df["family"] = df["experiment_id"].astype(str).map(_family_from_experiment_id)
    rows = []
    for family, group in df.groupby("family"):
        best = _best_numeric_row(group, metric_cold)
        rows.append(
            {
                "Family": family,
                "Method": best["method_name"],
                metric_all: float(pd.to_numeric(best[metric_all], errors="coerce")),
                metric_cold: float(pd.to_numeric(best[metric_cold], errors="coerce")),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("Family").reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=cfg.family_figsize, dpi=cfg.dpi)
    bars_all = ax.bar(x - width / 2, plot_df[metric_all], width, label=metric_all, alpha=0.9)
    bars_cold = ax.bar(x + width / 2, plot_df[metric_cold], width, label=metric_cold, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Family"], fontsize=cfg.label_fontsize)
    ax.set_title("Лучшие методы по семействам", fontsize=cfg.title_fontsize)
    ax.set_ylabel("Метрика", fontsize=cfg.label_fontsize)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=cfg.legend_fontsize)
    _annotate_bars(ax, bars_all, precision=3)
    _annotate_bars(ax, bars_cold, precision=3)

    for i, method in enumerate(plot_df["Method"]):
        ax.annotate(
            str(method),
            xy=(i, 0),
            xytext=(0, -24),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            rotation=20,
        )

    plt.tight_layout()
    out_path = Path(output_path)
    plt.savefig(str(out_path), dpi=cfg.dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_delta_vs_baseline(
    experiment_results: pd.DataFrame,
    baseline_experiment_id: str = "E0",
    metric_col: str = "NDCG@10 (холодные)",
    config: ConclusionPlotConfig | None = None,
    output_path: str = "results_delta_vs_baseline.png",
):
    """Plot delta metric relative to baseline experiment."""
    if experiment_results is None or experiment_results.empty:
        raise ValueError("experiment_results is required for delta plot.")
    cfg = config or ConclusionPlotConfig()

    df = experiment_results.copy()
    if "experiment_id" not in df.columns or "method_name" not in df.columns:
        raise KeyError("experiment_results must contain experiment_id and method_name columns.")
    if metric_col not in df.columns:
        raise KeyError(f"Missing metric column: {metric_col}")

    df["experiment_id"] = df["experiment_id"].astype(str).str.upper()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    baseline_id = str(baseline_experiment_id).upper()
    baseline_df = df[df["experiment_id"] == baseline_id]
    if baseline_df.empty:
        raise ValueError(f"Baseline {baseline_experiment_id} not found.")
    baseline_val = float(baseline_df[metric_col].mean())

    grouped = (
        df.groupby(["experiment_id", "method_name"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "metric_mean"})
    )
    grouped = grouped[grouped["experiment_id"] != baseline_id].copy()
    grouped["delta"] = grouped["metric_mean"] - baseline_val
    grouped = grouped.sort_values("delta", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=cfg.delta_figsize, dpi=cfg.dpi)
    x = np.arange(len(grouped))
    colors = ["tab:green" if d >= 0 else "tab:red" for d in grouped["delta"]]
    bars = ax.bar(x, grouped["delta"], color=colors, alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title(f"Δ {metric_col} vs baseline ({baseline_id})", fontsize=cfg.title_fontsize)
    ax.set_ylabel("Delta", fontsize=cfg.label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["experiment_id"], fontsize=cfg.label_fontsize)
    ax.grid(True, axis="y", alpha=0.3)
    _annotate_bars(ax, bars, precision=4)

    for i, method in enumerate(grouped["method_name"]):
        ax.annotate(
            str(method),
            xy=(i, grouped.loc[i, "delta"]),
            xytext=(0, -18 if grouped.loc[i, "delta"] >= 0 else 12),
            textcoords="offset points",
            ha="center",
            va="top" if grouped.loc[i, "delta"] >= 0 else "bottom",
            fontsize=8,
            rotation=30,
        )

    plt.tight_layout()
    out_path = Path(output_path)
    plt.savefig(str(out_path), dpi=cfg.dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def format_significance_table(significance_df: pd.DataFrame) -> pd.DataFrame:
    """Return a prettified significance table sorted by effect size."""
    if significance_df is None or significance_df.empty:
        return pd.DataFrame()
    out = significance_df.copy()
    for col in ["baseline_mean", "experiment_mean", "delta", "p_value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_col = "delta" if "delta" in out.columns else out.columns[0]
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


__all__ = [
    "ConclusionPlotConfig",
    "plot_conclusion_comparison",
    "plot_family_panels",
    "plot_delta_vs_baseline",
    "format_significance_table",
]
