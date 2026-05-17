from __future__ import annotations

"""Helpers for notebook-level conclusions and human-readable reports."""

import re

import numpy as np
import pandas as pd


def _pick(df: pd.DataFrame, patterns, col: str):
    method_values = df["Метод"].astype(str)
    normalized_methods = (
        method_values.str.lower()
        .str.strip()
        .str.replace(r"[^a-zа-я0-9]+", "", regex=True)
    )

    for pattern in patterns:
        normalized_pattern = re.sub(r"[^a-zа-я0-9]+", "", str(pattern).lower())
        mask = normalized_methods.str.contains(normalized_pattern, regex=False, na=False)
        if mask.any():
            row = df.loc[mask, col]
            return row.iloc[0]
    return np.nan


def print_conclusion(results_summary: pd.DataFrame) -> None:
    required_cols = [
        "Метод",
        "HitRate@10 (все)",
        "NDCG@10 (все)",
        "HitRate@10 (холодные)",
        "NDCG@10 (холодные)",
    ]
    missing = [c for c in required_cols if c not in results_summary.columns]
    if missing:
        raise KeyError(f"results_summary missing required columns: {missing}")

    df = results_summary.copy()
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    baseline_mask = (
        df["Метод"].astype(str).str.contains("letitgo|let-it-go|бейзлайн|baseline", case=False, regex=True)
    )
    if baseline_mask.any():
        baseline_row = df.loc[baseline_mask].iloc[0]
    else:
        baseline_row = df.iloc[0]

    best_all = df.loc[df["HitRate@10 (все)"].idxmax()]
    best_cold = df.loc[df["HitRate@10 (холодные)"].idxmax()]

    print("\n\u2705 Основные результаты:")
    print(f"   \u2022 Бейзлайн метод: {baseline_row['Метод']}")
    print(f"   \u2022 Бейзлайн HitRate@10 (все): {baseline_row['HitRate@10 (все)']:.4f}")
    print(f"   \u2022 Бейзлайн HitRate@10 (холодные): {baseline_row['HitRate@10 (холодные)']:.4f}")
    print(f"   \u2022 Лучший метод по all: {best_all['Метод']} ({best_all['HitRate@10 (все)']:.4f})")
    print(f"   \u2022 Лучший метод по cold: {best_cold['Метод']} ({best_cold['HitRate@10 (холодные)']:.4f})")

    baseline_hit_all = float(baseline_row["HitRate@10 (все)"])
    baseline_hit_cold = float(baseline_row["HitRate@10 (холодные)"])
    delta_all = float(best_all["HitRate@10 (все)"] - baseline_hit_all)
    delta_cold = float(best_cold["HitRate@10 (холодные)"] - baseline_hit_cold)

    print("\n\u2605 Улучшения (best vs baseline):")
    print(f"   \u2022 Для всех товаров: {delta_all * 100:+.2f}%")
    print(f"   \u2022 Для холодных товаров: {delta_cold * 100:+.2f}%")

    top_cold = df.sort_values("HitRate@10 (холодные)", ascending=False).head(3)
    print("\nTop-3 методов по холодным товарам:")
    for _, row in top_cold.iterrows():
        print(
            f"   \u2022 {row['Метод']}: HitRate@10={row['HitRate@10 (холодные)']:.4f}, "
            f"NDCG@10={row['NDCG@10 (холодные)']:.4f}"
        )
