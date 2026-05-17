#!/usr/bin/env python3
"""Paper-comparable sweep runner for E0 baseline plus pairwise variants."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

from experiment.pairwise_grid import _build_parser as _build_pairwise_parser
from experiment.pairwise_grid import _configure as _configure_pairwise
from experiment.pairwise_grid import run_pipeline as run_pairwise_pipeline
from experiment.pipeline import compute_significance_table
from paths import PAPER_COMPARISON_ARTIFACTS_DIR

from .comparison import (
    DATASET_SPECS,
    PAPER_FILTER_COLD_HISTORY,
    PAPER_RECOMMEND_COLD,
    _dataset_key,
    _dataset_stats_from_bundle,
    _load_official_bundle,
    _paper_record_from_mode_row,
    _resolve_dataset_dir,
    _save_outputs,
    _select_paper_mode_row,
)
from .repro import (
    _build_local_modules,
    _ensure_dir,
    _parse_int_list,
    _resolve_device,
    _set_seed,
    evaluate_all_modes,
    train_single_seed,
)


DEFAULT_SEEDS = "42,221,451,934,1984"
DEFAULT_PAIRWISE_EXPERIMENTS = "E3,E16"
DEFAULT_VARIANT_WARM_SAMPLE_SIZE = 4096
DEFAULT_VARIANT_WARM_SIMILARITY = "cosine"
DEFAULT_VARIANT_WARM_MIX_RATIO = 0.5


@dataclass(frozen=True)
class VariantSpec:
    label: str
    title: str
    description: str
    pairwise_target_mode: str = "full"
    pairwise_warm_sampler: str = "all"
    pairwise_warm_sample_size: int | None = None
    pairwise_warm_similarity: str | None = None
    pairwise_warm_mix_ratio: float | None = None
    pairwise_infer_scope: str = "cold"
    pairwise_item_role_mode: str = "current"
    pairwise_item_role_k: int = 5
    per_experiment_overrides: tuple[tuple[str, str, str], ...] | None = None


def _variant_overrides_to_dict(spec: VariantSpec) -> dict[str, dict[str, object]]:
    if not spec.per_experiment_overrides:
        return {}
    result: dict[str, dict[str, object]] = {}
    for exp_id, key, value in spec.per_experiment_overrides:
        exp_id = str(exp_id).upper()
        if exp_id not in result:
            result[exp_id] = {}
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = value
        result[exp_id][key] = parsed
    return result


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        label="control_current",
        title="Current pairwise control",
        description="Default E16 setup on official let-it-go splits (cold only).",
    ),
    VariantSpec(
        label="delta_target_only",
        title="Delta target only",
        description="Only switch mapper target from full embedding to collaborative-content delta.",
        pairwise_target_mode="delta_from_content",
    ),
    VariantSpec(
        label="popular_sampler",
        title="Popular warm sampler",
        description="Train the mapper only on top-N popular warm items.",
        pairwise_warm_sampler="popular",
    ),
    VariantSpec(
        label="closest_sampler",
        title="Closest warm sampler",
        description="Train the mapper on warm items closest to cold items by content similarity.",
        pairwise_warm_sampler="closest_to_cold",
    ),
    VariantSpec(
        label="mixed_sampler",
        title="Mixed warm sampler",
        description="Mix popular warm items and content-nearest warm items for mapper training.",
        pairwise_warm_sampler="mixed",
    ),
    VariantSpec(
        label="infer_all",
        title="Warm+cold fusion",
        description="Apply mapper fusion to both warm and cold items while keeping current training target.",
        pairwise_infer_scope="all",
    ),
    VariantSpec(
        label="infer_all_low_warm_alpha",
        title="Warm+cold, low warm alpha",
        description="Apply to all items but use small blend_alpha for warm items to limit perturbation.",
        pairwise_infer_scope="all",
        per_experiment_overrides=(
            ("E16", "blend_alpha_warm", "0.05"),
        ),
    ),
    VariantSpec(
        label="infer_all_freq_weighted",
        title="Warm+cold, interaction-weighted alpha",
        description="Apply to all items with blend_alpha decaying inversely with interaction count for warm items.",
        pairwise_infer_scope="all",
        per_experiment_overrides=(
            ("E16", "blend_alpha_warm", "0.10"),
            ("E16", "blend_alpha_freq_decay_k", "10"),
        ),
    ),
    VariantSpec(
        label="closest_infer_all_low_warm",
        title="Closest sampler + all + low warm alpha",
        description="Combine content-nearest warm sampling with all-items scope and low warm alpha.",
        pairwise_warm_sampler="closest_to_cold",
        pairwise_infer_scope="all",
        per_experiment_overrides=(
            ("E16", "blend_alpha_warm", "0.05"),
        ),
    ),
    VariantSpec(
        label="mixed_infer_all_low_warm",
        title="Mixed sampler + all + low warm alpha",
        description="Combine mixed warm sampling with all-items scope and low warm alpha.",
        pairwise_warm_sampler="mixed",
        pairwise_infer_scope="all",
        per_experiment_overrides=(
            ("E16", "blend_alpha_warm", "0.05"),
        ),
    ),
    VariantSpec(
        label="delta_mixed_infer_all",
        title="Delta + mixed + warm+cold",
        description="Combine delta target, mixed warm sampler, and mapper updates for all eligible items.",
        pairwise_target_mode="delta_from_content",
        pairwise_warm_sampler="mixed",
        pairwise_infer_scope="all",
    ),
    VariantSpec(
        label="closest_delta_freq_weighted",
        title="Closest + delta + freq-weighted all",
        description="Best-of-all combo: closest sampler, delta target, freq-weighted alpha on all items.",
        pairwise_target_mode="delta_from_content",
        pairwise_warm_sampler="closest_to_cold",
        pairwise_infer_scope="all",
        per_experiment_overrides=(
            ("E16", "blend_alpha_warm", "0.10"),
            ("E16", "blend_alpha_freq_decay_k", "10"),
        ),
    ),
)


def _parse_str_list(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-faithful E0 baseline once and compare it with a preset sweep "
            "of E3/E16 variants on the same official let-it-go setup."
        )
    )
    parser.add_argument(
        "--letitgo-dataset",
        choices=sorted(DATASET_SPECS.keys()),
        default="zvuk",
        help="Official let-it-go dataset key.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Dataset root containing processed/ and item_embeddings/. Defaults to let-it-go/data/<dataset>.",
    )
    parser.add_argument(
        "--letitgo-repo-path",
        default=str(Path.home() / "let-it-go"),
        help="Path to local let-it-go repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory where the unified sweep table will be written. "
            "Defaults to artifacts/paper_comparison/<dataset>_comparable_sweep."
        ),
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated random seeds.")
    parser.add_argument(
        "--pairwise-experiments",
        default=DEFAULT_PAIRWISE_EXPERIMENTS,
        help="Comma-separated pairwise experiments to compare, e.g. E3,E16.",
    )
    parser.add_argument("--topk", default="10", help="Comma-separated top-k values. This runner expects 10.")
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto.")
    parser.add_argument("--pairwise-mode", choices=["quick", "full"], default="full")
    parser.add_argument("--pairwise-model-epochs", type=int, default=None)
    parser.add_argument("--pairwise-letitgo-epochs", type=int, default=None)
    parser.add_argument("--pairwise-mapper-epochs", type=int, default=None)
    parser.add_argument("--pairwise-min-warm-interactions", type=int, default=15)
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Optional comma-separated preset labels. "
            "Defaults to all built-in paper-comparable variants."
        ),
    )
    parser.add_argument(
        "--variant-warm-sample-size",
        type=int,
        default=DEFAULT_VARIANT_WARM_SAMPLE_SIZE,
        help="Warm sample size used by popular/closest/mixed presets unless overridden inside the preset.",
    )
    parser.add_argument(
        "--variant-warm-similarity",
        choices=["cosine", "dot"],
        default=DEFAULT_VARIANT_WARM_SIMILARITY,
        help="Similarity metric used by closest/mixed presets.",
    )
    parser.add_argument(
        "--variant-warm-mix-ratio",
        type=float,
        default=DEFAULT_VARIANT_WARM_MIX_RATIO,
        help="Popular-vs-nearest ratio for the mixed preset.",
    )
    parser.add_argument("--paper-max-epochs", type=int, default=100)
    parser.add_argument("--paper-patience", type=int, default=5)
    parser.add_argument("--paper-learning-rate", type=float, default=1e-3)
    parser.add_argument("--paper-max-delta-norm", type=float, default=0.5)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def _resolve_variants(args: argparse.Namespace) -> list[tuple[int, VariantSpec]]:
    available = {variant.label: variant for variant in DEFAULT_VARIANTS}
    if not str(args.variants).strip():
        selected_labels = [variant.label for variant in DEFAULT_VARIANTS]
    else:
        selected_labels = [label.strip() for label in str(args.variants).split(",") if label.strip()]
    unknown = [label for label in selected_labels if label not in available]
    if unknown:
        raise ValueError(
            "Unknown variant labels: "
            f"{unknown}. Available: {sorted(available.keys())}"
        )
    return [(index + 1, available[label]) for index, label in enumerate(selected_labels)]


def _resolve_variant_flags(args: argparse.Namespace, variant: VariantSpec) -> dict[str, Any]:
    sample_size = variant.pairwise_warm_sample_size
    if sample_size is None:
        sample_size = int(args.variant_warm_sample_size) if variant.pairwise_warm_sampler != "all" else 0
    if variant.pairwise_warm_sampler != "all" and int(sample_size) <= 0:
        raise ValueError(
            f"Variant '{variant.label}' requires a positive --variant-warm-sample-size, "
            f"got {sample_size}."
        )
    similarity = (
        variant.pairwise_warm_similarity
        if variant.pairwise_warm_similarity is not None
        else str(args.variant_warm_similarity)
    )
    mix_ratio = (
        variant.pairwise_warm_mix_ratio
        if variant.pairwise_warm_mix_ratio is not None
        else float(args.variant_warm_mix_ratio)
    )
    return {
        "pairwise_target_mode": str(variant.pairwise_target_mode),
        "pairwise_warm_sampler": str(variant.pairwise_warm_sampler),
        "pairwise_warm_sample_size": int(sample_size),
        "pairwise_warm_similarity": str(similarity),
        "pairwise_warm_mix_ratio": float(mix_ratio),
        "pairwise_infer_scope": str(variant.pairwise_infer_scope),
        "pairwise_item_role_mode": str(variant.pairwise_item_role_mode),
        "pairwise_item_role_k": int(variant.pairwise_item_role_k),
    }


def _build_pairwise_cfg(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    experiments: list[str],
    topk_values: list[int],
    flags: dict[str, Any],
    variant_label: str,
) -> Any:
    parser = _build_pairwise_parser()
    pairwise_args = parser.parse_args([])
    pairwise_args.mode = str(args.pairwise_mode)
    pairwise_args.output_dir = str(output_dir / "_pairwise_intermediate" / variant_label / f"seed_{seed}")
    pairwise_args.experiments = ",".join(experiments)
    pairwise_args.topk = ",".join(str(v) for v in topk_values)
    pairwise_args.use_letitgo_splits = True
    pairwise_args.letitgo_dataset = str(args.letitgo_dataset)
    pairwise_args.letitgo_repo_path = str(Path(args.letitgo_repo_path).expanduser())
    pairwise_args.letitgo_processed_dir = str(dataset_dir / "processed")
    pairwise_args.letitgo_embeddings_dir = str(dataset_dir / "item_embeddings")
    pairwise_args.paper_eval = True
    pairwise_args.paper_eval_recommend_cold = int(PAPER_RECOMMEND_COLD)
    pairwise_args.paper_eval_filter_cold_history = int(PAPER_FILTER_COLD_HISTORY)
    pairwise_args.paper_eval_report_all_modes = False
    pairwise_args.min_warm_interactions = int(args.pairwise_min_warm_interactions)
    pairwise_args.pairwise_target_mode = str(flags["pairwise_target_mode"])
    pairwise_args.pairwise_warm_sampler = str(flags["pairwise_warm_sampler"])
    pairwise_args.pairwise_warm_sample_size = int(flags["pairwise_warm_sample_size"])
    pairwise_args.pairwise_warm_similarity = str(flags["pairwise_warm_similarity"])
    pairwise_args.pairwise_warm_mix_ratio = float(flags["pairwise_warm_mix_ratio"])
    pairwise_args.pairwise_infer_scope = str(flags["pairwise_infer_scope"])
    pairwise_args.pairwise_item_role_mode = str(flags["pairwise_item_role_mode"])
    pairwise_args.pairwise_item_role_k = int(flags["pairwise_item_role_k"])
    if args.pairwise_model_epochs is not None:
        pairwise_args.model_epochs = int(args.pairwise_model_epochs)
    if args.pairwise_letitgo_epochs is not None:
        pairwise_args.letitgo_epochs = int(args.pairwise_letitgo_epochs)
    if args.pairwise_mapper_epochs is not None:
        pairwise_args.mapper_epochs = int(args.pairwise_mapper_epochs)
    cfg = _configure_pairwise(pairwise_args)
    cfg.registry.save_csv = False
    cfg.registry.save_json = False
    cfg.registry.run_significance = False
    cfg.registry.topk_values = list(topk_values)
    return cfg


def _annotate_paper_baseline_record(record: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(record)
    exp_id = str(record["experiment_id"]).upper()
    annotated["base_experiment_id"] = exp_id
    annotated["variant_label"] = "paper_baseline"
    if exp_id == "E00":
        annotated["variant_title"] = "Paper content initialization"
        annotated["variant_description"] = "Paper-faithful content-init baseline (no delta)."
    else:
        annotated["variant_title"] = "Paper delta baseline"
        annotated["variant_description"] = "Paper-faithful let-it-go delta method."
    annotated["variant_order"] = 0
    return annotated


def _annotate_variant_rows(
    *,
    df: pd.DataFrame,
    variant: VariantSpec,
    variant_order: int,
    flags: dict[str, Any],
) -> pd.DataFrame:
    annotated = df.copy()
    annotated["experiment_id"] = annotated["experiment_id"].astype(str).str.upper()
    annotated["base_experiment_id"] = annotated["experiment_id"]
    annotated["variant_label"] = str(variant.label)
    annotated["variant_title"] = str(variant.title)
    annotated["variant_description"] = str(variant.description)
    annotated["variant_order"] = int(variant_order)
    annotated["experiment_id"] = annotated["base_experiment_id"] + "__" + str(variant.label)
    annotated["method_name"] = (
        annotated["method_name"].astype(str) + " | " + str(variant.title)
    )
    for column, value in flags.items():
        annotated[column] = value
    return annotated


def _ordered_experiment_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "experiment_id",
        "base_experiment_id",
        "variant_label",
        "variant_title",
        "variant_description",
        "variant_order",
        "method_name",
        "source_model_key",
        "source_model_label",
        "mapper_type",
        "runtime_sec",
        "num_predicted_cold",
        "num_updated_total",
        "num_updated_warm",
        "num_warm_for_mapper",
        "pairwise_target_mode",
        "pairwise_prediction_mode",
        "pairwise_infer_scope",
        "pairwise_warm_sampler",
        "pairwise_warm_sample_size",
        "pairwise_warm_similarity",
        "pairwise_warm_mix_ratio",
        "pairwise_item_role_mode",
        "pairwise_item_role_k",
        "HitRate@10 (все)",
        "NDCG@10 (все)",
        "HitRate@10 (холодные)",
        "NDCG@10 (холодные)",
        "seed",
        "override_json",
        "TotalColdExamples",
        "paper_eval_recommend_cold_items",
        "paper_eval_filter_cold_history",
        "postprocess_mode",
        "embedding_update_mode",
        "blend_alpha",
        "training_kwargs_json",
        "timestamp_utc",
    ]
    return [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]


def _build_results_summary(experiment_results: pd.DataFrame, *, topk: int = 10) -> pd.DataFrame:
    hit_all_col = f"HitRate@{topk} (все)"
    ndcg_all_col = f"NDCG@{topk} (все)"
    hit_cold_col = f"HitRate@{topk} (холодные)"
    ndcg_cold_col = f"NDCG@{topk} (холодные)"

    required_cols = [hit_all_col, ndcg_all_col, hit_cold_col, ndcg_cold_col]
    missing = [col for col in required_cols if col not in experiment_results.columns]
    if missing:
        raise KeyError(f"experiment_results is missing metric columns: {missing}")

    summary_rows: list[dict[str, Any]] = []
    for experiment_id, group in experiment_results.groupby("experiment_id", sort=False):
        first = group.iloc[0]
        summary_rows.append(
            {
                "ExperimentID": str(experiment_id),
                "BaseExperimentID": str(first.get("base_experiment_id", experiment_id)),
                "VariantLabel": str(first.get("variant_label", "")),
                "VariantTitle": str(first.get("variant_title", "")),
                "Метод": str(first["method_name"]),
                "HitRate@10 (все)": float(pd.to_numeric(group[hit_all_col], errors="coerce").mean()),
                "NDCG@10 (все)": float(pd.to_numeric(group[ndcg_all_col], errors="coerce").mean()),
                "HitRate@10 (холодные)": float(pd.to_numeric(group[hit_cold_col], errors="coerce").mean()),
                "NDCG@10 (холодные)": float(pd.to_numeric(group[ndcg_cold_col], errors="coerce").mean()),
                "Runs": int(len(group)),
                "pairwise_target_mode": first.get("pairwise_target_mode", np.nan),
                "pairwise_prediction_mode": first.get("pairwise_prediction_mode", np.nan),
                "pairwise_infer_scope": first.get("pairwise_infer_scope", np.nan),
                "pairwise_warm_sampler": first.get("pairwise_warm_sampler", np.nan),
                "pairwise_warm_sample_size": first.get("pairwise_warm_sample_size", np.nan),
                "pairwise_warm_similarity": first.get("pairwise_warm_similarity", np.nan),
                "pairwise_warm_mix_ratio": first.get("pairwise_warm_mix_ratio", np.nan),
                "pairwise_item_role_mode": first.get("pairwise_item_role_mode", np.nan),
                "pairwise_item_role_k": first.get("pairwise_item_role_k", np.nan),
                "_exp_order": float(
                    pd.Series([str(first.get("base_experiment_id", experiment_id))])
                    .str.extract(r"E(\d+)", expand=False)
                    .astype(float)
                    .fillna(9999.0)
                    .iloc[0]
                ),
                "_variant_order": int(first.get("variant_order", 0)),
            }
        )

    results_summary = pd.DataFrame(summary_rows)
    if results_summary.empty:
        return results_summary
    return (
        results_summary.sort_values(
            ["_variant_order", "_exp_order", "BaseExperimentID", "ExperimentID"],
            kind="stable",
        )
        .drop(columns=["_exp_order", "_variant_order"])
        .reset_index(drop=True)
    )


def main() -> int:
    args = _build_parser().parse_args()
    dataset_key = _dataset_key(args.letitgo_dataset)
    dataset_spec = DATASET_SPECS[dataset_key]
    letitgo_repo_path = Path(args.letitgo_repo_path).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(dataset_key, letitgo_repo_path, args.dataset_dir)
    if str(args.output_dir).strip():
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (PAPER_COMPARISON_ARTIFACTS_DIR / f"{dataset_key}_comparable_sweep").resolve()
    seeds = _parse_int_list(args.seeds)
    pairwise_experiments = [exp.upper() for exp in _parse_str_list(args.pairwise_experiments)]
    forbidden = {"E0", "E00"}
    if forbidden.intersection(pairwise_experiments):
        raise ValueError(
            "This runner adds the paper baseline itself. "
            f"Do not include {sorted(forbidden)} in --pairwise-experiments: got {pairwise_experiments}"
        )
    topk_values = _parse_int_list(args.topk)
    if topk_values != [10]:
        raise ValueError("This comparable sweep runner currently supports only --topk 10.")

    variant_specs = _resolve_variants(args)
    variant_manifest_rows = []
    for variant_order, variant in variant_specs:
        flags = _resolve_variant_flags(args, variant)
        variant_manifest_rows.append(
            {
                "variant_order": int(variant_order),
                "variant_label": str(variant.label),
                "variant_title": str(variant.title),
                "variant_description": str(variant.description),
                **flags,
            }
        )
    variant_manifest = pd.DataFrame(variant_manifest_rows)

    device = _resolve_device(args.device)
    bundle = _load_official_bundle(dataset_key, dataset_dir, letitgo_repo_path)
    warm_embeddings = np.load(Path(bundle["item_embeddings_dir"]) / "embeddings_warm.npy")
    cold_embeddings = np.load(Path(bundle["item_embeddings_dir"]) / "embeddings_cold.npy")
    dataset_stats = _dataset_stats_from_bundle(
        dataset_key=dataset_key,
        dataset_dir=dataset_dir,
        bundle=bundle,
        warm_embeddings=warm_embeddings,
        cold_embeddings=cold_embeddings,
    )
    if not dataset_stats["matches_readme_warm_items"] or not dataset_stats["matches_readme_cold_items"]:
        raise RuntimeError(
            f"Dataset at {dataset_dir} does not match the paper {dataset_key} artifact: {json.dumps(dataset_stats)}"
        )

    modules = _build_local_modules()
    train_df = pl.from_pandas(bundle["train_df"])
    val_df = pl.from_pandas(bundle["val_df"])
    test_df = pl.from_pandas(bundle["test_inputs_df"])
    ground_truth_df = pl.from_pandas(bundle["ground_truth_df"])
    total_cold_examples = int(bundle["ground_truth_df"]["is_cold"].astype(bool).sum())

    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed in seeds:
        print("=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)
        _set_seed(seed)

        for paper_kind, exp_id, method_name, mapper_type in [
            ("content_init", "E00", "Content initialization (paper)", "content_init"),
            ("delta", "E0", "LetItGo baseline (paper)", "delta_finetune"),
        ]:
            paper_args = SimpleNamespace(
                model_kind=paper_kind,
                embedding_dim=int(dataset_spec["embedding_dim"]),
                num_blocks=2,
                num_heads=1,
                dropout=0.3,
                max_length=int(dataset_spec["max_length"]),
                max_epochs=int(args.paper_max_epochs),
                patience=int(args.paper_patience),
                learning_rate=float(args.paper_learning_rate),
                max_delta_norm=float(args.paper_max_delta_norm),
                train_batch_size=int(args.train_batch_size),
                eval_batch_size=int(args.eval_batch_size),
                num_workers=int(args.num_workers),
            )

            baseline_started = time.perf_counter()
            baseline_model, train_info = train_single_seed(
                args=paper_args,
                modules=modules,
                train_df=train_df,
                val_df=val_df,
                warm_embeddings=warm_embeddings,
                cold_embeddings=cold_embeddings,
                device=device,
                seed=seed,
                topk_values=topk_values,
            )
            eval_rows = evaluate_all_modes(
                args=paper_args,
                modules=modules,
                model=baseline_model,
                test_df=test_df,
                ground_truth_df=ground_truth_df,
                device=device,
                topk_values=topk_values,
            )
            selected_row = _select_paper_mode_row(eval_rows)
            baseline_record = _paper_record_from_mode_row(
                experiment_id=exp_id,
                method_name=method_name,
                source_model_key=paper_kind,
                source_model_label="let-it-go paper",
                mapper_type=mapper_type,
                mode_row=selected_row,
                seed=seed,
                runtime_sec=float(time.perf_counter() - baseline_started),
                total_cold_examples=total_cold_examples,
            )
            baseline_record["training_kwargs_json"] = json.dumps(
                {
                    "source": "paper.comparable_sweep",
                    "model_kind": paper_kind,
                    "best_epoch": int(train_info["best_epoch"]),
                    "best_val_ndcg": float(train_info["best_val_ndcg"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            all_records.append(_annotate_paper_baseline_record(baseline_record))
            del baseline_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for variant_order, variant in variant_specs:
            flags = _resolve_variant_flags(args, variant)
            print("-" * 80)
            print(f"SEED {seed} | VARIANT {variant.label}")
            print(json.dumps(flags, ensure_ascii=False, sort_keys=True))
            print("-" * 80)
            pairwise_cfg = _build_pairwise_cfg(
                args=args,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                seed=seed,
                experiments=pairwise_experiments,
                topk_values=topk_values,
                flags=flags,
                variant_label=variant.label,
            )
            variant_exp_overrides = _variant_overrides_to_dict(variant)
            experiment_overrides = {}
            for exp_id in pairwise_experiments:
                override = {"random_state": int(seed)}
                if exp_id in variant_exp_overrides:
                    override.update(variant_exp_overrides[exp_id])
                experiment_overrides[exp_id] = override
            _set_seed(seed)
            pairwise_state = run_pairwise_pipeline(
                pairwise_cfg,
                experiment_overrides=experiment_overrides,
            )
            pairwise_df = pairwise_state["experiment_results"].copy()
            if "seed" in pairwise_df.columns:
                pairwise_df["seed"] = pairwise_df["seed"].fillna(seed).astype(int)
            else:
                pairwise_df["seed"] = int(seed)
            pairwise_df = _annotate_variant_rows(
                df=pairwise_df,
                variant=variant,
                variant_order=variant_order,
                flags=flags,
            )
            all_records.extend(pairwise_df.to_dict(orient="records"))
            del pairwise_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    experiment_results = pd.DataFrame(all_records)
    experiment_results = experiment_results.loc[:, _ordered_experiment_columns(experiment_results)]
    results_summary = _build_results_summary(experiment_results, topk=10)
    significance_df = compute_significance_table(
        experiment_results,
        baseline_id="E0",
        metric_col="NDCG@10 (холодные)",
        test="wilcoxon",
        alpha=0.05,
    )

    metadata = {
        "dataset_key": dataset_key,
        "dataset_dir": str(dataset_dir),
        "letitgo_repo_path": str(letitgo_repo_path),
        "device": str(device),
        "seeds": seeds,
        "pairwise_experiments": pairwise_experiments,
        "variant_labels": [variant.label for _, variant in variant_specs],
        "paper_mode_recommend_cold": PAPER_RECOMMEND_COLD,
        "paper_mode_filter_cold_history": PAPER_FILTER_COLD_HISTORY,
        "runtime_sec_total": float(time.perf_counter() - started),
        "dataset_stats": dataset_stats,
        "variant_manifest": variant_manifest.to_dict(orient="records"),
    }

    _save_outputs(
        output_dir=output_dir,
        title=f"{dataset_spec['title']} | COMPARABLE SWEEP",
        experiment_results=experiment_results,
        results_summary=results_summary,
        significance_df=significance_df,
        metadata=metadata,
    )
    _ensure_dir(output_dir)
    variant_manifest.to_csv(output_dir / "variant_manifest.csv", index=False)
    variant_manifest.to_json(
        output_dir / "variant_manifest.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    runtime_df = experiment_results[
        [
            "seed",
            "experiment_id",
            "base_experiment_id",
            "variant_label",
            "variant_title",
            "runtime_sec",
        ]
    ].copy()
    runtime_df.to_csv(output_dir / "runtime_details.csv", index=False)

    print("=" * 80)
    print("COMPARABLE SWEEP READY")
    print("=" * 80)
    print(results_summary.to_string(index=False))
    if significance_df is not None and not significance_df.empty:
        print("\n=== experiment_significance ===")
        print(significance_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
