#!/usr/bin/env python3
"""Unified let-it-go paper comparison runner.

This script runs:
1. let-it-go paper-faithful `content_init`;
2. let-it-go paper-faithful `delta`;
3. pairwise models `E3/E3S/E14/E16`

on the same official let-it-go splits and the same paper evaluation mode
(`recommend_cold_items=True`, `filter_cold_history=False`).

The output is a single analysis-ready `experiment_results.csv` plus
`results_summary.csv` and a text log, so no manual post-merge step is needed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

from experiment.config import ExperimentConfig
from experiment.data import load_letitgo_official_splits
from .repro import (
    _build_local_modules,
    _parse_int_list,
    _resolve_device,
    _set_seed,
    _ensure_dir,
    train_single_seed,
    evaluate_all_modes,
)
from experiment.pairwise_grid import _build_parser as _build_pairwise_parser
from experiment.pairwise_grid import _configure as _configure_pairwise
from experiment.pairwise_grid import _load_experiment_overrides
from experiment.pairwise_grid import run_pipeline as run_pairwise_pipeline
from paths import PAPER_COMPARISON_ARTIFACTS_DIR
from experiment.pipeline import build_results_step, compute_significance_table


PAPER_RECOMMEND_COLD = True
PAPER_FILTER_COLD_HISTORY = False
DEFAULT_SEEDS = "42,221,451,934,1984"
DEFAULT_PAIRWISE_EXPERIMENTS = "E3,E3S,E14,E16"
DATASET_SPECS = {
    "zvuk": {
        "dataset_dir_name": "zvuk",
        "embedding_dim": 128,
        "max_length": 128,
        "expected_warm_items": 107448,
        "expected_cold_items": 23637,
        "title": "ZVUK PAPER COMPARISON",
    },
    "amazon_m2": {
        "dataset_dir_name": "amazon_m2",
        "embedding_dim": 64,
        "max_length": 64,
        "expected_warm_items": 42647,
        "expected_cold_items": 1402,
        "title": "AMAZON_M2 PAPER COMPARISON",
    },
    "yambda": {
        "dataset_dir_name": "yambda",
        "embedding_dim": 128,
        "max_length": 128,
        "expected_warm_items": 255305,
        "expected_cold_items": 81786,
        "title": "YAMBDA PAPER COMPARISON",
    },
}


def _parse_str_list(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-faithful let-it-go baselines and pairwise models in one run.")
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
        help="Directory where unified experiment_results.csv and plots-ready artifacts will be written. "
        "Defaults to artifacts/paper_comparison/<dataset>_paper_comparison.",
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated random seeds.")
    parser.add_argument(
        "--pairwise-experiments",
        default=DEFAULT_PAIRWISE_EXPERIMENTS,
        help="Comma-separated pairwise experiments to compare, e.g. E3,E3S,E14,E16",
    )
    parser.add_argument("--topk", default="10", help="Comma-separated top-k values. This runner expects 10.")
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto.")
    parser.add_argument("--pairwise-mode", choices=["quick", "full"], default="full")
    parser.add_argument("--pairwise-model-epochs", type=int, default=None)
    parser.add_argument("--pairwise-letitgo-epochs", type=int, default=None)
    parser.add_argument("--pairwise-mapper-epochs", type=int, default=None)
    parser.add_argument("--pairwise-min-warm-interactions", type=int, default=15)
    parser.add_argument(
        "--pairwise-target-mode",
        choices=["full", "delta_from_content"],
        default="full",
        help="Pairwise mapper target mode for compared experiments.",
    )
    parser.add_argument(
        "--pairwise-warm-sampler",
        choices=["all", "popular", "closest_to_cold", "mixed"],
        default="all",
        help="Warm-item sampler for pairwise mapper training.",
    )
    parser.add_argument("--pairwise-warm-sample-size", type=int, default=0)
    parser.add_argument(
        "--pairwise-warm-similarity",
        choices=["cosine", "dot"],
        default="cosine",
    )
    parser.add_argument("--pairwise-warm-mix-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pairwise-infer-scope",
        choices=["cold", "warm", "all"],
        default="cold",
        help="Which item groups receive mapper updates in pairwise runs.",
    )
    parser.add_argument(
        "--pairwise-item-role-mode",
        choices=["current", "strict_zero_vs_gt_k"],
        default="current",
        help="Optional warm/cold/discarded role mode for official splits.",
    )
    parser.add_argument("--pairwise-item-role-k", type=int, default=5)
    parser.add_argument("--paper-max-epochs", type=int, default=100)
    parser.add_argument("--paper-patience", type=int, default=5)
    parser.add_argument("--paper-learning-rate", type=float, default=1e-3)
    parser.add_argument("--paper-max-delta-norm", type=float, default=0.5)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--experiment-overrides-json",
        default="",
        help="Path to JSON mapping experiment_id -> override dict (e.g. {\"E16\": {\"pairwise_infer_scope\": \"all\"}}).",
    )
    return parser


def _dataset_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset key: {value}")
    return key


def _resolve_dataset_dir(dataset_key: str, letitgo_repo_path: Path, dataset_dir_value: str) -> Path:
    if str(dataset_dir_value).strip():
        return Path(dataset_dir_value).expanduser().resolve()
    return (letitgo_repo_path / "data" / DATASET_SPECS[dataset_key]["dataset_dir_name"]).resolve()


def _load_official_bundle(dataset_key: str, dataset_dir: Path, letitgo_repo_path: Path) -> dict[str, Any]:
    cfg = SimpleNamespace(
        letitgo_dataset_key=dataset_key,
        letitgo_repo_path=str(letitgo_repo_path),
        letitgo_processed_dir=str(dataset_dir / "processed"),
        letitgo_item_embeddings_dir=str(dataset_dir / "item_embeddings"),
    )
    return load_letitgo_official_splits(cfg)


def _dataset_stats_from_bundle(
    *,
    dataset_key: str,
    dataset_dir: Path,
    bundle: dict[str, Any],
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    warm_items = int(len(bundle["warm_items"]))
    cold_items = int(len(bundle["cold_items"]))
    return {
        "dataset_key": dataset_key,
        "dataset_dir": str(dataset_dir),
        "train_rows": int(len(bundle["train_df"])),
        "val_rows": int(len(bundle["val_df"])),
        "test_rows": int(len(bundle["test_inputs_df"])),
        "ground_truth_rows": int(len(bundle["ground_truth_df"])),
        "warm_items_pickle": warm_items,
        "cold_items_pickle": cold_items,
        "warm_items_embeddings": int(warm_embeddings.shape[0]),
        "cold_items_embeddings": int(cold_embeddings.shape[0]),
        "embedding_dim": int(warm_embeddings.shape[1]) if warm_embeddings.ndim == 2 else 0,
        "matches_readme_warm_items": int(spec["expected_warm_items"]) == 0 or warm_items == int(spec["expected_warm_items"]),
        "matches_readme_cold_items": int(spec["expected_cold_items"]) == 0 or cold_items == int(spec["expected_cold_items"]),
        "expected_warm_items": int(spec["expected_warm_items"]),
        "expected_cold_items": int(spec["expected_cold_items"]),
    }


def _paper_record_from_mode_row(
    *,
    experiment_id: str,
    method_name: str,
    source_model_key: str,
    source_model_label: str,
    mapper_type: str,
    mode_row: dict[str, Any],
    seed: int,
    runtime_sec: float,
    total_cold_examples: int,
) -> dict[str, Any]:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    return {
        "experiment_id": experiment_id,
        "method_name": method_name,
        "source_model_key": source_model_key,
        "source_model_label": source_model_label,
        "mapper_type": mapper_type,
        "runtime_sec": float(runtime_sec),
        "num_predicted_cold": 0,
        "num_updated_total": 0,
        "num_updated_warm": 0,
        "num_warm_for_mapper": np.nan,
        "pairwise_target_mode": np.nan,
        "pairwise_prediction_mode": np.nan,
        "pairwise_infer_scope": np.nan,
        "pairwise_warm_sampler": np.nan,
        "pairwise_warm_sample_size": np.nan,
        "pairwise_warm_similarity": np.nan,
        "pairwise_warm_mix_ratio": np.nan,
        "pairwise_item_role_mode": np.nan,
        "pairwise_item_role_k": np.nan,
        "HitRate@10 (все)": float(mode_row["HR@10"]),
        "NDCG@10 (все)": float(mode_row["NDCG@10"]),
        "HitRate@10 (холодные)": float(mode_row["cold_HR@10"]),
        "NDCG@10 (холодные)": float(mode_row["cold_NDCG@10"]),
        "seed": int(seed),
        "override_json": "{}",
        "TotalColdExamples": int(total_cold_examples),
        "paper_eval_recommend_cold_items": PAPER_RECOMMEND_COLD,
        "paper_eval_filter_cold_history": PAPER_FILTER_COLD_HISTORY,
        "postprocess_mode": np.nan,
        "embedding_update_mode": np.nan,
        "blend_alpha": np.nan,
        "training_kwargs_json": json.dumps(
            {"source": "paper.comparison", "model_kind": source_model_key},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "timestamp_utc": timestamp_utc,
    }


def _select_paper_mode_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if (
            bool(row["recommend_cold_items"]) == PAPER_RECOMMEND_COLD
            and bool(row["filter_cold_items"]) == PAPER_FILTER_COLD_HISTORY
        ):
            return row
    raise RuntimeError("Paper evaluation mode row was not found in baseline evaluation output.")


def _build_pairwise_cfg(args: argparse.Namespace, dataset_dir: Path, output_dir: Path, experiments: list[str], topk_values: list[int]) -> ExperimentConfig:
    parser = _build_pairwise_parser()
    pairwise_args = parser.parse_args([])
    pairwise_args.mode = str(args.pairwise_mode)
    pairwise_args.output_dir = str(output_dir / "_pairwise_intermediate")
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
    pairwise_args.pairwise_target_mode = str(args.pairwise_target_mode)
    pairwise_args.pairwise_warm_sampler = str(args.pairwise_warm_sampler)
    pairwise_args.pairwise_warm_sample_size = int(args.pairwise_warm_sample_size)
    pairwise_args.pairwise_warm_similarity = str(args.pairwise_warm_similarity)
    pairwise_args.pairwise_warm_mix_ratio = float(args.pairwise_warm_mix_ratio)
    pairwise_args.pairwise_infer_scope = str(args.pairwise_infer_scope)
    pairwise_args.pairwise_item_role_mode = str(args.pairwise_item_role_mode)
    pairwise_args.pairwise_item_role_k = int(args.pairwise_item_role_k)
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


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "experiment_id",
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


def _save_outputs(
    *,
    output_dir: Path,
    title: str,
    experiment_results: pd.DataFrame,
    results_summary: pd.DataFrame,
    significance_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    _ensure_dir(output_dir)
    experiment_results = experiment_results.loc[:, _ordered_columns(experiment_results)]
    experiment_results.to_csv(output_dir / "experiment_results.csv", index=False)
    experiment_results.to_json(output_dir / "experiment_results.json", orient="records", force_ascii=False, indent=2)
    results_summary.to_csv(output_dir / "results_summary.csv", index=False)
    results_summary.to_json(output_dir / "results_summary.json", orient="records", force_ascii=False, indent=2)
    if significance_df is not None and not significance_df.empty:
        significance_df.to_csv(output_dir / "experiment_significance.csv", index=False)
        significance_df.to_json(
            output_dir / "experiment_significance.json",
            orient="records",
            force_ascii=False,
            indent=2,
        )
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    log_lines = [
        "=" * 80,
        title,
        "=" * 80,
        f"Output dir: {output_dir}",
        f"Paper mode: recommend_cold={PAPER_RECOMMEND_COLD}, filter_cold_history={PAPER_FILTER_COLD_HISTORY}",
        "",
        "=== results_summary ===",
        results_summary.to_string(index=False),
    ]
    if significance_df is not None and not significance_df.empty:
        log_lines.extend(["", "=== experiment_significance ===", significance_df.to_string(index=False)])
    (output_dir / "combined_run.log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    dataset_key = _dataset_key(args.letitgo_dataset)
    dataset_spec = DATASET_SPECS[dataset_key]
    letitgo_repo_path = Path(args.letitgo_repo_path).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(dataset_key, letitgo_repo_path, args.dataset_dir)
    if str(args.output_dir).strip():
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (PAPER_COMPARISON_ARTIFACTS_DIR / f"{dataset_key}_paper_comparison").resolve()
    seeds = _parse_int_list(args.seeds)
    pairwise_experiments = [exp.upper() for exp in _parse_str_list(args.pairwise_experiments)]
    topk_values = _parse_int_list(args.topk)
    if topk_values != [10]:
        raise ValueError("This unified runner currently supports only --topk 10.")

    user_overrides = _load_experiment_overrides(args.experiment_overrides_json)

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
    all_runtime_rows: list[dict[str, Any]] = []
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
            model, train_info = train_single_seed(
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
                model=model,
                test_df=test_df,
                ground_truth_df=ground_truth_df,
                device=device,
                topk_values=topk_values,
            )
            selected_row = _select_paper_mode_row(eval_rows)
            record = _paper_record_from_mode_row(
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
            all_records.append(record)
            all_runtime_rows.append(
                {
                    "seed": int(seed),
                    "experiment_id": exp_id,
                    "runtime_sec": float(record["runtime_sec"]),
                    "best_epoch": int(train_info["best_epoch"]),
                    "best_val_ndcg": float(train_info["best_val_ndcg"]),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pairwise_cfg = _build_pairwise_cfg(
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            experiments=pairwise_experiments,
            topk_values=topk_values,
        )
        experiment_overrides = {}
        for exp_id in pairwise_experiments:
            override = {"random_state": int(seed)}
            if exp_id in user_overrides:
                override.update(user_overrides[exp_id])
            experiment_overrides[exp_id] = override
        _set_seed(seed)
        pairwise_state = run_pairwise_pipeline(pairwise_cfg, experiment_overrides=experiment_overrides)
        pairwise_df = pairwise_state["experiment_results"].copy()
        pairwise_df["seed"] = pairwise_df["seed"].fillna(seed).astype(int)
        all_records.extend(pairwise_df.to_dict(orient="records"))
        del pairwise_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    experiment_results = pd.DataFrame(all_records)
    experiment_results = experiment_results.loc[:, _ordered_columns(experiment_results)]

    summary_cfg = ExperimentConfig()
    summary_cfg.registry.topk_values = [10]
    summary_cfg.registry.primary_baseline_id = "E0"
    summary_state = {"experiment_results": experiment_results}
    summary_state = build_results_step(summary_state, config=summary_cfg)
    results_summary = summary_state["results_summary"]
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
        "experiment_overrides": {k: v for k, v in user_overrides.items()},
        "paper_mode_recommend_cold": PAPER_RECOMMEND_COLD,
        "paper_mode_filter_cold_history": PAPER_FILTER_COLD_HISTORY,
        "runtime_sec_total": float(time.perf_counter() - started),
        "dataset_stats": dataset_stats,
    }
    _save_outputs(
        output_dir=output_dir,
        title=str(dataset_spec["title"]),
        experiment_results=experiment_results,
        results_summary=results_summary,
        significance_df=significance_df,
        metadata=metadata,
    )

    runtime_df = pd.DataFrame(all_runtime_rows)
    if not runtime_df.empty:
        runtime_df.to_csv(output_dir / "paper_runtime_details.csv", index=False)

    print("=" * 80)
    print("UNIFIED PAPER COMPARISON READY")
    print("=" * 80)
    print(results_summary.to_string(index=False))
    if not significance_df.empty:
        print("\n=== experiment_significance ===")
        print(significance_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
