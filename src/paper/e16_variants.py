#!/usr/bin/env python3
"""Single-pass E16 variant sweep against E00/E0 paper baselines.

For each seed the pipeline runs ONCE: data loading + SASRec training happen
once, then all E16_* variant experiments run their mappers sequentially
inside a single ``run_experiment_grid`` call.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

from experiment.config import ExperimentConfig
from experiment.pairwise_grid import _build_parser as _build_pairwise_parser
from experiment.pairwise_grid import _configure as _configure_pairwise
from experiment.pairwise_grid import run_pipeline as run_pairwise_pipeline
from experiment.pipeline import build_results_step, compute_significance_table
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

VARIANT_REGISTRY: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("E16_CLEAR", {}),
    ("E16_DELTA", {
        "pairwise_target_mode": "delta_from_content",
    }),
    ("E16_POPULAR", {
        "pairwise_warm_sampler": "popular",
        "pairwise_warm_sample_size": 4096,
    }),
    ("E16_CLOSEST", {
        "pairwise_warm_sampler": "closest_to_cold",
        "pairwise_warm_sample_size": 4096,
    }),
    ("E16_MIXED", {
        "pairwise_warm_sampler": "mixed",
        "pairwise_warm_sample_size": 4096,
    }),
    ("E16_ALL", {
        "pairwise_infer_scope": "all",
    }),
    ("E16_ALL_LOW", {
        "pairwise_infer_scope": "all",
        "blend_alpha_warm": 0.05,
    }),
    ("E16_ALL_FREQ", {
        "pairwise_infer_scope": "all",
        "blend_alpha_warm": 0.10,
        "blend_alpha_freq_decay_k": 10,
    }),
    ("E16_CLOSEST_ALL_LOW", {
        "pairwise_warm_sampler": "closest_to_cold",
        "pairwise_warm_sample_size": 4096,
        "pairwise_infer_scope": "all",
        "blend_alpha_warm": 0.05,
    }),
    ("E16_MIXED_ALL_LOW", {
        "pairwise_warm_sampler": "mixed",
        "pairwise_warm_sample_size": 4096,
        "pairwise_infer_scope": "all",
        "blend_alpha_warm": 0.05,
    }),
    ("E16_DELTA_MIXED_ALL", {
        "pairwise_target_mode": "delta_from_content",
        "pairwise_warm_sampler": "mixed",
        "pairwise_warm_sample_size": 4096,
        "pairwise_infer_scope": "all",
    }),
    ("E16_CLOSEST_DELTA_FREQ", {
        "pairwise_target_mode": "delta_from_content",
        "pairwise_warm_sampler": "closest_to_cold",
        "pairwise_warm_sample_size": 4096,
        "pairwise_infer_scope": "all",
        "blend_alpha_warm": 0.10,
        "blend_alpha_freq_decay_k": 10,
    }),
])


def _parse_str_list(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-pass E16 variant sweep with E00/E0 paper baselines.",
    )
    parser.add_argument(
        "--letitgo-dataset",
        choices=sorted(DATASET_SPECS.keys()),
        default="zvuk",
    )
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument(
        "--letitgo-repo-path",
        default=str(Path.home() / "let-it-go"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant IDs (e.g. E16_CLEAR,E16_ALL_FREQ). Empty = all.",
    )
    parser.add_argument("--topk", default="10")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pairwise-mode", choices=["quick", "full"], default="full")
    parser.add_argument("--pairwise-model-epochs", type=int, default=None)
    parser.add_argument("--pairwise-letitgo-epochs", type=int, default=None)
    parser.add_argument("--pairwise-mapper-epochs", type=int, default=None)
    parser.add_argument("--pairwise-min-warm-interactions", type=int, default=15)
    parser.add_argument("--paper-max-epochs", type=int, default=100)
    parser.add_argument("--paper-patience", type=int, default=5)
    parser.add_argument("--paper-learning-rate", type=float, default=1e-3)
    parser.add_argument("--paper-max-delta-norm", type=float, default=0.5)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def _resolve_variants(raw: str) -> list[str]:
    if not raw.strip():
        return list(VARIANT_REGISTRY.keys())
    selected = [v.strip().upper() for v in raw.split(",") if v.strip()]
    unknown = [v for v in selected if v not in VARIANT_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown E16 variants: {unknown}. "
            f"Available: {list(VARIANT_REGISTRY.keys())}"
        )
    return selected


def _build_pairwise_cfg(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    experiments: list[str],
    topk_values: list[int],
) -> Any:
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
    pairwise_args.pairwise_target_mode = "full"
    pairwise_args.pairwise_warm_sampler = "all"
    pairwise_args.pairwise_warm_sample_size = 0
    pairwise_args.pairwise_warm_similarity = "cosine"
    pairwise_args.pairwise_warm_mix_ratio = 0.5
    pairwise_args.pairwise_infer_scope = "cold"
    pairwise_args.pairwise_item_role_mode = "current"
    pairwise_args.pairwise_item_role_k = 5
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
    experiment_results.to_csv(output_dir / "experiment_results.csv", index=False)
    experiment_results.to_json(
        output_dir / "experiment_results.json",
        orient="records", force_ascii=False, indent=2,
    )
    results_summary.to_csv(output_dir / "results_summary.csv", index=False)
    results_summary.to_json(
        output_dir / "results_summary.json",
        orient="records", force_ascii=False, indent=2,
    )
    if significance_df is not None and not significance_df.empty:
        significance_df.to_csv(output_dir / "experiment_significance.csv", index=False)
        significance_df.to_json(
            output_dir / "experiment_significance.json",
            orient="records", force_ascii=False, indent=2,
        )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    log_lines = [
        "=" * 80,
        title,
        "=" * 80,
        f"Output dir: {output_dir}",
        "",
        "=== results_summary ===",
        results_summary.to_string(index=False),
    ]
    if significance_df is not None and not significance_df.empty:
        log_lines.extend([
            "",
            "=== experiment_significance ===",
            significance_df.to_string(index=False),
        ])
    (output_dir / "combined_run.log.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args()
    dataset_key = _dataset_key(args.letitgo_dataset)
    dataset_spec = DATASET_SPECS[dataset_key]
    letitgo_repo_path = Path(args.letitgo_repo_path).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(dataset_key, letitgo_repo_path, args.dataset_dir)
    if str(args.output_dir).strip():
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (PAPER_COMPARISON_ARTIFACTS_DIR / f"{dataset_key}_e16_variants").resolve()
    seeds = _parse_int_list(args.seeds)
    variant_ids = _resolve_variants(args.variants)
    topk_values = _parse_int_list(args.topk)
    if topk_values != [10]:
        raise ValueError("This runner currently supports only --topk 10.")

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
            f"Dataset at {dataset_dir} does not match the paper {dataset_key} artifact: "
            f"{json.dumps(dataset_stats)}"
        )

    modules = _build_local_modules()
    train_df = pl.from_pandas(bundle["train_df"])
    val_df = pl.from_pandas(bundle["val_df"])
    test_df = pl.from_pandas(bundle["test_inputs_df"])
    ground_truth_df = pl.from_pandas(bundle["ground_truth_df"])
    total_cold_examples = int(bundle["ground_truth_df"]["is_cold"].astype(bool).sum())

    print("=" * 80)
    print(f"E16 VARIANT SWEEP | {dataset_key}")
    print(f"Seeds: {seeds}")
    print(f"Variants ({len(variant_ids)}): {variant_ids}")
    print(f"Output: {output_dir}")
    print("=" * 80)

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
            print(
                f"  {exp_id} ({paper_kind}): "
                f"NDCG@10 all={record['NDCG@10 (все)']:.4f}, "
                f"cold={record['NDCG@10 (холодные)']:.4f}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        experiment_overrides: dict[str, dict[str, Any]] = {}
        for vid in variant_ids:
            override = {"random_state": int(seed)}
            override.update(VARIANT_REGISTRY[vid])
            experiment_overrides[vid] = override

        pairwise_cfg = _build_pairwise_cfg(
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            experiments=variant_ids,
            topk_values=topk_values,
        )

        _set_seed(seed)
        print(f"\n  Running {len(variant_ids)} E16 variants in single pipeline pass...")
        pairwise_state = run_pairwise_pipeline(
            pairwise_cfg,
            experiment_overrides=experiment_overrides,
        )
        pairwise_df = pairwise_state["experiment_results"].copy()
        pairwise_df["seed"] = pairwise_df["seed"].fillna(seed).astype(int)
        all_records.extend(pairwise_df.to_dict(orient="records"))

        for _, row in pairwise_df.iterrows():
            print(
                f"  {row['experiment_id']}: "
                f"NDCG@10 all={row.get('NDCG@10 (все)', float('nan')):.4f}, "
                f"cold={row.get('NDCG@10 (холодные)', float('nan')):.4f}"
            )

        del pairwise_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    experiment_results = pd.DataFrame(all_records)

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
        "variant_ids": variant_ids,
        "variant_overrides": {k: v for k, v in VARIANT_REGISTRY.items() if k in variant_ids},
        "paper_mode_recommend_cold": PAPER_RECOMMEND_COLD,
        "paper_mode_filter_cold_history": PAPER_FILTER_COLD_HISTORY,
        "runtime_sec_total": float(time.perf_counter() - started),
        "dataset_stats": dataset_stats,
    }

    _save_outputs(
        output_dir=output_dir,
        title=f"{dataset_spec['title']} | E16 VARIANT SWEEP",
        experiment_results=experiment_results,
        results_summary=results_summary,
        significance_df=significance_df,
        metadata=metadata,
    )

    print("\n" + "=" * 80)
    print("E16 VARIANT SWEEP READY")
    print("=" * 80)
    print(results_summary.to_string(index=False))
    if significance_df is not None and not significance_df.empty:
        print("\n=== experiment_significance ===")
        print(significance_df.to_string(index=False))
    print(f"\nTotal runtime: {time.perf_counter() - started:.0f}s")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
