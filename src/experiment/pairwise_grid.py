#!/usr/bin/env python3
"""CLI runner for full/ablation pairwise-alignment experiments.

Run from tmux, for example:
  python -m experiment.pairwise_grid --mode full
  python -m experiment.pairwise_grid --mode ablation --ablation-max-trials 24
  python -m experiment.pairwise_grid --mode quick --synthetic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import ExperimentConfig
from paths import PAIRWISE_ARTIFACTS_DIR
from .pipeline import (
    build_results_step,
    run_baseline_training,
    run_data_pipeline,
    run_embedding_training,
    run_encoding_pipeline,
    run_experiment_grid,
    run_implicit_slim_step,
    run_split_and_sequences,
)


def _parse_list(value: str, cast):
    if not value:
        return []
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(cast(token))
    return out


def _default_experiments_for_mode(mode: str) -> list[str]:
    if mode == "quick":
        return ["E0", "E11", "E3", "E3S", "E10", "E12", "E13", "E14"]
    return ["E0", "E11", "E1", "E2", "E3", "E3S", "E4", "E5", "E6", "E7", "E8", "E10", "E12", "E13", "E14"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise alignment experiment grid (E0-E16/E3S, E9=ablation).")
    parser.add_argument("--mode", choices=["quick", "full", "ablation"], default="full")
    parser.add_argument("--output-dir", default=str(PAIRWISE_ARTIFACTS_DIR))
    parser.add_argument("--experiments", default="", help="Comma-separated experiments, e.g. E0,E1,E4")
    parser.add_argument("--topk", default="10,20,50", help="Comma-separated K values, e.g. 10,20,50")

    parser.add_argument("--zvuk-sample-fraction", type=float, default=0.01)
    parser.add_argument(
        "--use-letitgo-splits",
        action="store_true",
        help="Use official preprocessed splits from let-it-go (train/val/test + warm/cold ids).",
    )
    parser.add_argument(
        "--letitgo-dataset",
        choices=["zvuk", "amazon_m2"],
        default="zvuk",
        help="Dataset key for official let-it-go splits.",
    )
    parser.add_argument(
        "--letitgo-repo-path",
        default=str(Path.home() / "let-it-go"),
        help="Path to local let-it-go repository root.",
    )
    parser.add_argument(
        "--letitgo-processed-dir",
        default="",
        help="Optional explicit path to let-it-go processed split directory.",
    )
    parser.add_argument(
        "--letitgo-embeddings-dir",
        default="",
        help="Optional explicit path to let-it-go item_embeddings directory.",
    )
    parser.add_argument(
        "--paper-eval",
        action="store_true",
        help="Use paper-like evaluation protocol for official let-it-go splits.",
    )
    parser.add_argument(
        "--paper-eval-recommend-cold",
        type=int,
        choices=[0, 1],
        default=1,
        help="Paper eval mode: include cold items in candidate set (1/0).",
    )
    parser.add_argument(
        "--paper-eval-filter-cold-history",
        type=int,
        choices=[0, 1],
        default=0,
        help="Paper eval mode: filter cold items from user history (1/0).",
    )
    parser.add_argument(
        "--paper-eval-report-all-modes",
        action="store_true",
        help="Additionally log all four paper protocol mode combinations.",
    )
    parser.add_argument("--model-epochs", type=int, default=5)
    parser.add_argument(
        "--letitgo-epochs",
        type=int,
        default=25,
        help="Epochs for let-it-go delta training.",
    )
    parser.add_argument("--letitgo-patience", type=int, default=4)
    parser.add_argument("--letitgo-min-delta", type=float, default=1e-4)
    parser.add_argument("--model-batch-size", type=int, default=256)

    parser.add_argument("--mapper-epochs", type=int, default=None)
    parser.add_argument("--mapper-hidden-dim", type=int, default=None)
    parser.add_argument("--mapper-batch-size", type=int, default=None)
    parser.add_argument("--mapper-layers", type=int, default=None)
    parser.add_argument("--mapper-heads", type=int, default=None)
    parser.add_argument("--min-warm-interactions", type=int, default=15)
    parser.add_argument(
        "--pairwise-target-mode",
        choices=["full", "delta_from_content"],
        default="full",
        help="Train/evaluate mapper in full-target or delta-from-content mode.",
    )
    parser.add_argument(
        "--pairwise-warm-sampler",
        choices=["all", "popular", "closest_to_cold", "mixed"],
        default="all",
        help="Warm-item sampler for mapper training.",
    )
    parser.add_argument(
        "--pairwise-warm-sample-size",
        type=int,
        default=0,
        help="Number of warm items to keep for sampler modes. <=0 keeps all.",
    )
    parser.add_argument(
        "--pairwise-warm-similarity",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity used for closest_to_cold warm sampling.",
    )
    parser.add_argument(
        "--pairwise-warm-mix-ratio",
        type=float,
        default=0.5,
        help="Share of popular items inside mixed warm sampler.",
    )
    parser.add_argument(
        "--pairwise-infer-scope",
        choices=["cold", "warm", "all"],
        default="cold",
        help="Which item groups receive mapper-based updates.",
    )
    parser.add_argument(
        "--pairwise-item-role-mode",
        choices=["current", "strict_zero_vs_gt_k"],
        default="current",
        help="How warm/cold/discarded item groups are defined.",
    )
    parser.add_argument(
        "--pairwise-item-role-k",
        type=int,
        default=5,
        help="Threshold K for strict_zero_vs_gt_k mode.",
    )
    parser.add_argument(
        "--pairwise-source",
        choices=["embeddings", "implicit", "auto"],
        default="embeddings",
        help="Source model for pairwise mappers (E1..E8,E10,E12,E13,E14,E15,E16,E3S).",
    )
    parser.add_argument("--distill-pair-rounds", type=int, default=None)
    parser.add_argument("--distill-candidate-count", type=int, default=None)
    parser.add_argument("--distill-teacher-margin", type=float, default=None)

    parser.add_argument("--ablation-method", default="E4", help="Parent method for E9 ablation, e.g. E3S, E10, E14")
    parser.add_argument("--ablation-max-trials", type=int, default=24)
    parser.add_argument(
        "--experiment-overrides-json",
        default="",
        help="Path to JSON mapping experiment_id -> override dict (applied in full/quick runs).",
    )

    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data only (for fast sanity checks)")
    parser.add_argument("--no-significance", action="store_true")
    parser.add_argument("--no-save-json", action="store_true")
    parser.add_argument("--no-save-csv", action="store_true")
    return parser


def _apply_mode_defaults(cfg: ExperimentConfig, mode: str) -> None:
    if mode == "quick":
        cfg.align.pairwise_transformer_epochs = 8
        cfg.align.pairwise_transformer_hidden_dim = 128
        cfg.align.pairwise_transformer_batch_size = 128
        cfg.align.pairwise_transformer_layers = 2
        cfg.align.pairwise_transformer_heads = 2
        return

    # full / ablation defaults
    cfg.align.pairwise_transformer_epochs = 25
    cfg.align.pairwise_transformer_hidden_dim = 256
    cfg.align.pairwise_transformer_batch_size = 256
    cfg.align.pairwise_transformer_layers = 3
    cfg.align.pairwise_transformer_heads = 4


def _configure(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    _apply_mode_defaults(cfg, args.mode)

    # Dataset
    cfg.data.zvuk_sample_fraction = float(args.zvuk_sample_fraction)
    cfg.data.use_letitgo_official_splits = bool(args.use_letitgo_splits)
    cfg.data.letitgo_dataset_key = str(args.letitgo_dataset)
    cfg.data.letitgo_repo_path = str(Path(args.letitgo_repo_path).expanduser())
    cfg.data.letitgo_processed_dir = str(args.letitgo_processed_dir or "")
    cfg.data.letitgo_item_embeddings_dir = str(args.letitgo_embeddings_dir or "")
    cfg.data.item_role_mode = str(args.pairwise_item_role_mode)
    cfg.data.item_role_k = int(args.pairwise_item_role_k)
    if cfg.data.use_letitgo_official_splits:
        cfg.data.create_cold_items_flag = False
        cfg.data.auto_download_zvuk_dataset = False
        cfg.data.use_zvuk_dataset = False
        cfg.data.use_kaggle_dataset = False
        # Match let-it-go paper defaults more closely on official splits.
        if cfg.data.letitgo_dataset_key == "zvuk":
            cfg.model.num_items_hidden = 128
            cfg.model.max_len = 128
        elif cfg.data.letitgo_dataset_key == "amazon_m2":
            cfg.model.num_items_hidden = 64
            cfg.model.max_len = 64
        cfg.model.num_heads = 1
        cfg.model.dropout_rate = 0.3
    if args.synthetic:
        cfg.data.use_zvuk_dataset = False
        cfg.data.use_kaggle_dataset = False
        cfg.data.data_dir = "./_no_real_data_here"

    cfg.registry.paper_eval_enabled = bool(args.paper_eval)
    cfg.registry.paper_eval_recommend_cold_items = bool(int(args.paper_eval_recommend_cold))
    cfg.registry.paper_eval_filter_cold_history = bool(int(args.paper_eval_filter_cold_history))
    cfg.registry.paper_eval_report_all_modes = bool(args.paper_eval_report_all_modes)

    # Base model
    cfg.model.epochs = int(args.model_epochs)
    cfg.model.letitgo_epochs = int(args.letitgo_epochs)
    cfg.model.letitgo_patience = int(args.letitgo_patience)
    cfg.model.letitgo_min_delta = float(args.letitgo_min_delta)
    cfg.model.batch_size = int(args.model_batch_size)

    # Mapper overrides
    if args.mapper_epochs is not None:
        cfg.align.pairwise_transformer_epochs = int(args.mapper_epochs)
    if args.mapper_hidden_dim is not None:
        cfg.align.pairwise_transformer_hidden_dim = int(args.mapper_hidden_dim)
    if args.mapper_batch_size is not None:
        cfg.align.pairwise_transformer_batch_size = int(args.mapper_batch_size)
    if args.mapper_layers is not None:
        cfg.align.pairwise_transformer_layers = int(args.mapper_layers)
    if args.mapper_heads is not None:
        cfg.align.pairwise_transformer_heads = int(args.mapper_heads)
    cfg.align.pairwise_transformer_min_warm_interactions = int(args.min_warm_interactions)
    cfg.align.pairwise_target_mode = str(args.pairwise_target_mode)
    cfg.align.pairwise_warm_sampler = str(args.pairwise_warm_sampler)
    cfg.align.pairwise_warm_sample_size = int(args.pairwise_warm_sample_size)
    cfg.align.pairwise_warm_similarity = str(args.pairwise_warm_similarity)
    cfg.align.pairwise_warm_mix_ratio = float(args.pairwise_warm_mix_ratio)
    cfg.align.pairwise_infer_scope = str(args.pairwise_infer_scope)
    if args.pairwise_source == "implicit":
        cfg.align.pairwise_source_model_preference = "model_with_implicit_slim"
    elif args.pairwise_source == "auto":
        cfg.align.pairwise_source_model_preference = "auto"
    else:
        cfg.align.pairwise_source_model_preference = "model_with_embeddings"
    if args.distill_pair_rounds is not None:
        cfg.align.pairwise_distill_pair_rounds = int(args.distill_pair_rounds)
    if args.distill_candidate_count is not None:
        cfg.align.pairwise_distill_candidate_count = int(args.distill_candidate_count)
    if args.distill_teacher_margin is not None:
        cfg.align.pairwise_distill_teacher_margin = float(args.distill_teacher_margin)

    # Registry / output
    cfg.registry.save_dir = args.output_dir
    cfg.registry.save_json = not bool(args.no_save_json)
    cfg.registry.save_csv = not bool(args.no_save_csv)
    cfg.registry.run_significance = not bool(args.no_significance)
    cfg.registry.topk_values = _parse_list(args.topk, int) or [10, 20, 50]

    if args.experiments:
        cfg.registry.enabled_experiments = [v.upper() for v in _parse_list(args.experiments, str)]
    else:
        cfg.registry.enabled_experiments = _default_experiments_for_mode(args.mode)

    # Ablation
    cfg.ablation.enabled = args.mode == "ablation"
    cfg.ablation.method_id = str(args.ablation_method).upper()
    cfg.ablation.max_trials = int(args.ablation_max_trials)

    return cfg


def _load_experiment_overrides(path_value: str) -> dict[str, dict]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Overrides JSON must be an object: {\"E3\": { ... }, ...}")
    out: dict[str, dict] = {}
    for exp_id, override in raw.items():
        exp_key = str(exp_id).upper()
        if not isinstance(override, dict):
            raise ValueError(f"Override for {exp_key} must be a JSON object.")
        out[exp_key] = dict(override)
    return out


def _save_cli_run_metadata(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    runtime_sec: float,
    experiment_overrides: dict[str, dict] | None = None,
) -> None:
    out_dir = Path(cfg.registry.save_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "args": vars(args),
        "runtime_sec": float(runtime_sec),
        "enabled_experiments": cfg.registry.enabled_experiments,
        "ablation_enabled": cfg.ablation.enabled,
        "topk_values": cfg.registry.topk_values,
        "experiment_overrides": experiment_overrides or {},
    }
    meta_path = out_dir / "cli_run_metadata.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _should_run_baseline_training(cfg: ExperimentConfig) -> bool:
    enabled = {str(v).upper() for v in cfg.registry.enabled_experiments}
    if cfg.ablation.enabled:
        return True
    # The standalone SASRec baseline is only needed for summary comparisons
    # and experiments that explicitly require it as a teacher/reference model.
    return "E13" in enabled


def run_pipeline(cfg: ExperimentConfig, experiment_overrides: dict[str, dict] | None = None) -> dict:
    state = run_data_pipeline(cfg)
    state = run_encoding_pipeline(state, cfg)
    state = run_split_and_sequences(state, cfg)
    if _should_run_baseline_training(cfg):
        state = run_baseline_training(state, cfg)
    else:
        print("Пропускаем обучение SASRec baseline: не требуется для выбранных экспериментов.")
    state = run_embedding_training(state, cfg)
    state = run_implicit_slim_step(state, cfg)
    state = run_experiment_grid(
        state,
        cfg,
        experiment_ids=cfg.registry.enabled_experiments,
        run_ablation=cfg.ablation.enabled,
        experiment_overrides=experiment_overrides or {},
    )
    state = build_results_step(state, cfg)
    return state


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _configure(args)
    experiment_overrides = _load_experiment_overrides(args.experiment_overrides_json)

    print("=" * 80)
    print(f"RUN MODE: {args.mode}")
    print(f"EXPERIMENTS: {cfg.registry.enabled_experiments}")
    print(f"ABLATION ENABLED: {cfg.ablation.enabled}")
    print(f"OUTPUT DIR: {cfg.registry.save_dir}")
    print(
        "PAPER EVAL: "
        f"{cfg.registry.paper_eval_enabled} "
        f"(recommend_cold={cfg.registry.paper_eval_recommend_cold_items}, "
        f"filter_cold_history={cfg.registry.paper_eval_filter_cold_history}, "
        f"report_all_modes={cfg.registry.paper_eval_report_all_modes})"
    )
    print(
        "PAIRWISE MODES: "
        f"target={cfg.align.pairwise_target_mode}, "
        f"warm_sampler={cfg.align.pairwise_warm_sampler}, "
        f"infer_scope={cfg.align.pairwise_infer_scope}, "
        f"item_role_mode={cfg.data.item_role_mode}, "
        f"item_role_k={cfg.data.item_role_k}"
    )
    if experiment_overrides:
        print(f"EXPERIMENT OVERRIDES: {sorted(experiment_overrides.keys())}")
    print("=" * 80)

    started = time.perf_counter()
    state = run_pipeline(cfg, experiment_overrides=experiment_overrides)
    elapsed = time.perf_counter() - started
    _save_cli_run_metadata(cfg, args, elapsed, experiment_overrides=experiment_overrides)

    out_dir = Path(cfg.registry.save_dir).expanduser()
    print("\nRun completed")
    print(f"Runtime: {elapsed:.1f} sec")
    print(f"Artifacts dir: {out_dir}")

    if "results_summary" in state:
        print("\n=== results_summary ===")
        print(state["results_summary"].to_string(index=False))

    if "experiment_significance" in state and len(state["experiment_significance"]) > 0:
        print("\n=== experiment_significance ===")
        print(state["experiment_significance"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

