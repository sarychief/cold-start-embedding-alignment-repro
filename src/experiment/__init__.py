"""Experiment pipeline package for the cold-start recommendation stack."""

from .config import ExperimentConfig
from . import data
from . import models
from . import training
from . import pairwise
from . import analysis
from . import pipeline
from .pipeline import (
    build_results_step,
    compute_significance_table,
    run_ablation_grid,
    run_baseline_training,
    run_cold_evaluation_step,
    run_data_pipeline,
    run_embedding_training,
    run_encoding_pipeline,
    run_experiment_grid,
    run_full_pipeline,
    run_implicit_slim_step,
    run_let_it_go_step,
    run_pairwise_step,
    run_pairwise_transformer_step,
    run_seed_stability,
    run_split_and_sequences,
)

SASRecWithTrainableDelta = getattr(models, "SASRecWithTrainableDelta", None)

__all__ = [
    "ExperimentConfig",
    "SASRecWithTrainableDelta",
    "data",
    "models",
    "training",
    "pairwise",
    "analysis",
    "pipeline",
    "build_results_step",
    "compute_significance_table",
    "run_ablation_grid",
    "run_baseline_training",
    "run_cold_evaluation_step",
    "run_data_pipeline",
    "run_embedding_training",
    "run_encoding_pipeline",
    "run_experiment_grid",
    "run_full_pipeline",
    "run_implicit_slim_step",
    "run_let_it_go_step",
    "run_pairwise_step",
    "run_pairwise_transformer_step",
    "run_seed_stability",
    "run_split_and_sequences",
]
