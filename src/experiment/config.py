"""Configuration objects used across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from paths import PAIRWISE_ARTIFACTS_DIR


@dataclass
class DataConfig:
    """Options for dataset loading and synthetic fallbacks."""

    data_dir: str = "./data"
    use_letitgo_official_splits: bool = False
    letitgo_repo_path: str = field(default_factory=lambda: str(Path.home() / "let-it-go"))
    letitgo_dataset_key: str = "zvuk"
    letitgo_processed_dir: str = ""
    letitgo_item_embeddings_dir: str = ""
    use_zvuk_dataset: bool = True
    use_kaggle_dataset: bool = False
    zvuk_dataset_id: str = "alexxl/zvuk-dataset"
    auto_download_zvuk_dataset: bool = True
    zvuk_data_path: str = field(
        default_factory=lambda: str(
            Path.home() / ".cache" / "kagglehub" / "datasets" / "alexxl" / "zvuk-dataset" / "versions" / "1"
        )
    )
    kaggle_data_path: str = field(
        default_factory=lambda: str(
            Path.home() / ".cache" / "kagglehub" / "datasets" / "marquis03" / "amazon-m2" / "versions" / "1"
        )
    )
    locale: str = "DE"
    dataset_name: str = "Beauty"
    min_interactions: int = 5
    zvuk_sample_fraction: float = 0.01
    zvuk_min_user_interactions: int = 5
    zvuk_min_item_interactions: int = 5
    create_cold_items_flag: bool = True
    cold_threshold: int = 5
    cold_fraction: float = 0.2
    item_role_mode: str = "current"
    item_role_k: int = 5


@dataclass
class ModelConfig:
    """Model and optimization defaults."""

    max_len: int = 50
    num_items_hidden: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.2
    epochs: int = 5
    patience: int = 5
    min_delta: float = 1e-4
    letitgo_epochs: int = 25
    letitgo_patience: int = 4
    letitgo_min_delta: float = 1e-4
    letitgo_max_delta_norm: float = 0.5
    lr: float = 0.001
    batch_size: int = 256


@dataclass
class AlignmentConfig:
    """Settings for regularization and pairwise alignment."""

    use_interaction_features: bool = False
    interaction_dim: int = 64
    lam: float = 1000.0
    alpha: float = 1.0
    similarity_top_k: int | None = 100
    pairwise_model_alpha: float = 0.5
    pairwise_warm_regularization: float = 0.3
    pairwise_use_transformer_mapper: bool = True
    pairwise_source_model_preference: str = "model_with_embeddings"
    pairwise_target_mode: str = "full"
    pairwise_infer_scope: str = "cold"
    pairwise_warm_sampler: str = "all"
    pairwise_warm_sample_size: int = 0
    pairwise_warm_similarity: str = "cosine"
    pairwise_warm_mix_ratio: float = 0.5
    pairwise_transformer_epochs: int = 15
    pairwise_transformer_lr: float = 0.001
    pairwise_transformer_layers: int = 2
    pairwise_transformer_heads: int = 2
    pairwise_transformer_hidden_dim: int = 128
    pairwise_transformer_batch_size: int = 128
    pairwise_transformer_token_count: int = 8
    pairwise_transformer_dropout: float = 0.1
    pairwise_transformer_weight_decay: float = 1e-4
    pairwise_transformer_grad_clip: float = 1.0
    pairwise_transformer_val_fraction: float = 0.1
    pairwise_transformer_patience: int = 4
    pairwise_transformer_min_delta: float = 1e-4
    pairwise_transformer_loss_mse_weight: float = 0.2
    pairwise_transformer_loss_cosine_weight: float = 0.6
    pairwise_transformer_loss_nce_weight: float = 0.2
    pairwise_transformer_nce_temperature: float = 0.07
    pairwise_transformer_sample_weight_power: float = 0.5
    pairwise_transformer_min_warm_interactions: int = 10
    pairwise_transformer_blend_alpha: float = 0.35
    pairwise_transformer_adapt_epochs: int = 1
    pairwise_transformer_adapt_lr: float = 5e-4
    pairwise_transformer_structure_weight: float = 0.2
    pairwise_transformer_hard_negative_weight: float = 0.2
    pairwise_transformer_hard_negative_top_k: int = 512
    pairwise_distill_pair_rounds: int = 2
    pairwise_distill_candidate_count: int = 64
    pairwise_distill_teacher_margin: float = 0.01


@dataclass
class ExperimentRegistryConfig:
    """Global registry and execution defaults for E0..E16/E3S experiments."""

    enabled_experiments: list[str] = field(
        default_factory=lambda: [
            "E0", "E11", "E1", "E2", "E3", "E3S", "E4", "E5", "E6", "E7", "E8", "E10", "E12", "E13", "E14"
        ]
    )
    ablation_experiment_id: str = "E9"
    primary_baseline_id: str = "E0"
    save_dir: str = field(default_factory=lambda: str(PAIRWISE_ARTIFACTS_DIR))
    save_json: bool = True
    save_csv: bool = True
    include_runtime: bool = True
    topk_values: list[int] = field(default_factory=lambda: [10, 20, 50])
    run_significance: bool = True
    significance_test: str = "wilcoxon"
    significance_alpha: float = 0.05
    significance_metric: str = "NDCG@10 (холодные)"
    paper_eval_enabled: bool = False
    paper_eval_recommend_cold_items: bool = True
    paper_eval_filter_cold_history: bool = False
    paper_eval_report_all_modes: bool = False


@dataclass
class AblationConfig:
    """Hyperparameter sweeps for E9 controlled ablations."""

    enabled: bool = True
    method_id: str = "E4"
    projection_dims: list[int] = field(default_factory=lambda: [128, 192, 256])
    temperatures: list[float] = field(default_factory=lambda: [0.05, 0.07, 0.1])
    hard_negative_top_k: list[int] = field(default_factory=lambda: [256, 512, 1024])
    loss_weight_sets: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"mse": 0.20, "cosine": 0.60, "nce": 0.20},
            {"mse": 0.10, "cosine": 0.50, "nce": 0.40},
            {"mse": 0.15, "cosine": 0.55, "nce": 0.30},
        ]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [128, 256])
    sample_weight_power: list[float] = field(default_factory=lambda: [0.4, 0.6, 0.8])
    blend_alpha: list[float] = field(default_factory=lambda: [0.25, 0.35, 0.50])
    adapt_epochs: list[int] = field(default_factory=lambda: [0, 1])
    distill_pair_rounds: list[int] = field(default_factory=lambda: [1, 2, 4])
    distill_candidate_count: list[int] = field(default_factory=lambda: [32, 64, 128])
    distill_teacher_margin: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.03])
    max_trials: int | None = 24
    random_state: int = 42


@dataclass
class SeedConfig:
    """Settings for repeated seed experiments."""

    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 2024])
    epochs: int = 5


@dataclass
class ExperimentConfig:
    """All experiment configuration in one object."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    align: AlignmentConfig = field(default_factory=AlignmentConfig)
    registry: ExperimentRegistryConfig = field(default_factory=ExperimentRegistryConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
