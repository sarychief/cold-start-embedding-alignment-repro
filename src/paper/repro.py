#!/usr/bin/env python3
"""Standalone let-it-go reproduction runner for Zvuk.

This script is intentionally kept separate from the pairwise pipeline.
It focuses on paper-like reproduction for the original let-it-go path:

1. optionally rebuild the Zvuk 10k splits from raw parquet files;
2. train SASRec / content-init / trainable-delta with the original let-it-go
   model and optimizer definitions;
3. evaluate the four cold-start modes from let-it-go;
4. aggregate metrics over multiple random seeds.

It uses the current Python environment and therefore does not depend on
Hydra / ClearML / replay at runtime. Validation and test metrics are computed
with the standard single-positive-item formulas, which match HR/Recall and
NDCG for this setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from paths import PAPER_REPRO_ARTIFACTS_DIR


DEFAULT_RAW_ZVUK_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "alexxl"
    / "zvuk-dataset"
    / "versions"
    / "1"
)
DEFAULT_DATASET_DIR = Path.home() / "let-it-go" / "data" / "zvuk_paper_repro"
DEFAULT_OUTPUT_DIR = PAPER_REPRO_ARTIFACTS_DIR / "letitgo_paper_repro"
README_WARM_ITEMS = 107448
README_COLD_ITEMS = 23637
README_TOTAL_USERS = 10000


def _parse_int_list(raw_value: str) -> list[int]:
    values = [part.strip() for part in str(raw_value).split(",")]
    return [int(part) for part in values if part]


def _parse_bool(raw_value: str | int | bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return bool(raw_value)
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {raw_value!r}")


def _resolve_device(raw_value: str) -> torch.device:
    normalized = str(raw_value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)


class EmbeddingManager(Pipeline):
    def __init__(self, embedding_dim: int, reduce: bool = True, normalize: bool = True) -> None:
        steps = []
        if reduce:
            steps.append(("scaler", StandardScaler()))
            steps.append(("pca", PCA(n_components=embedding_dim)))
        if normalize:
            steps.append(("normalizer", Normalizer()))
        super().__init__(steps)
        self.embedding_dim = embedding_dim
        self.reduce = reduce
        self.normalize = normalize


class ConstrainedNormAdam(torch.optim.Adam):
    def __init__(
        self,
        params: Any,
        constrained_params: Any,
        pad_token_id: int = 0,
        max_norm: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self.constrained_params = next(constrained_params)
        self.pad_token_id = pad_token_id
        self.max_norm = max_norm
        self.mask = torch.ones(self.constrained_params.data.shape[0], dtype=torch.bool)
        self.mask[pad_token_id] = False

    def step(self, closure: Any = None) -> Any:
        loss = super().step(closure=closure)
        parameter = self.constrained_params
        with torch.no_grad():
            parameter[self.mask] = torch.renorm(parameter[self.mask], 2, 0, self.max_norm)
        return loss


class _SequentialDataset(Dataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        add_labels: bool,
        user_column: str,
        item_column: str,
        timestamp_column: str,
        max_length: int,
        pad_token_id: int,
        ignore_index: int,
    ) -> None:
        self.add_labels = add_labels
        self.user_column = user_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.interactions = self._make_sequential(interactions)
        if add_labels:
            self.interactions = self.interactions.filter(pl.col("history").list.len() > 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.interactions.row(index, named=True)
        item["inputs"] = torch.tensor(item["history"][-self.max_length :], dtype=torch.long)
        if "labels" in item:
            item["labels"] = torch.tensor(item["labels"], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.interactions)

    def _make_sequential(self, interactions: pl.DataFrame) -> pl.DataFrame:
        return (
            interactions.sort(self.timestamp_column)
            .group_by(self.user_column, maintain_order=True)
            .agg(pl.col(self.item_column).alias("history"))
        )

    def _create_padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs != self.pad_token_id).float()

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch: dict[str, Any] = {}
        for key in batch[0]:
            collated_batch[key] = [item[key] for item in batch]

        collated_batch["inputs"] = pad_sequence(
            collated_batch["inputs"],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        collated_batch["padding_mask"] = self._create_padding_mask(collated_batch["inputs"])

        if "labels" in collated_batch:
            if collated_batch["labels"][0].dim() == 0:
                collated_batch["labels"] = torch.stack(collated_batch["labels"])
            else:
                collated_batch["labels"] = pad_sequence(
                    collated_batch["labels"],
                    batch_first=True,
                    padding_value=self.ignore_index,
                    padding_side="left",
                )
        return collated_batch


class _TestMixin:
    def __init__(self, ground_truth: pl.DataFrame | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if ground_truth is not None:
            if self.add_labels:
                raise ValueError("`add_labels` must be False when `ground_truth` is provided.")
            self.interactions = self.interactions.join(
                ground_truth.rename({self.item_column: "labels"}),
                on=self.user_column,
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if self.add_labels:
            item["labels"] = item["labels"][-1]
            item["history"] = item["history"][:-1]
        return item


class _TrainMixin:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(add_labels=True, **kwargs)


class _CausalMixin:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.add_labels:
            self.max_length += 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if self.add_labels:
            item["labels"] = item["inputs"][1:].clone()
            item["inputs"] = item["inputs"][:-1]
        return item


class TestCausalDataset(_TestMixin, _CausalMixin, _SequentialDataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        ground_truth: pl.DataFrame | None = None,
        add_labels: bool = False,
        user_column: str = "user_id",
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        max_length: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(
            interactions=interactions,
            ground_truth=ground_truth,
            add_labels=add_labels,
            user_column=user_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            max_length=max_length,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
        )


class TrainCausalDataset(_TrainMixin, _CausalMixin, _SequentialDataset):
    def __init__(
        self,
        interactions: pl.DataFrame,
        user_column: str = "user_id",
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        max_length: int = 64,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(
            interactions=interactions,
            user_column=user_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            max_length=max_length,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
        )


class _RecommenderModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        intermediate_dim: int,
        p: float,
        max_length: int,
        init_range: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.p = p
        self.max_length = max_length
        self.init_range = init_range
        self.pad_token_id = pad_token_id

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeddings = self.item_embedding(inputs)
        output = self._forward(inputs_embeddings, padding_mask)
        return torch.matmul(output, self.item_embedding.weight.T)

    def _forward(self, inputs_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _freeze_padding_embedding_hook(self, grad: torch.Tensor) -> torch.Tensor:
        grad[self.pad_token_id] = 0.0
        return grad

    def _add_padding_embedding(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.vstack(
            (
                item_embeddings[: self.pad_token_id],
                torch.zeros(self.embedding_dim),
                item_embeddings[self.pad_token_id :],
            )
        )

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        add_padding_embedding: bool = True,
        freeze: bool = False,
    ) -> None:
        if add_padding_embedding:
            item_embeddings = self._add_padding_embedding(item_embeddings)
        self.item_embedding = nn.Embedding.from_pretrained(
            item_embeddings, freeze=freeze, padding_idx=self.pad_token_id
        )


class _FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int, intermediate_dim: int, p: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embedding_dim),
            nn.Dropout(p=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        intermediate_dim: int,
        p: float,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=p, batch_first=True)
        self.ffn = _FeedForwardNetwork(embedding_dim, intermediate_dim, p)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        q = self.layer_norm1(x)
        causal_mask = torch.triu(
            torch.full((x.shape[1], x.shape[1]), -torch.inf, device=x.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(q, x, x, key_padding_mask=padding_mask, attn_mask=causal_mask)
        x = q + attn_out
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class SASRecModel(_RecommenderModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        intermediate_dim: int = 128,
        p: float = 0.1,
        max_length: int = 64,
        init_range: float = 0.02,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            p=p,
            max_length=max_length,
            init_range=init_range,
            pad_token_id=pad_token_id,
        )
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=pad_token_id)
        self.item_embedding.weight.register_hook(self._freeze_padding_embedding_hook)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [_SASRecBlock(embedding_dim, num_heads, intermediate_dim, p) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward(self, inputs_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = inputs_embeddings
        x *= self.embedding_dim**0.5
        x += self.pos_embedding(torch.arange(x.shape[1], dtype=torch.long, device=x.device))
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.layer_norm(x)


class _TrainableDeltaMixin:
    def __init__(self, max_delta_norm: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_delta_norm = max_delta_norm
        self.delta_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=self.pad_token_id,
            max_norm=max_delta_norm,
        )
        self.delta_embedding.weight.register_hook(self._freeze_padding_embedding_hook)
        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeddings = self.item_embedding(inputs) + self.delta_embedding(inputs)
        output = self._forward(inputs_embeddings, padding_mask)
        return torch.matmul(output, (self.item_embedding.weight + self.delta_embedding.weight).T)

    def set_pretrained_item_embeddings(
        self,
        item_embeddings: torch.Tensor,
        delta_embeddings: torch.Tensor | None = None,
        add_padding_embedding: bool = True,
        freeze: bool = False,
    ) -> None:
        del freeze
        super().set_pretrained_item_embeddings(
            item_embeddings,
            add_padding_embedding=add_padding_embedding,
            freeze=True,
        )
        if delta_embeddings is not None:
            if item_embeddings.shape != delta_embeddings.shape:
                raise ValueError(
                    "item_embeddings and delta_embeddings must have the same shape. "
                    f"Found {item_embeddings.shape} and {delta_embeddings.shape}."
                )
            self.delta_embedding = nn.Embedding.from_pretrained(
                delta_embeddings,
                freeze=True,
                padding_idx=self.pad_token_id,
            )


class SASRecModelWithTrainableDelta(_TrainableDeltaMixin, SASRecModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        intermediate_dim: int = 128,
        p: float = 0.1,
        max_length: int = 64,
        init_range: float = 0.02,
        pad_token_id: int = 0,
        max_delta_norm: float = 0.5,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            p=p,
            max_length=max_length,
            init_range=init_range,
            pad_token_id=pad_token_id,
            max_delta_norm=max_delta_norm,
        )


def _build_local_modules() -> dict[str, Any]:
    return {
        "TrainCausalDataset": TrainCausalDataset,
        "TestCausalDataset": TestCausalDataset,
        "EmbeddingManager": EmbeddingManager,
        "ConstrainedNormAdam": ConstrainedNormAdam,
        "SASRecModel": SASRecModel,
        "SASRecModelWithTrainableDelta": SASRecModelWithTrainableDelta,
    }


def _remove_consecutive_duplicates(interactions: pl.DataFrame) -> pl.DataFrame:
    return (
        interactions.sort(["user_id", "timestamp"])
        .with_columns(pl.col("item_id").shift(1).over("user_id").alias("__prev_item"))
        .filter((pl.col("item_id") != pl.col("__prev_item")).fill_null(True))
        .drop("__prev_item")
    )


def _iterative_n_core_filter(interactions: pl.DataFrame, n_core: int) -> pl.DataFrame:
    if n_core <= 1:
        return interactions

    prev_shape = None
    current = interactions
    while prev_shape != current.shape:
        prev_shape = current.shape
        valid_users = (
            current.group_by("user_id")
            .len()
            .filter(pl.col("len") >= n_core)
            .get_column("user_id")
        )
        current = current.filter(pl.col("user_id").is_in(valid_users.implode()))

        valid_items = (
            current.group_by("item_id")
            .len()
            .filter(pl.col("len") >= n_core)
            .get_column("item_id")
        )
        current = current.filter(pl.col("item_id").is_in(valid_items.implode()))
    return current


def rebuild_zvuk_dataset(
    *,
    raw_zvuk_dir: Path,
    output_dataset_dir: Path,
    split_seed: int,
    num_users: int,
    min_duration_sec: float,
    n_core: int,
    time_threshold_quantile: float,
    train_users_fraction: float,
) -> dict[str, Any]:
    raw_zvuk_dir = raw_zvuk_dir.expanduser().resolve()
    output_dataset_dir = output_dataset_dir.expanduser().resolve()
    processed_dir = output_dataset_dir / "processed"
    embeddings_dir = output_dataset_dir / "item_embeddings"
    _ensure_dir(processed_dir)
    _ensure_dir(embeddings_dir)

    interactions_path = raw_zvuk_dir / "zvuk-interactions.parquet"
    embeddings_path = raw_zvuk_dir / "zvuk-track_artist_embedding.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Raw interactions file not found: {interactions_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Raw embeddings file not found: {embeddings_path}")

    logging.info("Loading raw interactions from %s", interactions_path)
    interactions = pl.read_parquet(interactions_path)
    interactions = interactions.rename({"datetime": "timestamp", "track_id": "item_id"})
    interactions = interactions.filter(pl.col("play_duration") > float(min_duration_sec)).drop(
        "play_duration"
    )
    interactions = interactions.with_columns(pl.col("timestamp").dt.epoch(time_unit="ms"))

    users = interactions.get_column("user_id").unique().sort()
    sampled_users = users.sample(n=int(num_users), seed=int(split_seed))
    interactions = interactions.filter(pl.col("user_id").is_in(sampled_users.implode()))
    interactions = _remove_consecutive_duplicates(interactions)

    time_threshold = interactions.get_column("timestamp").quantile(float(time_threshold_quantile))
    train_val = interactions.filter(pl.col("timestamp") <= time_threshold)
    test = interactions.filter(pl.col("timestamp") > time_threshold)

    train_val = _iterative_n_core_filter(train_val, int(n_core))

    test_users = test.get_column("user_id").unique()
    test = pl.concat(
        [test, train_val.filter(pl.col("user_id").is_in(test_users.implode()))],
        how="vertical",
    )

    train_users = (
        train_val.get_column("user_id")
        .unique()
        .sort()
        .sample(fraction=float(train_users_fraction), seed=int(split_seed))
    )
    train = train_val.filter(pl.col("user_id").is_in(train_users.implode()))
    val = train_val.filter(~pl.col("user_id").is_in(train_users.implode()))

    warm_items = train.get_column("item_id").unique().sort()
    item2index_warm = {int(item): idx + 1 for idx, item in enumerate(warm_items.to_list())}

    train = train.with_columns(pl.col("item_id").replace_strict(item2index_warm))
    val = val.filter(pl.col("item_id").is_in(warm_items.implode()))
    val = val.with_columns(pl.col("item_id").replace_strict(item2index_warm))

    test = test.with_columns((~pl.col("item_id").is_in(warm_items.implode())).alias("is_cold"))
    cold_items = test.filter(pl.col("is_cold")).get_column("item_id").unique().sort()
    bias = max(item2index_warm.values()) + 1
    item2index_cold = {int(item): idx + bias for idx, item in enumerate(cold_items.to_list())}
    item2index_all = {**item2index_warm, **item2index_cold}
    test = test.with_columns(pl.col("item_id").replace_strict(item2index_all))

    test = test.with_columns(
        pl.col("user_id")
        .cum_count(reverse=True)
        .over("user_id", order_by="timestamp")
        .alias("position")
    )
    ground_truth = test.filter(pl.col("position") == 1)
    test_inputs = test.filter(pl.col("position") != 1)

    train.write_parquet(processed_dir / "train_interactions.parquet")
    val.write_parquet(processed_dir / "val_interactions.parquet")
    test_inputs.write_parquet(processed_dir / "test_interactions.parquet")
    ground_truth.write_parquet(processed_dir / "ground_truth.parquet")

    with (processed_dir / "item2index_warm.pkl").open("wb") as fp:
        pickle.dump(item2index_warm, fp)
    with (processed_dir / "item2index_cold.pkl").open("wb") as fp:
        pickle.dump(item2index_cold, fp)

    logging.info("Loading raw item embeddings from %s", embeddings_path)
    metadata = (
        pl.read_parquet(embeddings_path)
        .rename({"track_id": "item_id"})
        .filter(pl.col("item_id").is_in(pl.Series(list(item2index_all.keys())).implode()))
        .with_columns(pl.col("item_id").replace_strict(item2index_all))
        .unique(["item_id", "vector"])
        .sort("item_id")
    )
    if len(metadata) != len(item2index_all):
        raise RuntimeError(
            "Metadata item count does not match warm+cold catalog: "
            f"{len(metadata)} vs {len(item2index_all)}"
        )

    item_embeddings = np.vstack(metadata.get_column("vector").to_list()).astype(np.float32)
    warm_embeddings = item_embeddings[: len(item2index_warm)]
    cold_embeddings = item_embeddings[len(item2index_warm) :]
    np.save(embeddings_dir / "embeddings_warm.npy", warm_embeddings)
    np.save(embeddings_dir / "embeddings_cold.npy", cold_embeddings)

    stats = inspect_dataset(
        output_dataset_dir,
        expected_warm_items=README_WARM_ITEMS,
        expected_cold_items=README_COLD_ITEMS,
    )
    stats["raw_zvuk_dir"] = str(raw_zvuk_dir)
    stats["split_seed"] = int(split_seed)
    stats["num_users"] = int(num_users)
    stats["min_duration_sec"] = float(min_duration_sec)
    stats["n_core"] = int(n_core)
    stats["time_threshold_quantile"] = float(time_threshold_quantile)
    stats["train_users_fraction"] = float(train_users_fraction)
    _save_json(output_dataset_dir / "dataset_stats.json", stats)
    return stats


def inspect_dataset(
    dataset_dir: Path,
    *,
    expected_warm_items: int,
    expected_cold_items: int,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    processed_dir = dataset_dir / "processed"
    item_embeddings_dir = dataset_dir / "item_embeddings"

    paths = {
        "train": processed_dir / "train_interactions.parquet",
        "val": processed_dir / "val_interactions.parquet",
        "test": processed_dir / "test_interactions.parquet",
        "ground_truth": processed_dir / "ground_truth.parquet",
        "warm_pickle": processed_dir / "item2index_warm.pkl",
        "cold_pickle": processed_dir / "item2index_cold.pkl",
        "warm_embeddings": item_embeddings_dir / "embeddings_warm.npy",
        "cold_embeddings": item_embeddings_dir / "embeddings_cold.npy",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset directory is incomplete. Missing: " + ", ".join(f"{name}={paths[name]}" for name in missing)
        )

    train = pl.read_parquet(paths["train"])
    val = pl.read_parquet(paths["val"])
    test = pl.read_parquet(paths["test"])
    ground_truth = pl.read_parquet(paths["ground_truth"])
    with paths["warm_pickle"].open("rb") as fp:
        warm_map = pickle.load(fp)
    with paths["cold_pickle"].open("rb") as fp:
        cold_map = pickle.load(fp)
    warm_embeddings = np.load(paths["warm_embeddings"])
    cold_embeddings = np.load(paths["cold_embeddings"])

    stats = {
        "dataset_dir": str(dataset_dir),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "ground_truth_rows": int(len(ground_truth)),
        "warm_items_pickle": int(len(warm_map)),
        "cold_items_pickle": int(len(cold_map)),
        "warm_items_embeddings": int(warm_embeddings.shape[0]),
        "cold_items_embeddings": int(cold_embeddings.shape[0]),
        "embedding_dim": int(warm_embeddings.shape[1]) if warm_embeddings.ndim == 2 else 0,
        "matches_readme_warm_items": int(len(warm_map)) == int(expected_warm_items),
        "matches_readme_cold_items": int(len(cold_map)) == int(expected_cold_items),
        "expected_warm_items": int(expected_warm_items),
        "expected_cold_items": int(expected_cold_items),
    }
    return stats


def _build_loader(
    dataset_cls: Any,
    interactions: pl.DataFrame,
    *,
    add_labels: bool,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if add_labels:
        try:
            dataset = dataset_cls(
                interactions,
                add_labels=True,
                max_length=max_length,
            )
        except TypeError:
            dataset = dataset_cls(
                interactions,
                max_length=max_length,
            )
    else:
        dataset = dataset_cls(
            interactions,
            add_labels=False,
            max_length=max_length,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(add_labels),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def _prepare_pretrained_embeddings(
    modules: dict[str, Any],
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_manager = modules["EmbeddingManager"](
        embedding_dim=embedding_dim,
        reduce=embedding_dim != warm_embeddings.shape[1],
        normalize=True,
    )
    warm_prepared = embedding_manager.fit_transform(warm_embeddings)
    cold_prepared = embedding_manager.transform(cold_embeddings)
    return warm_prepared.astype(np.float32), cold_prepared.astype(np.float32)


def _build_model(modules: dict[str, Any], args: argparse.Namespace, num_items: int) -> nn.Module:
    common_kwargs = {
        "num_items": int(num_items),
        "embedding_dim": int(args.embedding_dim),
        "num_blocks": int(args.num_blocks),
        "num_heads": int(args.num_heads),
        "intermediate_dim": int(args.embedding_dim),
        "p": float(args.dropout),
        "max_length": int(args.max_length),
    }
    if args.model_kind == "delta":
        return modules["SASRecModelWithTrainableDelta"](
            max_delta_norm=float(args.max_delta_norm),
            **common_kwargs,
        )
    return modules["SASRecModel"](**common_kwargs)


def _build_optimizer(
    modules: dict[str, Any],
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.model_kind == "delta":
        return modules["ConstrainedNormAdam"](
            model.parameters(),
            constrained_params=model.delta_embedding.parameters(),
            max_norm=float(args.max_delta_norm),
            lr=float(args.learning_rate),
        )
    return torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))


def _add_cold_item_embeddings(model: nn.Module, cold_item_embeddings: np.ndarray) -> None:
    cold_item_embeddings_tensor = torch.tensor(cold_item_embeddings).float()
    item_embeddings = model.item_embedding.weight[: model.num_items + 1].detach().cpu()

    if hasattr(model, "delta_embedding"):
        delta_embeddings = model.delta_embedding.weight[: model.num_items + 1].detach().cpu()
        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack([item_embeddings, cold_item_embeddings_tensor]),
            delta_embeddings=torch.vstack(
                [delta_embeddings, torch.zeros_like(cold_item_embeddings_tensor)]
            ),
            add_padding_embedding=False,
            freeze=True,
        )
    else:
        model.set_pretrained_item_embeddings(
            item_embeddings=torch.vstack([item_embeddings, cold_item_embeddings_tensor]),
            add_padding_embedding=False,
            freeze=True,
        )


def _recommend_topk(
    logits: torch.Tensor,
    histories: list[list[int]],
    *,
    topk: int,
    remove_seen: bool,
    recommend_cold_items: bool,
    warm_num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.clone()
    if remove_seen:
        for idx, seen in enumerate(histories):
            if seen:
                logits[idx, torch.tensor(seen, device=logits.device, dtype=torch.long)] = -torch.inf

    if not recommend_cold_items:
        logits = logits[:, : warm_num_items + 1]

    scores, items = torch.topk(logits, k=min(topk, logits.shape[1]), dim=1)
    return scores, items


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _compute_eval_metrics(
    predictions: dict[int, list[int]],
    ground_truth: pl.DataFrame,
    topk_values: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    by_subset: dict[str, dict[int, list[float]]] = {
        "cold_hr": {k: [] for k in topk_values},
        "cold_ndcg": {k: [] for k in topk_values},
        "warm_hr": {k: [] for k in topk_values},
        "warm_ndcg": {k: [] for k in topk_values},
    }

    for row in ground_truth.iter_rows(named=True):
        user_id = int(row["user_id"])
        target_item = int(row["item_id"])
        is_cold = bool(row["is_cold"])
        predicted_items = predictions.get(user_id, [])
        subset_prefix = "cold" if is_cold else "warm"

        for k in topk_values:
            top_items = predicted_items[:k]
            if target_item in top_items:
                rank = top_items.index(target_item)
                hr_value = 1.0
                ndcg_value = 1.0 / math.log2(rank + 2.0)
            else:
                hr_value = 0.0
                ndcg_value = 0.0
            by_subset[f"{subset_prefix}_hr"][k].append(hr_value)
            by_subset[f"{subset_prefix}_ndcg"][k].append(ndcg_value)

    for k in topk_values:
        cold_hr = _mean(by_subset["cold_hr"][k])
        warm_hr = _mean(by_subset["warm_hr"][k])
        cold_ndcg = _mean(by_subset["cold_ndcg"][k])
        warm_ndcg = _mean(by_subset["warm_ndcg"][k])
        num_cold = len(by_subset["cold_hr"][k])
        num_warm = len(by_subset["warm_hr"][k])
        denom = max(1, num_cold + num_warm)

        total_hr = (cold_hr * num_cold + warm_hr * num_warm) / denom
        total_ndcg = (cold_ndcg * num_cold + warm_ndcg * num_warm) / denom

        metrics[f"cold_Recall@{k}"] = cold_hr
        metrics[f"warm_Recall@{k}"] = warm_hr
        metrics[f"Recall@{k}"] = total_hr
        metrics[f"cold_HR@{k}"] = cold_hr
        metrics[f"warm_HR@{k}"] = warm_hr
        metrics[f"HR@{k}"] = total_hr
        metrics[f"cold_NDCG@{k}"] = cold_ndcg
        metrics[f"warm_NDCG@{k}"] = warm_ndcg
        metrics[f"NDCG@{k}"] = total_ndcg

    return metrics


@torch.no_grad()
def _evaluate_validation_ndcg(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    topk: int,
) -> float:
    model.eval()
    total_ndcg = 0.0
    total_examples = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(inputs, padding_mask)[:, -1, :]
        _, items = _recommend_topk(
            logits,
            histories=batch["history"],
            topk=topk,
            remove_seen=True,
            recommend_cold_items=True,
            warm_num_items=model.num_items,
        )

        hits = items.eq(labels.unsqueeze(1))
        any_hit = hits.any(dim=1)
        hit_positions = hits.float().argmax(dim=1)
        discounts = 1.0 / torch.log2(hit_positions.float() + 2.0)
        total_ndcg += float(torch.where(any_hit, discounts, torch.zeros_like(discounts)).sum().item())
        total_examples += int(labels.shape[0])

    return total_ndcg / max(1, total_examples)


def train_single_seed(
    *,
    args: argparse.Namespace,
    modules: dict[str, Any],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    warm_embeddings: np.ndarray,
    cold_embeddings: np.ndarray,
    device: torch.device,
    seed: int,
    topk_values: list[int],
) -> tuple[nn.Module, dict[str, Any]]:
    _set_seed(seed)

    if len(train_df) == 0:
        raise ValueError("Train split is empty. Rebuild the paper preprocessing artifacts before training.")
    if len(val_df) == 0:
        raise ValueError(
            "Validation split is empty. Rebuild the paper preprocessing artifacts before training. "
            "For Amazon M2 this usually means the validation rows were filtered after item-id remapping."
        )

    train_loader = _build_loader(
        modules["TrainCausalDataset"],
        train_df,
        add_labels=True,
        max_length=int(args.max_length),
        batch_size=int(args.train_batch_size),
        num_workers=int(args.num_workers),
    )
    val_loader = _build_loader(
        modules["TestCausalDataset"],
        val_df,
        add_labels=True,
        max_length=int(args.max_length),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )

    model = _build_model(modules, args, num_items=warm_embeddings.shape[0])
    if args.model_kind in {"content_init", "delta"}:
        warm_prepared, cold_prepared = _prepare_pretrained_embeddings(
            modules,
            warm_embeddings=warm_embeddings,
            cold_embeddings=cold_embeddings,
            embedding_dim=int(args.embedding_dim),
        )
        model.set_pretrained_item_embeddings(
            torch.tensor(warm_prepared).float(),
            add_padding_embedding=True,
            freeze=False,
        )
    else:
        cold_prepared = cold_embeddings.astype(np.float32)

    model.to(device)
    optimizer = _build_optimizer(modules, model, args)
    loss_fn = nn.CrossEntropyLoss()

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_ndcg = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = time.time()

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, padding_mask)
            loss = loss_fn(logits.view(-1, model.num_items + 1), labels.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1

        val_ndcg = _evaluate_validation_ndcg(
            model,
            val_loader,
            device=device,
            topk=topk_values[0],
        )
        logging.info(
            "seed=%s epoch=%s train_loss=%.6f val_ndcg@%s=%.6f",
            seed,
            epoch,
            epoch_loss / max(1, epoch_steps),
            topk_values[0],
            val_ndcg,
        )

        if val_ndcg > best_val_ndcg + 1e-12:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(args.patience):
            break

    model.load_state_dict(best_state)
    model.to(device)
    if args.model_kind in {"content_init", "delta"}:
        _add_cold_item_embeddings(model, cold_prepared)
        model.to(device)

    train_info = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_ndcg": float(best_val_ndcg),
        "runtime_sec": float(time.time() - started_at),
    }
    return model, train_info


@torch.no_grad()
def evaluate_all_modes(
    *,
    args: argparse.Namespace,
    modules: dict[str, Any],
    model: nn.Module,
    test_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    device: torch.device,
    topk_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cold_items_available = int(model.item_embedding.weight.shape[0]) > int(model.num_items + 1)
    k_max = max(topk_values)

    loader_cache: dict[bool, DataLoader] = {}

    for recommend_cold_items in (True, False):
        for filter_cold_items in (True, False):
            if not cold_items_available:
                if recommend_cold_items:
                    continue
                if not filter_cold_items:
                    continue

            if filter_cold_items not in loader_cache:
                interactions = (
                    test_df.filter(~pl.col("is_cold")) if filter_cold_items else test_df
                )
                loader_cache[filter_cold_items] = _build_loader(
                    modules["TestCausalDataset"],
                    interactions,
                    add_labels=False,
                    max_length=int(args.max_length),
                    batch_size=int(args.eval_batch_size),
                    num_workers=int(args.num_workers),
                )
            test_loader = loader_cache[filter_cold_items]

            predictions: dict[int, list[int]] = {}
            model.eval()
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                logits = model(inputs, padding_mask)[:, -1, :]
                _, items = _recommend_topk(
                    logits,
                    histories=batch["history"],
                    topk=k_max,
                    remove_seen=True,
                    recommend_cold_items=recommend_cold_items,
                    warm_num_items=model.num_items,
                )
                for user_id, predicted_items in zip(batch["user_id"], items.cpu().tolist(), strict=True):
                    predictions[int(user_id)] = [int(item_id) for item_id in predicted_items]

            metrics = _compute_eval_metrics(predictions, ground_truth_df, topk_values)
            row = {
                "recommend_cold_items": bool(recommend_cold_items),
                "filter_cold_items": bool(filter_cold_items),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for row in rows:
        key = (bool(row["recommend_cold_items"]), bool(row["filter_cold_items"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (recommend_cold_items, filter_cold_items), mode_rows in grouped.items():
        metric_keys = [
            key
            for key in mode_rows[0]
            if key not in {"seed", "model_kind", "recommend_cold_items", "filter_cold_items"}
        ]
        summary: dict[str, Any] = {
            "recommend_cold_items": recommend_cold_items,
            "filter_cold_items": filter_cold_items,
            "num_runs": len(mode_rows),
        }
        for metric_key in metric_keys:
            values = [float(row[metric_key]) for row in mode_rows]
            summary[f"{metric_key}_mean"] = float(np.mean(values))
            summary[f"{metric_key}_std"] = float(np.std(values))
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (not row["recommend_cold_items"], not row["filter_cold_items"]))
    return summary_rows


def build_primary_mode_summary(
    summary_rows: list[dict[str, Any]],
    *,
    recommend_cold_items: bool,
    filter_cold_items: bool,
    topk: int,
) -> dict[str, Any]:
    for row in summary_rows:
        if (
            bool(row["recommend_cold_items"]) == bool(recommend_cold_items)
            and bool(row["filter_cold_items"]) == bool(filter_cold_items)
        ):
            return {
                "recommend_cold_items": bool(recommend_cold_items),
                "filter_cold_items": bool(filter_cold_items),
                "NDCG": {
                    "cold_mean": float(row[f"cold_NDCG@{topk}_mean"]),
                    "cold_std": float(row[f"cold_NDCG@{topk}_std"]),
                    "warm_mean": float(row[f"warm_NDCG@{topk}_mean"]),
                    "warm_std": float(row[f"warm_NDCG@{topk}_std"]),
                    "total_mean": float(row[f"NDCG@{topk}_mean"]),
                    "total_std": float(row[f"NDCG@{topk}_std"]),
                },
                "HR": {
                    "cold_mean": float(row[f"cold_HR@{topk}_mean"]),
                    "cold_std": float(row[f"cold_HR@{topk}_std"]),
                    "warm_mean": float(row[f"warm_HR@{topk}_mean"]),
                    "warm_std": float(row[f"warm_HR@{topk}_std"]),
                    "total_mean": float(row[f"HR@{topk}_mean"]),
                    "total_std": float(row[f"HR@{topk}_std"]),
                },
            }
    raise RuntimeError(
        "Primary mode summary not found for "
        f"recommend_cold_items={recommend_cold_items}, filter_cold_items={filter_cold_items}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-faithful let-it-go reproduction for Zvuk.")
    parser.add_argument(
        "--action",
        choices=["check", "rebuild", "run", "all"],
        default="check",
        help="check dataset stats, rebuild the dataset, run training/evaluation, or do both",
    )
    parser.add_argument(
        "--raw-zvuk-dir",
        default=str(DEFAULT_RAW_ZVUK_DIR),
        help="Directory with raw Zvuk parquet files",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Target directory with processed/ and item_embeddings/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where run artifacts will be written",
    )
    parser.add_argument(
        "--model-kind",
        choices=["sasrec", "content_init", "delta"],
        default="delta",
        help="Which let-it-go model variant to run",
    )
    parser.add_argument(
        "--seeds",
        default="42,221,451,934,1984",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for Zvuk split creation")
    parser.add_argument("--num-users", type=int, default=README_TOTAL_USERS)
    parser.add_argument("--min-duration-sec", type=float, default=60.0)
    parser.add_argument("--n-core", type=int, default=3)
    parser.add_argument("--time-threshold-quantile", type=float, default=0.9)
    parser.add_argument("--train-users-fraction", type=float, default=0.9)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-delta-norm", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--topk", default="10", help="Comma-separated top-k values")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--primary-recommend-cold", default="1")
    parser.add_argument("--primary-filter-cold-history", default="0")
    parser.add_argument("--expected-warm-items", type=int, default=README_WARM_ITEMS)
    parser.add_argument("--expected-cold-items", type=int, default=README_COLD_ITEMS)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_zvuk_dir = Path(args.raw_zvuk_dir).expanduser().resolve()
    topk_values = _parse_int_list(args.topk)
    device = _resolve_device(args.device)
    primary_recommend_cold = _parse_bool(args.primary_recommend_cold)
    primary_filter_cold = _parse_bool(args.primary_filter_cold_history)

    _ensure_dir(output_dir)
    _save_json(
        output_dir / "run_config.json",
        {
            key: getattr(args, key)
            for key in vars(args)
        },
    )

    if args.action in {"rebuild", "all"}:
        stats = rebuild_zvuk_dataset(
            raw_zvuk_dir=raw_zvuk_dir,
            output_dataset_dir=dataset_dir,
            split_seed=int(args.split_seed),
            num_users=int(args.num_users),
            min_duration_sec=float(args.min_duration_sec),
            n_core=int(args.n_core),
            time_threshold_quantile=float(args.time_threshold_quantile),
            train_users_fraction=float(args.train_users_fraction),
        )
        logging.info("Rebuilt Zvuk dataset at %s", dataset_dir)
        logging.info("Dataset stats: %s", json.dumps(stats, ensure_ascii=True))

    dataset_stats = inspect_dataset(
        dataset_dir,
        expected_warm_items=int(args.expected_warm_items),
        expected_cold_items=int(args.expected_cold_items),
    )
    _save_json(output_dir / "dataset_stats.json", dataset_stats)
    logging.info("Dataset stats: %s", json.dumps(dataset_stats, ensure_ascii=True))

    if args.action == "check":
        return 0

    modules = _build_local_modules()
    processed_dir = dataset_dir / "processed"
    item_embeddings_dir = dataset_dir / "item_embeddings"
    train_df = pl.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pl.read_parquet(processed_dir / "val_interactions.parquet")
    test_df = pl.read_parquet(processed_dir / "test_interactions.parquet")
    ground_truth_df = pl.read_parquet(processed_dir / "ground_truth.parquet")
    warm_embeddings = np.load(item_embeddings_dir / "embeddings_warm.npy")
    cold_embeddings = np.load(item_embeddings_dir / "embeddings_cold.npy")

    seeds = _parse_int_list(args.seeds)
    all_rows: list[dict[str, Any]] = []
    train_runs: list[dict[str, Any]] = []

    for seed in seeds:
        logging.info("Starting seed %s on device %s", seed, device)
        model, train_info = train_single_seed(
            args=args,
            modules=modules,
            train_df=train_df,
            val_df=val_df,
            warm_embeddings=warm_embeddings,
            cold_embeddings=cold_embeddings,
            device=device,
            seed=seed,
            topk_values=topk_values,
        )
        train_runs.append(train_info)

        seed_rows = evaluate_all_modes(
            args=args,
            modules=modules,
            model=model,
            test_df=test_df,
            ground_truth_df=ground_truth_df,
            device=device,
            topk_values=topk_values,
        )
        for row in seed_rows:
            row["seed"] = int(seed)
            row["model_kind"] = args.model_kind
            all_rows.append(row)

    summary_rows = aggregate_results(all_rows)
    primary_summary = build_primary_mode_summary(
        summary_rows,
        recommend_cold_items=primary_recommend_cold,
        filter_cold_items=primary_filter_cold,
        topk=topk_values[0],
    )

    all_rows_df = pl.DataFrame(all_rows)
    summary_df = pl.DataFrame(summary_rows)
    train_runs_df = pl.DataFrame(train_runs)
    all_rows_df.write_csv(output_dir / "per_seed_results.csv")
    summary_df.write_csv(output_dir / "summary_by_mode.csv")
    train_runs_df.write_csv(output_dir / "train_runs.csv")
    _save_json(output_dir / "per_seed_results.json", all_rows_df.to_dicts())
    _save_json(output_dir / "summary_by_mode.json", summary_df.to_dicts())
    _save_json(output_dir / "train_runs.json", train_runs_df.to_dicts())
    _save_json(output_dir / "primary_mode_summary.json", primary_summary)

    logging.info("Primary mode summary: %s", json.dumps(primary_summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
