#!/usr/bin/env python3
"""Prepare Zvuk splits in let-it-go official format.

This script creates:
  - processed/train_interactions.parquet
  - processed/val_interactions.parquet
  - processed/test_interactions.parquet
  - processed/ground_truth.parquet
  - processed/item2index_warm.pkl
  - processed/item2index_cold.pkl
  - item_embeddings/embeddings_warm.npy
  - item_embeddings/embeddings_cold.npy
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 1%%/N%% Zvuk official-style splits for let-it-go mode.")
    parser.add_argument(
        "--zvuk-data-path",
        default=str(
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "alexxl"
            / "zvuk-dataset"
            / "versions"
            / "1"
        ),
        help="Directory containing zvuk-interactions.parquet and zvuk-track_artist_embedding.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "let-it-go" / "data" / "zvuk_1pct"),
        help="Output directory where processed/ and item_embeddings/ will be written",
    )
    parser.add_argument("--user-fraction", type=float, default=0.01, help="Fraction of users to keep")
    parser.add_argument(
        "--num-users",
        type=int,
        default=0,
        help="If > 0, keep exactly this many users (overrides --user-fraction).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration-sec", type=float, default=60.0)
    parser.add_argument("--n-core", type=int, default=3)
    parser.add_argument("--time-threshold-quantile", type=float, default=0.9)
    parser.add_argument("--train-users-fraction", type=float, default=0.9)
    return parser


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


def _build_embedding_arrays(
    metadata: pl.DataFrame,
    warm_ids_sorted: list[int],
    cold_ids_sorted: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    id_to_vec = {
        int(item_id): np.asarray(vec, dtype=np.float32)
        for item_id, vec in zip(metadata.get_column("item_id").to_list(), metadata.get_column("vector").to_list())
    }
    if id_to_vec:
        embedding_dim = int(len(next(iter(id_to_vec.values()))))
    else:
        embedding_dim = 128

    rng = np.random.RandomState(seed)

    def _vec(item_id: int) -> np.ndarray:
        if item_id in id_to_vec:
            return id_to_vec[item_id]
        # deterministic fallback per item
        local_rng = np.random.RandomState((seed * 1315423911 + item_id) % (2**32))
        return local_rng.randn(embedding_dim).astype(np.float32)

    warm_arr = np.vstack([_vec(item_id) for item_id in warm_ids_sorted]) if warm_ids_sorted else np.empty((0, embedding_dim), dtype=np.float32)
    cold_arr = np.vstack([_vec(item_id) for item_id in cold_ids_sorted]) if cold_ids_sorted else np.empty((0, embedding_dim), dtype=np.float32)
    # keep random state consumed consistently
    _ = rng.rand(1)
    return warm_arr, cold_arr


def main() -> int:
    args = _build_parser().parse_args()

    data_dir = Path(args.zvuk_data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    processed_dir = output_dir / "processed"
    embeddings_dir = output_dir / "item_embeddings"
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = data_dir / "zvuk-interactions.parquet"
    embeddings_path = data_dir / "zvuk-track_artist_embedding.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Не найден файл interactions: {interactions_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Не найден файл embeddings: {embeddings_path}")

    print(f"Loading interactions from: {interactions_path}")
    interactions = pl.read_parquet(interactions_path)
    interactions = interactions.rename({"datetime": "timestamp", "track_id": "item_id"})
    interactions = interactions.filter(pl.col("play_duration") > float(args.min_duration_sec)).drop("play_duration")
    interactions = interactions.with_columns(pl.col("timestamp").dt.epoch(time_unit="ms"))
    print(f"After positive filter: {interactions.shape}")

    users = interactions.get_column("user_id").unique().sort()
    n_users = len(users)
    if int(args.num_users) > 0:
        keep_n = min(int(args.num_users), int(n_users))
    else:
        keep_n = max(1, int(round(n_users * float(args.user_fraction))))
    sampled_users = users.sample(n=keep_n, seed=int(args.seed), shuffle=True)
    interactions = interactions.filter(pl.col("user_id").is_in(sampled_users.implode()))
    print(f"After user subsample ({keep_n}/{n_users} users): {interactions.shape}")

    interactions = _remove_consecutive_duplicates(interactions)
    print(f"After consecutive-duplicate filter: {interactions.shape}")

    q = float(args.time_threshold_quantile)
    time_threshold = interactions.get_column("timestamp").quantile(q)
    train_val = interactions.filter(pl.col("timestamp") <= time_threshold)
    test = interactions.filter(pl.col("timestamp") > time_threshold)
    print(f"Initial split train_val/test: {train_val.shape} / {test.shape}")

    train_val = _iterative_n_core_filter(train_val, int(args.n_core))
    print(f"After {args.n_core}-core on train_val: {train_val.shape}")

    test_users = test.get_column("user_id").unique()
    test = pl.concat([test, train_val.filter(pl.col("user_id").is_in(test_users.implode()))], how="vertical")
    print(f"After adding history for test users: {test.shape}")

    train_users = (
        train_val.get_column("user_id")
        .unique()
        .sort()
        .sample(fraction=float(args.train_users_fraction), seed=int(args.seed))
    )
    train = train_val.filter(pl.col("user_id").is_in(train_users.implode()))
    val = train_val.filter(~pl.col("user_id").is_in(train_users.implode()))
    print(f"Train/Val split: {train.shape} / {val.shape}")

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
    print(
        "Final splits train/val/test_inputs/ground_truth: "
        f"{train.shape} / {val.shape} / {test_inputs.shape} / {ground_truth.shape}"
    )

    train.write_parquet(processed_dir / "train_interactions.parquet")
    val.write_parquet(processed_dir / "val_interactions.parquet")
    test_inputs.write_parquet(processed_dir / "test_interactions.parquet")
    ground_truth.write_parquet(processed_dir / "ground_truth.parquet")

    with (processed_dir / "item2index_warm.pkl").open("wb") as f:
        pickle.dump(item2index_warm, f)
    with (processed_dir / "item2index_cold.pkl").open("wb") as f:
        pickle.dump(item2index_cold, f)

    selected_raw_items = list(item2index_all.keys())
    metadata = (
        pl.scan_parquet(embeddings_path)
        .select(["track_id", "vector"])
        .filter(pl.col("track_id").is_in(selected_raw_items))
        .rename({"track_id": "item_id"})
        .with_columns(pl.col("item_id").replace_strict(item2index_all))
        .group_by("item_id")
        .agg(pl.col("vector").first())
        .sort("item_id")
        .collect()
    )
    print(f"Filtered metadata rows: {metadata.shape}")

    warm_ids_sorted = sorted(item2index_warm.values())
    cold_ids_sorted = sorted(item2index_cold.values())
    warm_emb, cold_emb = _build_embedding_arrays(
        metadata=metadata,
        warm_ids_sorted=warm_ids_sorted,
        cold_ids_sorted=cold_ids_sorted,
        seed=int(args.seed),
    )

    np.save(embeddings_dir / "embeddings_warm.npy", warm_emb.astype(np.float32))
    np.save(embeddings_dir / "embeddings_cold.npy", cold_emb.astype(np.float32))

    print(f"Saved processed splits to: {processed_dir}")
    print(f"Saved embeddings to: {embeddings_dir}")
    print(f"warm/cold items: {len(warm_ids_sorted)} / {len(cold_ids_sorted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
