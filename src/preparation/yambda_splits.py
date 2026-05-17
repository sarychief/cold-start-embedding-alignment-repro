#!/usr/bin/env python3
"""Prepare Yambda-50M splits in let-it-go official format.

Downloads Yambda-50M from HuggingFace (yandex/yambda), processes into:
  - processed/train_interactions.parquet
  - processed/val_interactions.parquet
  - processed/test_interactions.parquet
  - processed/ground_truth.parquet
  - processed/item2index_warm.pkl
  - processed/item2index_cold.pkl
  - item_embeddings/embeddings_warm.npy
  - item_embeddings/embeddings_cold.npy
  - dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Yambda-50M official-style splits for let-it-go mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "let-it-go" / "data" / "yambda"),
        help="Output directory where processed/ and item_embeddings/ will be written",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-played-ratio",
        type=int,
        default=50,
        help="Minimum played_ratio_pct to count as a positive interaction",
    )
    parser.add_argument("--n-core", type=int, default=5, help="N-core filtering threshold")
    parser.add_argument("--time-threshold-quantile", type=float, default=0.9)
    parser.add_argument("--train-users-fraction", type=float, default=0.9)
    parser.add_argument(
        "--num-users",
        type=int,
        default=0,
        help="If > 0, keep exactly this many users (overrides --user-fraction).",
    )
    parser.add_argument("--user-fraction", type=float, default=1.0, help="Fraction of users to keep")
    parser.add_argument(
        "--listens-path",
        default="",
        help="Path to pre-downloaded listens.parquet (skip HuggingFace download).",
    )
    parser.add_argument(
        "--embeddings-path",
        default="",
        help="Path to pre-downloaded embeddings.parquet (skip HuggingFace download).",
    )
    return parser


def _download_hf_file(repo_file: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"Downloading {repo_file} from yandex/yambda...")
    path = hf_hub_download("yandex/yambda", repo_file, repo_type="dataset")
    print(f"  cached at: {path}")
    return path


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
    embeddings_path: str,
    item2index_all: dict[int, int],
    warm_ids_sorted: list[int],
    cold_ids_sorted: list[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    needed_raw_ids = list(item2index_all.keys())
    print(f"Loading embeddings for {len(needed_raw_ids)} items from {embeddings_path} ...")
    emb_df = (
        pl.scan_parquet(embeddings_path)
        .filter(pl.col("item_id").is_in(needed_raw_ids))
        .select(["item_id", "normalized_embed"])
        .collect()
    )
    print(f"  found embeddings for {emb_df.height} / {len(needed_raw_ids)} items")

    id_to_vec: dict[int, np.ndarray] = {}
    for row in emb_df.iter_rows(named=True):
        raw_id = int(row["item_id"])
        reindexed_id = item2index_all[raw_id]
        id_to_vec[reindexed_id] = np.asarray(row["normalized_embed"], dtype=np.float32)

    embedding_dim = 128
    if id_to_vec:
        embedding_dim = int(len(next(iter(id_to_vec.values()))))

    def _vec(item_id: int) -> np.ndarray:
        if item_id in id_to_vec:
            return id_to_vec[item_id]
        local_rng = np.random.RandomState((seed * 1315423911 + item_id) % (2**32))
        return local_rng.randn(embedding_dim).astype(np.float32)

    warm_arr = (
        np.vstack([_vec(i) for i in warm_ids_sorted])
        if warm_ids_sorted
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    cold_arr = (
        np.vstack([_vec(i) for i in cold_ids_sorted])
        if cold_ids_sorted
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    return warm_arr, cold_arr


def main() -> int:
    args = _build_parser().parse_args()

    output_dir = Path(args.output_dir).expanduser()
    processed_dir = output_dir / "processed"
    embeddings_dir = output_dir / "item_embeddings"
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    listens_path = (
        args.listens_path
        if args.listens_path.strip()
        else _download_hf_file("flat/50m/listens.parquet")
    )
    embeddings_path = (
        args.embeddings_path
        if args.embeddings_path.strip()
        else _download_hf_file("embeddings.parquet")
    )

    print(f"Loading interactions from: {listens_path}")
    interactions = pl.read_parquet(listens_path)
    interactions = interactions.rename({"uid": "user_id"})
    print(f"Raw interactions: {interactions.shape}")

    interactions = interactions.filter(
        pl.col("played_ratio_pct") >= int(args.min_played_ratio)
    ).select(["user_id", "item_id", "timestamp"])
    print(f"After positive filter (played >= {args.min_played_ratio}%): {interactions.shape}")

    users = interactions.get_column("user_id").unique().sort()
    n_users = len(users)
    if int(args.num_users) > 0:
        keep_n = min(int(args.num_users), int(n_users))
    else:
        keep_n = max(1, int(round(n_users * float(args.user_fraction))))

    if keep_n < n_users:
        sampled_users = users.sample(n=keep_n, seed=int(args.seed), shuffle=True)
        interactions = interactions.filter(pl.col("user_id").is_in(sampled_users.implode()))
        print(f"After user subsample ({keep_n}/{n_users} users): {interactions.shape}")
    else:
        print(f"Using all {n_users} users")

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
    test = pl.concat(
        [test, train_val.filter(pl.col("user_id").is_in(test_users.implode()))],
        how="vertical",
    )
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

    warm_ids_sorted = sorted(item2index_warm.values())
    cold_ids_sorted = sorted(item2index_cold.values())
    warm_emb, cold_emb = _build_embedding_arrays(
        embeddings_path=embeddings_path,
        item2index_all=item2index_all,
        warm_ids_sorted=warm_ids_sorted,
        cold_ids_sorted=cold_ids_sorted,
        seed=int(args.seed),
    )

    np.save(embeddings_dir / "embeddings_warm.npy", warm_emb.astype(np.float32))
    np.save(embeddings_dir / "embeddings_cold.npy", cold_emb.astype(np.float32))

    stats = {
        "dataset_key": "yambda",
        "num_warm_items": len(warm_ids_sorted),
        "num_cold_items": len(cold_ids_sorted),
        "embedding_dim": int(warm_emb.shape[1]) if warm_emb.ndim == 2 else 128,
        "train_rows": int(train.shape[0]),
        "val_rows": int(val.shape[0]),
        "test_inputs_rows": int(test_inputs.shape[0]),
        "ground_truth_rows": int(ground_truth.shape[0]),
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\nSaved processed splits to: {processed_dir}")
    print(f"Saved embeddings to: {embeddings_dir}")
    print(f"warm/cold items: {len(warm_ids_sorted)} / {len(cold_ids_sorted)}")
    emb_dim = int(warm_emb.shape[1]) if warm_emb.ndim == 2 else 0
    print(f"embedding dim: {emb_dim}")
    print(f"Dataset stats saved to: {output_dir / 'dataset_stats.json'}")
    print(
        f"\n>>> After preparation, update DATASET_SPECS in src/paper/comparison.py:\n"
        f'    "yambda": {{\n'
        f'        "dataset_dir_name": "yambda",\n'
        f'        "embedding_dim": {emb_dim},\n'
        f'        "max_length": 128,\n'
        f'        "expected_warm_items": {stats["num_warm_items"]},\n'
        f'        "expected_cold_items": {stats["num_cold_items"]},\n'
        f'        "title": "YAMBDA PAPER COMPARISON",\n'
        f"    }},"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
