"""Orchestrates the experiment as a sequence of importable functions."""

from __future__ import annotations

from typing import Any

import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig


def _resolve_config(config: ExperimentConfig | None) -> ExperimentConfig:
    return config if config is not None else ExperimentConfig()
from . import data as data_mod
from . import models
from . import training
from . import pairwise


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collect_item_interaction_frequencies(train_df: pd.DataFrame | None) -> dict[int, int]:
    if train_df is None or 'item_id_encoded' not in train_df.columns:
        return {}
    counts = train_df['item_id_encoded'].value_counts()
    return {int(item_idx): int(count) for item_idx, count in counts.items()}


def _filter_reliable_warm_items(
    warm_item_indices: list[int],
    item_frequencies: dict[int, int],
    min_interactions: int,
) -> list[int]:
    if min_interactions <= 1 or not item_frequencies:
        return sorted({int(idx) for idx in warm_item_indices if int(idx) > 0})
    filtered = [
        int(idx)
        for idx in warm_item_indices
        if int(idx) > 0 and int(item_frequencies.get(int(idx), 0)) >= int(min_interactions)
    ]
    return sorted(set(filtered))


def _get_training_excluded_item_ids(state: dict) -> list[int]:
    return [int(v) for v in state.get("train_excluded_item_ids", []) if int(v) > 0]


def _get_model_catalog_size(model) -> int:
    return max(int(model.item_emb.weight.size(0)) - 1, 0)


def _get_warm_catalog_size(state: dict) -> int:
    warm_items = [int(v) for v in state.get("warm_items", []) if int(v) > 0]
    if warm_items:
        return int(max(warm_items))
    return int(state.get("num_items", 0))


def _sorted_positive_item_ids(values) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def _dedupe_preserve_order(values) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item_idx = int(value)
        if item_idx <= 0 or item_idx in seen:
            continue
        seen.add(item_idx)
        out.append(item_idx)
    return out


def _filter_df_by_item_ids(df: pd.DataFrame | None, blocked_item_ids) -> pd.DataFrame | None:
    if df is None:
        return None
    blocked = {int(v) for v in blocked_item_ids if int(v) > 0}
    if df.empty or not blocked:
        return df.copy()
    item_col = None
    if "item_id_encoded" in df.columns:
        item_col = "item_id_encoded"
    elif "item_id" in df.columns:
        item_col = "item_id"
    if item_col is None:
        return df.copy()
    return df.loc[~df[item_col].isin(blocked)].copy()


def _derive_item_role_sets(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    role_mode: str,
    role_k: int,
    cold_threshold: int,
    cold_fraction: float,
    warm_hint: list[int] | None = None,
    cold_hint: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    role_mode = str(role_mode).strip().lower()
    role_k = int(max(0, role_k))
    train_counts = train_df['item_id_encoded'].value_counts() if not train_df.empty else pd.Series(dtype=int)
    eval_items = _sorted_positive_item_ids(eval_df['item_id_encoded'].unique().tolist()) if not eval_df.empty else []

    if role_mode == "strict_zero_vs_gt_k":
        warm_items = _sorted_positive_item_ids(train_counts[train_counts > role_k].index.tolist())
        discarded_items = _sorted_positive_item_ids(
            train_counts[(train_counts > 0) & (train_counts <= role_k)].index.tolist()
        )
        seen_in_train = {int(v) for v in train_counts.index.tolist() if int(v) > 0}
        discarded_set = set(discarded_items)
        cold_items = sorted(
            {
                int(v)
                for v in eval_items
                if int(v) > 0 and int(v) not in seen_in_train and int(v) not in discarded_set
            }
        )
        return warm_items, cold_items, discarded_items

    warm_items = _sorted_positive_item_ids(warm_hint or [])
    cold_items = _sorted_positive_item_ids(cold_hint or [])
    if warm_items or cold_items:
        return warm_items, cold_items, []

    threshold = int(cold_threshold)
    cold_items = _sorted_positive_item_ids(train_counts[train_counts <= threshold].index.tolist())
    warm_items = _sorted_positive_item_ids(train_counts[train_counts > threshold].index.tolist())
    if not cold_items:
        fraction = float(cold_fraction)
        min_cold_items = max(1, int(len(train_counts) * fraction))
        cold_items = _sorted_positive_item_ids(train_counts.nsmallest(min_cold_items).index.tolist())
        cold_set = set(cold_items)
        warm_items = [idx for idx in _sorted_positive_item_ids(train_counts.index.tolist()) if idx not in cold_set]
    return warm_items, cold_items, []


def _resolve_pairwise_target_item_indices(state: dict, infer_scope: str) -> list[int]:
    infer_scope = str(infer_scope).strip().lower()
    if infer_scope == "warm":
        return _sorted_positive_item_ids(state.get("warm_items", []))
    if infer_scope == "all":
        return _sorted_positive_item_ids(list(state.get("warm_items", [])) + list(state.get("cold_items", [])))
    return _sorted_positive_item_ids(state.get("cold_items", []))


def _resolve_pairwise_prediction_mode(target_mode: str, update_mode: str) -> str:
    normalized_target_mode = str(target_mode).strip().lower()
    normalized_update_mode = str(update_mode).strip().lower()
    if normalized_target_mode == "delta_from_content" and normalized_update_mode == "delta_add":
        return "delta"
    return "full"


def _count_item_group_members(item_ids, group_ids) -> int:
    group_set = {int(v) for v in group_ids if int(v) > 0}
    return int(sum(1 for item_idx in item_ids if int(item_idx) in group_set))


def _rank_warm_items_by_cold_similarity(
    state: dict,
    metric: str,
) -> list[int]:
    metric = str(metric).strip().lower()
    warm_pool = _sorted_positive_item_ids(state.get("warm_items", []))
    cold_pool = _sorted_positive_item_ids(state.get("cold_items", []))
    if not warm_pool or not cold_pool:
        return []

    cache = state.setdefault("_warm_to_cold_rank_cache", {})
    cache_key = (
        metric,
        len(warm_pool),
        int(sum(warm_pool)),
        len(cold_pool),
        int(sum(cold_pool)),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)

    item_embeddings = state.get("item_embeddings", {})
    idx_to_item = state.get("idx_to_item", {})

    warm_idx: list[int] = []
    warm_vectors: list[np.ndarray] = []
    for item_idx in warm_pool:
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        warm_idx.append(int(item_idx))
        warm_vectors.append(vec)

    cold_vectors: list[np.ndarray] = []
    for item_idx in cold_pool:
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        cold_vectors.append(vec)

    if not warm_vectors or not cold_vectors:
        return []

    warm_mat = np.asarray(warm_vectors, dtype=np.float32)
    cold_mat = np.asarray(cold_vectors, dtype=np.float32)
    if metric == "cosine":
        warm_mat = warm_mat / (np.linalg.norm(warm_mat, axis=1, keepdims=True) + 1e-8)
        cold_mat = cold_mat / (np.linalg.norm(cold_mat, axis=1, keepdims=True) + 1e-8)

    batch_size = 512 if cold_mat.shape[0] >= 4096 else 2048
    scores = np.full((warm_mat.shape[0],), -np.inf, dtype=np.float32)
    for start in range(0, warm_mat.shape[0], batch_size):
        end = min(warm_mat.shape[0], start + batch_size)
        sims = np.matmul(warm_mat[start:end], cold_mat.T)
        scores[start:end] = np.max(sims, axis=1)

    order = np.argsort(-scores)
    ranked = [warm_idx[int(pos)] for pos in order.tolist()]
    cache[cache_key] = list(ranked)
    return ranked


def _select_warm_items_for_mapper(
    state: dict,
    config: ExperimentConfig,
    *,
    item_frequencies: dict[int, int],
    override: dict | None = None,
    default_min_warm_interactions: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    override = override or {}
    warm_pool = _sorted_positive_item_ids(state.get("warm_items", []))
    min_warm_interactions = int(
        override.get(
            "pairwise_transformer_min_warm_interactions",
            config.align.pairwise_transformer_min_warm_interactions
            if default_min_warm_interactions is None
            else default_min_warm_interactions,
        )
    )
    reliable_warm = _filter_reliable_warm_items(warm_pool, item_frequencies, max(1, min_warm_interactions))
    base_pool = reliable_warm or warm_pool

    sampler = str(override.get("pairwise_warm_sampler", config.align.pairwise_warm_sampler)).strip().lower()
    sample_size = int(override.get("pairwise_warm_sample_size", config.align.pairwise_warm_sample_size))
    similarity = str(override.get("pairwise_warm_similarity", config.align.pairwise_warm_similarity)).strip().lower()
    mix_ratio = float(override.get("pairwise_warm_mix_ratio", config.align.pairwise_warm_mix_ratio))
    mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))

    info = {
        "sampler": sampler,
        "sample_size": int(sample_size),
        "similarity": similarity,
        "mix_ratio": float(mix_ratio),
        "min_warm_interactions": int(min_warm_interactions),
        "reliable_warm_count": int(len(reliable_warm)),
        "pool_count": int(len(base_pool)),
    }

    if not base_pool:
        return [], info
    if sampler == "all" or sample_size <= 0 or sample_size >= len(base_pool):
        return list(base_pool), info

    popularity_rank = sorted(
        base_pool,
        key=lambda idx: (-int(item_frequencies.get(int(idx), 0)), int(idx)),
    )

    if sampler == "popular":
        return popularity_rank[:sample_size], info

    similarity_rank_full = _rank_warm_items_by_cold_similarity(state, similarity)
    base_pool_set = set(base_pool)
    similarity_rank = [idx for idx in similarity_rank_full if idx in base_pool_set]
    if not similarity_rank:
        similarity_rank = popularity_rank

    if sampler == "closest_to_cold":
        return similarity_rank[:sample_size], info

    popular_take = int(round(sample_size * mix_ratio))
    popular_take = max(0, min(popular_take, sample_size))
    similarity_take = max(0, sample_size - popular_take)
    selected = _dedupe_preserve_order(
        popularity_rank[:popular_take] + similarity_rank[:similarity_take]
    )
    if len(selected) < sample_size:
        fallback_rank = _dedupe_preserve_order(popularity_rank + similarity_rank + base_pool)
        for item_idx in fallback_rank:
            if item_idx in selected:
                continue
            selected.append(int(item_idx))
            if len(selected) >= sample_size:
                break
    return selected[:sample_size], info


def _fit_or_get_content_projector_bundle(state: dict, config: ExperimentConfig) -> dict | None:
    projector_bundle = state.get("content_projector_bundle")
    if projector_bundle is not None:
        return projector_bundle

    reference_items = state.get('warm_items')
    if not reference_items:
        reference_items = list(state['idx_to_item'].keys())

    projector_bundle = _fit_content_embedding_projector(
        item_embeddings=state['item_embeddings'],
        idx_to_item=state['idx_to_item'],
        reference_item_indices=reference_items,
        output_dim=config.model.num_items_hidden,
    )
    state['content_projector_bundle'] = projector_bundle
    return projector_bundle


def _initialize_model_item_embeddings(
    model,
    state: dict,
    config: ExperimentConfig,
    device: torch.device,
    item_indices: list[int] | None = None,
) -> int:
    projector_bundle = _fit_or_get_content_projector_bundle(state, config)
    target_indices = item_indices if item_indices is not None else list(state['idx_to_item'].keys())
    max_item_idx = _get_model_catalog_size(model)

    initialized = 0
    with torch.no_grad():
        for item_idx in target_indices:
            item_idx = int(item_idx)
            if item_idx <= 0 or item_idx > max_item_idx:
                continue
            item_id = state['idx_to_item'].get(item_idx)
            if item_id not in state['item_embeddings']:
                continue
            projected = _project_content_embedding(
                state['item_embeddings'][item_id],
                projector_bundle=projector_bundle,
                output_dim=config.model.num_items_hidden,
            )
            model.item_emb.weight[item_idx].copy_(
                torch.tensor(projected, dtype=model.item_emb.weight.dtype, device=device)
            )
            initialized += 1

        model.item_emb.weight[0].zero_()
        if hasattr(model, "delta"):
            model.delta[0].zero_()

    return initialized


def _extend_model_with_cold_embeddings(
    source_model,
    state: dict,
    config: ExperimentConfig,
    device: torch.device,
):
    target_num_items = int(state.get("num_total_items", state.get("num_items", 0)))
    source_num_items = _get_model_catalog_size(source_model)
    if target_num_items <= source_num_items:
        return source_model

    delta_cls = getattr(models, "SASRecWithTrainableDelta", None)
    is_delta_model = delta_cls is not None and isinstance(source_model, delta_cls)
    if is_delta_model:
        target_model = delta_cls(
            num_items=target_num_items,
            hidden_units=source_model.hidden_units,
            num_blocks=source_model.num_blocks,
            num_heads=source_model.num_heads,
            dropout_rate=source_model.dropout_rate,
            max_len=source_model.max_len,
            apply_to_all_items=True,
            max_delta_norm=float(getattr(source_model, "max_delta_norm", 0.5)),
        ).to(device)
    else:
        target_model = models.SASRec(
            num_items=target_num_items,
            hidden_units=source_model.hidden_units,
            num_blocks=source_model.num_blocks,
            num_heads=source_model.num_heads,
            dropout_rate=source_model.dropout_rate,
            max_len=source_model.max_len,
        ).to(device)

    state_dict = {
        key: value
        for key, value in source_model.state_dict().items()
        if key not in {"item_emb.weight", "delta", "_delta_mask"}
    }
    target_model.load_state_dict(state_dict, strict=False)

    projector_bundle = _fit_or_get_content_projector_bundle(state, config)
    with torch.no_grad():
        target_model.item_emb.weight[: source_num_items + 1].copy_(source_model.item_emb.weight)
        if is_delta_model and hasattr(target_model, "delta"):
            target_model.delta.zero_()
            target_model.delta[: source_num_items + 1].copy_(source_model.delta)

        for item_idx, item_id in state['idx_to_item'].items():
            item_idx = int(item_idx)
            if item_idx <= source_num_items or item_idx > target_num_items:
                continue
            if item_id not in state['item_embeddings']:
                continue
            projected = _project_content_embedding(
                state['item_embeddings'][item_id],
                projector_bundle=projector_bundle,
                output_dim=config.model.num_items_hidden,
            )
            target_model.item_emb.weight[item_idx].copy_(
                torch.tensor(projected, dtype=target_model.item_emb.weight.dtype, device=device)
            )

        target_model.item_emb.weight[0].zero_()
        if is_delta_model and hasattr(target_model, "delta"):
            target_model.delta[0].zero_()
            target_model.constrain_delta_()

    return target_model


def _build_official_eval_sequences(
    test_inputs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    filter_cold_history: bool,
    warm_item_indices: list[int],
    excluded_item_indices: list[int] | None = None,
) -> list[list[int]]:
    """Build per-user sequences [history..., target] from official split tables.

    Unlike prepare_sequences(), this keeps users even with empty history
    (sequence length == 1), which is closer to the paper evaluation protocol.
    """
    test_inputs = test_inputs_df.copy()
    excluded_set = {int(v) for v in (excluded_item_indices or []) if int(v) > 0}
    if excluded_set and "item_id_encoded" in test_inputs.columns:
        test_inputs = test_inputs[~test_inputs["item_id_encoded"].isin(excluded_set)].copy()
    if filter_cold_history:
        if "is_cold" in test_inputs.columns:
            test_inputs = test_inputs[~test_inputs["is_cold"].astype(bool)].copy()
        else:
            warm_set = {int(v) for v in warm_item_indices if int(v) > 0}
            test_inputs = test_inputs[test_inputs["item_id_encoded"].isin(warm_set)].copy()

    if not test_inputs.empty:
        test_inputs = test_inputs.sort_values(["user_id", "timestamp"])
        history_map = (
            test_inputs.groupby("user_id")["item_id_encoded"]
            .apply(lambda s: [int(v) for v in s.tolist() if int(v) > 0])
            .to_dict()
        )
    else:
        history_map = {}

    gt = ground_truth_df.copy()
    if excluded_set and "item_id_encoded" in gt.columns:
        gt = gt[~gt["item_id_encoded"].isin(excluded_set)].copy()
    if "timestamp" in gt.columns:
        gt = gt.sort_values(["user_id", "timestamp"])
    else:
        gt = gt.sort_values(["user_id"])

    sequences: list[list[int]] = []
    for row in gt.itertuples(index=False):
        user_id = getattr(row, "user_id")
        target = int(getattr(row, "item_id_encoded"))
        if target <= 0:
            continue
        history = list(history_map.get(user_id, []))
        sequences.append(history + [target])
    return sequences


def _build_eval_loader_from_sequences(
    sequences: list[list[int]],
    batch_size: int,
    max_len: int,
):
    dataset = models.SequentialDataset(sequences, max_len=max_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _evaluate_validation_ndcg(
    model,
    state: dict,
    device: torch.device,
    config: ExperimentConfig,
) -> float | None:
    val_loader = state.get("val_loader")
    if val_loader is None:
        return None
    _, ndcg = training.evaluate(
        model,
        val_loader,
        device,
        _get_model_catalog_size(model),
        k=10,
    )
    return float(ndcg)


def _maybe_restore_best_state(model, best_state: dict | None) -> None:
    if best_state is not None:
        model.load_state_dict(best_state)


def _adapt_selected_item_embeddings(
    model,
    train_loader,
    device: torch.device,
    num_items: int,
    target_item_indices: list[int],
    epochs: int,
    lr: float,
    excluded_item_ids: list[int] | None = None,
) -> None:
    if epochs <= 0:
        return

    target_indices = sorted({int(idx) for idx in target_item_indices if 0 < int(idx) <= int(num_items)})
    if not target_indices:
        return
    excluded_item_set = {
        int(v) for v in (excluded_item_ids or []) if 0 < int(v) <= int(num_items)
    }
    if excluded_item_set and set(target_indices).issubset(excluded_item_set):
        print(
            "⚠ Адаптация выбранных item embeddings пропущена: все target ids исключены "
            "из train-loss для текущего протокола."
        )
        return

    requires_grad_backup = {id(param): param.requires_grad for param in model.parameters()}
    for param in model.parameters():
        param.requires_grad = False
    model.item_emb.weight.requires_grad = True

    optimizer = torch.optim.Adam([model.item_emb.weight], lr=lr)
    target_index_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
    cold_mask = torch.zeros((num_items + 1, 1), dtype=model.item_emb.weight.dtype, device=device)
    cold_mask[target_index_tensor] = 1.0
    excluded_idx_tensor = None
    if excluded_item_ids:
        excluded_idx = sorted({int(v) - 1 for v in excluded_item_ids if 0 < int(v) <= int(num_items)})
        if excluded_idx:
            excluded_idx_tensor = torch.tensor(excluded_idx, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for seq, pos in train_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            optimizer.zero_grad(set_to_none=True)

            seq_emb = model(seq, pos)
            last_emb = seq_emb[:, -1, :]
            pos_items = seq[:, -1]
            scores = torch.matmul(last_emb, model.item_emb.weight[1:].t())
            if excluded_idx_tensor is not None:
                pos_idx = (pos_items - 1).clamp(min=0, max=int(num_items) - 1)
                pos_scores = scores.gather(1, pos_idx.unsqueeze(1))
                scores = scores.index_fill(dim=1, index=excluded_idx_tensor, value=float("-inf"))
                scores = scores.scatter(1, pos_idx.unsqueeze(1), pos_scores)
            loss = torch.nn.functional.cross_entropy(scores, pos_items - 1)

            loss.backward()
            if model.item_emb.weight.grad is not None:
                model.item_emb.weight.grad.mul_(cold_mask)
            optimizer.step()

            with torch.no_grad():
                cold_weights = model.item_emb.weight[target_index_tensor]
                model.item_emb.weight[target_index_tensor] = torch.nn.functional.normalize(cold_weights, p=2, dim=1)

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Эпоха адаптации item embeddings {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    for param in model.parameters():
        param.requires_grad = requires_grad_backup[id(param)]


def _select_pairwise_source_model(state: dict, preference: str | None = None):
    pref = str(preference or "model_with_embeddings").strip().lower()

    if pref in {"model_with_embeddings", "embeddings", "content"}:
        ordered_keys = ["model_with_embeddings", "model_with_implicit_slim"]
    elif pref in {"model_with_implicit_slim", "implicit", "implicit_slim"}:
        ordered_keys = ["model_with_implicit_slim", "model_with_embeddings"]
    else:
        ordered_keys = ["model_with_embeddings", "model_with_implicit_slim"]

    for key in ordered_keys:
        if key in state:
            if key == "model_with_embeddings":
                return state[key], key, "контекстных эмбеддингов"
            if key == "model_with_implicit_slim":
                return state[key], key, "ImplicitSLIM"

    raise KeyError(
        "Не найдена базовая модель для pairwise-этапа. "
        "Ожидалась `model_with_embeddings` (или `model_with_implicit_slim`). "
        "Запустите run_embedding_training и run_implicit_slim_step."
    )


def run_data_pipeline(config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    data, metadata, item_embeddings = data_mod.load_data_bundle(config)
    out = {
        'data': data,
        'metadata': metadata,
        'item_embeddings': item_embeddings,
    }
    if isinstance(metadata, dict) and "letitgo_bundle" in metadata:
        out["letitgo_bundle"] = metadata["letitgo_bundle"]
    return out


def run_encoding_pipeline(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    data = state['data']

    if config.data.use_letitgo_official_splits and 'letitgo_bundle' in state:
        data_encoded, item_to_idx, idx_to_item, user_to_idx, idx_to_user, num_items, num_users = (
            data_mod.encode_preencoded_interactions(data)
        )
        bundle = copy.deepcopy(state['letitgo_bundle'])
        for split_key in ('train_df', 'val_df', 'test_inputs_df', 'ground_truth_df'):
            if split_key in bundle:
                bundle[split_key] = data_mod.apply_encoded_columns_to_split(bundle[split_key], user_to_idx)
        state['letitgo_bundle'] = bundle
    else:
        data_encoded, item_to_idx, idx_to_item, user_to_idx, idx_to_user, num_items, num_users = (
            data_mod.encode_interactions(data)
        )

    return {
        **state,
        'data_encoded': data_encoded,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'num_items': num_items,
        'num_users': num_users,
    }


def run_split_and_sequences(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    paper_eval_sequences_full = None
    paper_eval_sequences_warm_history = None
    train_excluded_item_ids: list[int] = []
    val_loader = None
    discarded_items: list[int] = []

    if config.data.use_letitgo_official_splits and 'letitgo_bundle' in state:
        bundle = state['letitgo_bundle']
        train_df = bundle['train_df'].copy()
        val_df = bundle['val_df'].copy()
        test_inputs_df = bundle['test_inputs_df'].copy()
        ground_truth_df = bundle['ground_truth_df'].copy()
        raw_test_df = data_mod.build_test_dataframe_from_inputs_and_ground_truth(
            test_inputs_df,
            ground_truth_df,
        )
        warm_hint = _sorted_positive_item_ids(bundle.get('warm_items', []))
        if not warm_hint:
            warm_hint = _sorted_positive_item_ids(train_df['item_id_encoded'].unique().tolist())
        cold_hint = _sorted_positive_item_ids(bundle.get('cold_items', []))
        if not cold_hint:
            warm_set = set(warm_hint)
            cold_hint = sorted(
                {
                    int(v)
                    for v in raw_test_df['item_id_encoded'].unique().tolist()
                    if int(v) > 0 and int(v) not in warm_set
                }
            )

        warm_items, cold_items, discarded_items = _derive_item_role_sets(
            train_df=train_df,
            eval_df=raw_test_df,
            role_mode=config.data.item_role_mode,
            role_k=config.data.item_role_k,
            cold_threshold=config.data.cold_threshold,
            cold_fraction=config.data.cold_fraction,
            warm_hint=warm_hint,
            cold_hint=cold_hint,
        )

        if discarded_items:
            train_df = _filter_df_by_item_ids(train_df, discarded_items)
            val_df = _filter_df_by_item_ids(val_df, discarded_items)
            test_inputs_df = _filter_df_by_item_ids(test_inputs_df, discarded_items)
            ground_truth_df = _filter_df_by_item_ids(ground_truth_df, discarded_items)
        test_df = data_mod.build_test_dataframe_from_inputs_and_ground_truth(
            test_inputs_df,
            ground_truth_df,
        )

        train_sequences = models.prepare_sequences(train_df, 'user_id', 'item_id_encoded', 'timestamp')
        val_sequences = models.prepare_sequences(val_df, 'user_id', 'item_id_encoded', 'timestamp')
        test_sequences = models.prepare_sequences(test_df, 'user_id', 'item_id_encoded', 'timestamp')

        train_loader, test_loader = training.build_dataloaders(
            train_sequences,
            test_sequences,
            batch_size=config.model.batch_size,
            max_len=config.model.max_len,
        )
        if val_sequences:
            val_loader = _build_eval_loader_from_sequences(
                sequences=val_sequences,
                batch_size=int(config.model.batch_size),
                max_len=int(config.model.max_len),
            )

        # For official splits cold items are unseen in train and must not be
        # suppressed by train-time negative gradients. In strict mode also
        # exclude discarded tail items from train negatives.
        train_excluded_item_ids = _sorted_positive_item_ids(list(cold_items) + list(discarded_items))

        paper_eval_sequences_full = _build_official_eval_sequences(
            test_inputs_df=test_inputs_df,
            ground_truth_df=ground_truth_df,
            filter_cold_history=False,
            warm_item_indices=warm_items,
            excluded_item_indices=discarded_items,
        )
        paper_eval_sequences_warm_history = _build_official_eval_sequences(
            test_inputs_df=test_inputs_df,
            ground_truth_df=ground_truth_df,
            filter_cold_history=True,
            warm_item_indices=warm_items,
            excluded_item_indices=discarded_items,
        )
    else:
        data_encoded = state['data_encoded']
        train_df, raw_test_df = data_mod.split_train_test_by_time(data_encoded, ratio=0.8)
        warm_items, cold_items, discarded_items = _derive_item_role_sets(
            train_df=train_df,
            eval_df=raw_test_df,
            role_mode=config.data.item_role_mode,
            role_k=config.data.item_role_k,
            cold_threshold=config.data.cold_threshold,
            cold_fraction=config.data.cold_fraction,
        )
        if discarded_items:
            train_df = _filter_df_by_item_ids(train_df, discarded_items)
            test_df = _filter_df_by_item_ids(raw_test_df, discarded_items)
        else:
            test_df = raw_test_df

        train_sequences = models.prepare_sequences(train_df, 'user_id', 'item_id_encoded', 'timestamp')
        test_sequences = models.prepare_sequences(test_df, 'user_id', 'item_id_encoded', 'timestamp')

        train_loader, test_loader = training.build_dataloaders(
            train_sequences,
            test_sequences,
            batch_size=config.model.batch_size,
            max_len=config.model.max_len,
        )
        if str(config.data.item_role_mode).strip().lower() == "strict_zero_vs_gt_k":
            train_excluded_item_ids = _sorted_positive_item_ids(list(cold_items) + list(discarded_items))

    cold_items = sorted(set(cold_items), key=int)
    warm_items = sorted(set(warm_items), key=int)
    discarded_items = sorted(set(discarded_items), key=int)
    num_total_items = int(state.get('num_items', 0))
    num_warm_items = int(max(warm_items)) if warm_items else int(num_total_items)

    state.update({
        'train_df': train_df,
        'test_df': test_df,
        'train_sequences': train_sequences,
        'test_sequences': test_sequences,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'cold_items': cold_items,
        'warm_items': warm_items,
        'discarded_items': discarded_items,
        'num_total_items': num_total_items,
        'num_warm_items': num_warm_items,
        'paper_eval_sequences_full': paper_eval_sequences_full,
        'paper_eval_sequences_warm_history': paper_eval_sequences_warm_history,
        'train_excluded_item_ids': train_excluded_item_ids,
    })
    return state


def run_baseline_training(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    device = _resolve_device()
    state['device'] = device

    model_baseline = models.SASRec(
        num_items=state['num_items'],
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model_baseline.parameters(), lr=config.model.lr)
    for epoch in range(config.model.epochs):
        loss = training.train_epoch(
            model_baseline,
            state['train_loader'],
            optimizer,
            device,
            state['num_items'],
            excluded_item_ids=_get_training_excluded_item_ids(state),
        )
        print(f"Эпоха {epoch + 1}/{config.model.epochs}, Loss: {loss:.4f}")

    hit_rate, ndcg = training.evaluate(
        model_baseline,
        state['test_loader'],
        device,
        state['num_items'],
        k=10,
    )
    print(f"HitRate@10: {hit_rate:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")

    state['model_baseline'] = model_baseline
    state['metrics_baseline'] = {'HitRate': hit_rate, 'NDCG': ndcg}
    return state


def _fit_content_embedding_projector(
    item_embeddings: dict,
    idx_to_item: dict[int, int],
    reference_item_indices: list[int],
    output_dim: int,
):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    reference_vectors: list[np.ndarray] = []
    for item_idx in reference_item_indices:
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        reference_vectors.append(vec)

    if not reference_vectors:
        return None

    X = np.asarray(reference_vectors, dtype=np.float32)
    input_dim = int(X.shape[1])
    if input_dim == int(output_dim):
        projector = Pipeline([("normalizer", Normalizer())])
    else:
        n_components = min(int(output_dim), int(X.shape[0]), input_dim)
        projector = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=42)),
                ("normalizer", Normalizer()),
            ]
        )
    projector.fit(X)
    return {"projector": projector, "output_dim": int(output_dim)}


def _project_content_embedding(
    vec: np.ndarray,
    projector_bundle,
    output_dim: int,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if projector_bundle is not None:
        projected = projector_bundle["projector"].transform(arr).astype(np.float32)[0]
    else:
        projected = arr.astype(np.float32)[0]

    if projected.shape[0] < int(output_dim):
        projected = np.pad(projected, (0, int(output_dim) - projected.shape[0]))
    elif projected.shape[0] > int(output_dim):
        projected = projected[: int(output_dim)]

    norm = float(np.linalg.norm(projected))
    if norm > 0:
        projected = projected / norm
    return np.asarray(projected, dtype=np.float32)


def run_embedding_training(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    device = state.get('device', _resolve_device())
    state['device'] = device
    use_official_protocol = bool(config.data.use_letitgo_official_splits)
    model_num_items = (
        int(state.get('num_warm_items', _get_warm_catalog_size(state)))
        if use_official_protocol
        else int(state['num_items'])
    )

    model_with_embeddings = models.SASRec(
        num_items=model_num_items,
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
    ).to(device)

    state['content_projector_bundle'] = _fit_or_get_content_projector_bundle(
        state=state,
        config=config,
    )
    initialized = _initialize_model_item_embeddings(
        model=model_with_embeddings,
        state=state,
        config=config,
        device=device,
        item_indices=state.get('warm_items') if use_official_protocol else None,
    )

    print(f"✓ Инициализировано {initialized} item embeddings контекстными эмбеддингами")

    optimizer_emb = torch.optim.Adam(model_with_embeddings.parameters(), lr=config.model.lr)
    best_val_ndcg = float("-inf")
    best_state = None
    patience = max(int(getattr(config.model, "patience", 5)), 1)
    min_delta = max(float(getattr(config.model, "min_delta", 1e-4)), 0.0)
    epochs_without_improvement = 0

    for epoch in range(config.model.epochs):
        loss = training.train_epoch(
            model_with_embeddings,
            state['train_loader'],
            optimizer_emb,
            device,
            _get_model_catalog_size(model_with_embeddings),
            excluded_item_ids=_get_training_excluded_item_ids(state),
        )
        print(f"Эпоха {epoch + 1}/{config.model.epochs}, Loss: {loss:.4f}")

        val_ndcg = _evaluate_validation_ndcg(model_with_embeddings, state, device, config)
        if val_ndcg is not None:
            print(f"Val NDCG@10: {val_ndcg:.4f}")
            if val_ndcg > (best_val_ndcg + min_delta):
                best_val_ndcg = float(val_ndcg)
                best_state = copy.deepcopy(model_with_embeddings.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"Ранняя остановка base-model на эпохе {epoch + 1} "
                        f"(best_val_ndcg={best_val_ndcg:.4f}, patience={patience})"
                    )
                    break

    _maybe_restore_best_state(model_with_embeddings, best_state)
    if use_official_protocol:
        state['model_with_embeddings_warm'] = model_with_embeddings
        model_with_embeddings = _extend_model_with_cold_embeddings(
            source_model=model_with_embeddings,
            state=state,
            config=config,
            device=device,
        )

    hit_rate_emb, ndcg_emb = training.evaluate(
        model_with_embeddings,
        state['test_loader'],
        device,
        _get_model_catalog_size(model_with_embeddings),
        k=10,
    )
    print(f"HitRate@10: {hit_rate_emb:.4f}")
    print(f"NDCG@10: {ndcg_emb:.4f}")

    state['model_with_embeddings'] = model_with_embeddings
    state['metrics_with_embeddings'] = {'HitRate': hit_rate_emb, 'NDCG': ndcg_emb}
    return state


def run_implicit_slim_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    device = state.get('device', _resolve_device())
    state['device'] = device

    embed_dim = len(next(iter(state['item_embeddings'].values())))
    x_ctx = pairwise.build_context_matrix(
        state['item_embeddings'],
        state['idx_to_item'],
        embed_dim,
        state['num_items'],
    )
    X = x_ctx

    if config.align.use_interaction_features:
        interaction_X = pairwise.build_interaction_features_sparse(
            state['train_df'],
            num_users=state['num_users'],
            num_items=state['num_items'],
            user_col='user_id_encoded',
            item_col='item_id_encoded',
            n_components=config.align.interaction_dim,
        )
        X = np.vstack([x_ctx, interaction_X])

    W = state['model_with_embeddings'].item_emb.weight[1:].detach().cpu().numpy().T
    fused = pairwise.implicit_slim(W, X, lam=config.align.lam, alpha=config.align.alpha)
    fused = fused / (np.linalg.norm(fused, axis=0, keepdims=True) + 1e-8)

    model_with_implicit_slim = models.SASRec(
        num_items=state['num_items'],
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
    ).to(device)
    model_with_implicit_slim.load_state_dict(state['model_with_embeddings'].state_dict())
    with torch.no_grad():
        model_with_implicit_slim.item_emb.weight[1:] = torch.tensor(
            fused.T,
            dtype=torch.float32,
            device=device,
        )

    hit_rate_implicit, ndcg_implicit = training.evaluate(
        model_with_implicit_slim,
        state['test_loader'],
        device,
        state['num_items'],
        k=10,
    )

    state['model_with_implicit_slim'] = model_with_implicit_slim
    state['metrics_with_implicit_slim'] = {
        'HitRate': hit_rate_implicit,
        'NDCG': ndcg_implicit,
    }
    return state


def run_let_it_go_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    device = state.get('device', _resolve_device())
    state['device'] = device

    if 'cold_items' not in state:
        raise RuntimeError("Не найден список cold_items. Сначала выполните этап split_and_sequences.")

    model_cls = getattr(models, 'SASRecWithTrainableDelta', None)
    if model_cls is None:
        raise ImportError(
            "Модель SASRecWithTrainableDelta не найдена в модуле models. "
            "Перезапустите ядро и выполните заново ячейку импорта с полной перезагрузкой модуля."
        )

    delta_all_items = bool(config.data.use_letitgo_official_splits)
    model_num_items = (
        int(state.get('num_warm_items', _get_warm_catalog_size(state)))
        if delta_all_items
        else int(state['num_items'])
    )
    model_with_trainable_delta = model_cls(
        num_items=model_num_items,
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
        cold_item_indices=None if delta_all_items else state['cold_items'],
        apply_to_all_items=delta_all_items,
        max_delta_norm=float(getattr(config.model, "letitgo_max_delta_norm", 0.5)),
    ).to(device)

    if delta_all_items:
        state['content_projector_bundle'] = _fit_or_get_content_projector_bundle(
            state=state,
            config=config,
        )
        initialized = _initialize_model_item_embeddings(
            model=model_with_trainable_delta,
            state=state,
            config=config,
            device=device,
            item_indices=state.get('warm_items'),
        )
        print(f"✓ E0 warm-only init: {initialized} warm item embeddings")

        # Official let-it-go protocol: warm pretrained embeddings are frozen,
        # transformer/backbone parameters and delta vectors remain trainable.
        for param in model_with_trainable_delta.parameters():
            param.requires_grad = True
        model_with_trainable_delta.item_emb.weight.requires_grad = False
        optimizer_params = [param for param in model_with_trainable_delta.parameters() if param.requires_grad]
        optimizer_delta = torch.optim.Adam(optimizer_params, lr=config.model.lr)
    else:
        model_with_trainable_delta.load_state_dict(state['model_with_embeddings'].state_dict(), strict=False)

        # Legacy/non-official path: keep the previous lightweight delta-only fine-tuning.
        for param in model_with_trainable_delta.parameters():
            param.requires_grad = False
        model_with_trainable_delta.delta.requires_grad = True
        optimizer_delta = torch.optim.Adam([model_with_trainable_delta.delta], lr=config.model.lr)

    letitgo_epochs = max(int(config.model.letitgo_epochs), 1)
    letitgo_patience = max(int(getattr(config.model, "letitgo_patience", 4)), 1)
    letitgo_min_delta = max(float(getattr(config.model, "letitgo_min_delta", 1e-4)), 0.0)

    best_loss = float("inf")
    epochs_without_improvement = 0
    best_val_ndcg = float("-inf")
    best_state = None
    use_validation = state.get("val_loader") is not None

    for epoch in range(letitgo_epochs):
        loss = training.train_epoch(
            model_with_trainable_delta,
            state['train_loader'],
            optimizer_delta,
            device,
            _get_model_catalog_size(model_with_trainable_delta),
            excluded_item_ids=_get_training_excluded_item_ids(state),
        )
        print(f"Этап let-it-go: эпоха {epoch + 1}/{letitgo_epochs}, Loss: {loss:.4f}")

        if use_validation:
            val_ndcg = _evaluate_validation_ndcg(model_with_trainable_delta, state, device, config)
            print(f"Val NDCG@10 (let-it-go): {val_ndcg:.4f}")
            if val_ndcg is not None and val_ndcg > (best_val_ndcg + letitgo_min_delta):
                best_val_ndcg = float(val_ndcg)
                best_state = copy.deepcopy(model_with_trainable_delta.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= letitgo_patience:
                    print(
                        f"Ранняя остановка let-it-go на эпохе {epoch + 1} "
                        f"(best_val_ndcg={best_val_ndcg:.4f}, patience={letitgo_patience})"
                    )
                    break
        else:
            current_loss = float(loss)
            if current_loss < (best_loss - letitgo_min_delta):
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= letitgo_patience:
                    print(
                        f"Ранняя остановка let-it-go на эпохе {epoch + 1} "
                        f"(best_loss={best_loss:.4f}, patience={letitgo_patience})"
                    )
                    break

    _maybe_restore_best_state(model_with_trainable_delta, best_state)
    if delta_all_items:
        state['model_with_trainable_delta_warm'] = model_with_trainable_delta
        model_with_trainable_delta = _extend_model_with_cold_embeddings(
            source_model=model_with_trainable_delta,
            state=state,
            config=config,
            device=device,
        )

    hit_rate_delta, ndcg_delta = training.evaluate(
        model_with_trainable_delta,
        state['test_loader'],
        device,
        _get_model_catalog_size(model_with_trainable_delta),
        k=10,
    )
    print(f"HitRate@10 (let-it-go): {hit_rate_delta:.4f}")
    print(f"NDCG@10 (let-it-go): {ndcg_delta:.4f}")

    state['model_with_trainable_delta'] = model_with_trainable_delta
    state['metrics_with_trainable_delta'] = {'HitRate': hit_rate_delta, 'NDCG': ndcg_delta}
    return state


def run_pairwise_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    source_model, source_key, source_label = _select_pairwise_source_model(
        state,
        preference=config.align.pairwise_source_model_preference,
    )
    print(f"Pairwise-regularization использует базовую модель: {source_label}")

    similarity_matrix, embedding_to_idx = pairwise.compute_textual_similarity_matrix(
        state['item_embeddings'],
        state['item_to_idx'],
        top_k=config.align.similarity_top_k,
        use_sparse=False,
    )
    state['similarity_matrix'] = similarity_matrix
    state['embedding_to_idx'] = embedding_to_idx

    # Copy and regularize both source model and final pairwise model
    model_for_pairwise = models.SASRec(
        num_items=state['num_items'],
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
    ).to(state['device'])
    model_for_pairwise.load_state_dict(source_model.state_dict())

    pairwise.regularize_cold_items_embeddings(
        model_for_pairwise,
        similarity_matrix,
        state['cold_items'],
        state['warm_items'],
        alpha=config.align.pairwise_warm_regularization,
        embedding_to_idx=embedding_to_idx,
        idx_to_item=state['idx_to_item'],
    )

    state['model_with_pairwise'] = model_for_pairwise
    state['pairwise_source_model_key'] = source_key
    state['pairwise_source_label'] = source_label
    return state


def run_pairwise_transformer_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)
    device = state.get('device', _resolve_device())
    state['device'] = device

    required_state_keys = [
        'item_embeddings',
        'model_with_embeddings',
        'idx_to_item',
        'cold_items',
        'warm_items',
        'num_items',
    ]
    missing_keys = [key for key in required_state_keys if key not in state]
    if missing_keys:
        raise KeyError(
            "run_pairwise_transformer_step требует полного состояния пайплайна. "
            f"Отсутствуют поля: {missing_keys}. Проверьте порядок выполнения: "
            "run_data_pipeline -> run_encoding_pipeline -> run_split_and_sequences -> "
            "run_embedding_training."
        )

    source_model, source_key, source_label = _select_pairwise_source_model(
        state,
        preference=config.align.pairwise_source_model_preference,
    )
    print(f"Pairwise-Transformer использует базовую модель: {source_label}")

    content_dim = len(next(iter(state['item_embeddings'].values())))
    item_frequencies = _collect_item_interaction_frequencies(state.get('train_df'))
    content_projector_bundle = _fit_or_get_content_projector_bundle(state, config)
    warm_for_mapper, warm_sampler_info = _select_warm_items_for_mapper(
        state,
        config,
        item_frequencies=item_frequencies,
    )
    target_mode = str(config.align.pairwise_target_mode).strip().lower()
    infer_scope = str(config.align.pairwise_infer_scope).strip().lower()
    update_mode = "delta_add" if target_mode == "delta_from_content" else "blend"
    prediction_mode = _resolve_pairwise_prediction_mode(target_mode, update_mode)

    mapper = pairwise.train_content_to_collab_transformer(
        model_for_projection=source_model,
        item_embeddings=state['item_embeddings'],
        idx_to_item=state['idx_to_item'],
        warm_item_indices=warm_for_mapper,
        content_dim=content_dim,
        device=device,
        hidden_dim=config.model.num_items_hidden,
        n_layers=config.align.pairwise_transformer_layers,
        n_heads=config.align.pairwise_transformer_heads,
        transformer_dim=config.align.pairwise_transformer_hidden_dim,
        epochs=config.align.pairwise_transformer_epochs,
        lr=config.align.pairwise_transformer_lr,
        batch_size=config.align.pairwise_transformer_batch_size,
        item_frequencies=item_frequencies,
        val_fraction=config.align.pairwise_transformer_val_fraction,
        patience=config.align.pairwise_transformer_patience,
        min_delta=config.align.pairwise_transformer_min_delta,
        mse_weight=config.align.pairwise_transformer_loss_mse_weight,
        cosine_weight=config.align.pairwise_transformer_loss_cosine_weight,
        nce_weight=config.align.pairwise_transformer_loss_nce_weight,
        nce_temperature=config.align.pairwise_transformer_nce_temperature,
        sample_weight_power=config.align.pairwise_transformer_sample_weight_power,
        weight_decay=config.align.pairwise_transformer_weight_decay,
        grad_clip=config.align.pairwise_transformer_grad_clip,
        token_count=config.align.pairwise_transformer_token_count,
        dropout=config.align.pairwise_transformer_dropout,
        projector_bundle=content_projector_bundle,
        target_mode=target_mode,
    )

    target_item_indices = _resolve_pairwise_target_item_indices(state, infer_scope)
    predicted_embeddings = pairwise.predict_embeddings_with_mapper(
        mapper,
        item_embeddings=state['item_embeddings'],
        idx_to_item=state['idx_to_item'],
        target_item_indices=target_item_indices,
        device=device,
        batch_size=config.align.pairwise_transformer_batch_size,
        prediction_mode=prediction_mode,
        projector_bundle=content_projector_bundle,
        output_dim=int(config.model.num_items_hidden),
    )

    model_with_pairwise_transformer = models.SASRec(
        num_items=state['num_items'],
        hidden_units=config.model.num_items_hidden,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        max_len=config.model.max_len,
    ).to(device)
    model_with_pairwise_transformer.load_state_dict(source_model.state_dict())

    default_blend_alpha = (
        1.0
        if config.data.use_letitgo_official_splits and update_mode == "blend"
        else float(config.align.pairwise_transformer_blend_alpha)
    )
    blend_alpha = float(np.clip(default_blend_alpha, 0.0, 1.0))
    target_norm = None
    if config.data.use_letitgo_official_splits and prediction_mode == "full":
        target_norm = _mean_item_embedding_norm(model_with_pairwise_transformer, state.get("warm_items", []))
    updated, updated_ids = _apply_predicted_embeddings(
        model_with_pairwise_transformer,
        predicted_embeddings=predicted_embeddings,
        blend_alpha=blend_alpha,
        device=device,
        update_mode=update_mode,
        target_norm=target_norm,
        normalize_output=True,
    )
    updated_cold = _count_item_group_members(updated_ids, state.get("cold_items", []))
    updated_warm = _count_item_group_members(updated_ids, state.get("warm_items", []))

    print(
        f"✓ Transformer-mapper обновил {updated} item embeddings "
        f"(cold={updated_cold}, warm={updated_warm}, blend_alpha={blend_alpha:.2f}, scope={infer_scope})"
    )
    if updated == 0:
        print("⚠ Нет предсказанных item embeddings с доступным контентом; использованы исходные веса.")

    adapt_epochs = int(max(0, config.align.pairwise_transformer_adapt_epochs))
    if adapt_epochs > 0:
        if 'train_loader' not in state:
            print("⚠ train_loader не найден, адаптация после мэппера пропущена.")
        elif updated_ids:
            _adapt_selected_item_embeddings(
                model=model_with_pairwise_transformer,
                train_loader=state['train_loader'],
                device=device,
                num_items=state['num_items'],
                target_item_indices=updated_ids,
                epochs=adapt_epochs,
                lr=float(config.align.pairwise_transformer_adapt_lr),
                excluded_item_ids=_get_training_excluded_item_ids(state),
            )

    state['model_with_pairwise_transformer'] = model_with_pairwise_transformer
    state['pairwise_transformer_mapper'] = mapper
    state['pairwise_transformer_predicted_cold_count'] = updated_cold
    state['pairwise_transformer_predicted_total_count'] = updated
    state['pairwise_transformer_predicted_warm_count'] = updated_warm
    state['pairwise_transformer_used_warm_items'] = len(warm_for_mapper)
    state['pairwise_transformer_warm_sampler_info'] = warm_sampler_info
    state['pairwise_transformer_target_mode'] = target_mode
    state['pairwise_transformer_prediction_mode'] = prediction_mode
    state['pairwise_transformer_infer_scope'] = infer_scope
    state['pairwise_transformer_source_model_key'] = source_key
    state['pairwise_transformer_source_label'] = source_label
    state['pairwise_transformer_metrics'] = None
    return state


def run_cold_evaluation_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)

    required_state_keys = [
        'test_loader',
        'device',
        'num_items',
        'cold_items',
    ]
    missing_base = [key for key in required_state_keys if key not in state]
    if missing_base:
        raise KeyError(
            "run_cold_evaluation_step требует состояния после split и подготовки моделей. "
            f"Отсутствуют поля: {missing_base}."
        )

    model_with_embeddings = state.get('model_with_embeddings')
    if model_with_embeddings is None:
        raise KeyError("Не найдено model_with_embeddings. Запустите run_embedding_training.")
    model_baseline = state.get('model_baseline')

    if model_baseline is not None:
        hit_rate_cold_base, ndcg_cold_base, total_cold = training.evaluate_cold_items(
            model_baseline,
            state['test_loader'],
            state['device'],
            state['num_items'],
            state['cold_items'],
            k=10,
        )
    else:
        hit_rate_cold_base = np.nan
        ndcg_cold_base = np.nan
        total_cold = np.nan

    hit_rate_cold_emb, ndcg_cold_emb, total_cold_emb = training.evaluate_cold_items(
        model_with_embeddings,
        state['test_loader'],
        state['device'],
        state['num_items'],
        state['cold_items'],
        k=10,
    )
    if np.isnan(total_cold):
        total_cold = total_cold_emb

    if 'model_with_implicit_slim' in state:
        hit_rate_cold_implicit, ndcg_cold_implicit, _ = training.evaluate_cold_items(
            state['model_with_implicit_slim'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            state['cold_items'],
            k=10,
        )
        state['hit_rate_cold_implicit'] = hit_rate_cold_implicit
        state['ndcg_cold_implicit'] = ndcg_cold_implicit

    if 'model_with_pairwise' in state:
        hit_rate_cold_pairwise, ndcg_cold_pairwise, _ = training.evaluate_cold_items(
            state['model_with_pairwise'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            state['cold_items'],
            k=10,
        )
    else:
        hit_rate_cold_pairwise, ndcg_cold_pairwise = np.nan, np.nan

    state.update(
        {
            'hit_rate_cold_base': hit_rate_cold_base,
            'ndcg_cold_base': ndcg_cold_base,
            'total_cold': total_cold,
            'hit_rate_cold_emb': hit_rate_cold_emb,
            'ndcg_cold_emb': ndcg_cold_emb,
            'hit_rate_cold_pairwise': hit_rate_cold_pairwise,
            'ndcg_cold_pairwise': ndcg_cold_pairwise,
        }
    )

    if 'model_with_pairwise_transformer' in state:
        hit_rate_cold_pairwise_trans, ndcg_cold_pairwise_trans, _ = training.evaluate_cold_items(
            state['model_with_pairwise_transformer'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            state['cold_items'],
            k=10,
        )
        state['hit_rate_cold_pairwise_transformer'] = hit_rate_cold_pairwise_trans
        state['ndcg_cold_pairwise_transformer'] = ndcg_cold_pairwise_trans

    if 'model_with_trainable_delta' in state:
        hit_rate_cold_delta, ndcg_cold_delta, _ = training.evaluate_cold_items(
            state['model_with_trainable_delta'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            state['cold_items'],
            k=10,
        )
        state['hit_rate_cold_delta'] = hit_rate_cold_delta
        state['ndcg_cold_delta'] = ndcg_cold_delta

    if state['total_cold'] == 0:
        print("⚠ В тестовой выборке не найдено холодных позиций для оценки (total_cold = 0).")
        print("  Проверьте число холодных товаров в split-шаге и критерий холодности.")

    return state


def build_results_step(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)

    if isinstance(state.get("experiment_results"), pd.DataFrame) and not state["experiment_results"].empty:
        exp_df = state["experiment_results"].copy()
        exp_df["experiment_id"] = exp_df["experiment_id"].astype(str).str.upper()

        topk = int(config.registry.topk_values[0]) if config.registry.topk_values else 10
        hit_all_col = f"HitRate@{topk} (все)"
        ndcg_all_col = f"NDCG@{topk} (все)"
        hit_cold_col = f"HitRate@{topk} (холодные)"
        ndcg_cold_col = f"NDCG@{topk} (холодные)"

        required_cols = [hit_all_col, ndcg_all_col, hit_cold_col, ndcg_cold_col]
        missing_metric_cols = [c for c in required_cols if c not in exp_df.columns]
        if missing_metric_cols:
            raise KeyError(
                "experiment_results does not contain expected metric columns for summary: "
                f"{missing_metric_cols}"
            )

        summary_rows = []
        main_df = exp_df[exp_df["experiment_id"] != "E9"]
        for exp_id, group in main_df.groupby("experiment_id", sort=True):
            summary_rows.append(
                {
                    "Метод": group["method_name"].iloc[0],
                    "HitRate@10 (все)": float(group[hit_all_col].mean()),
                    "NDCG@10 (все)": float(group[ndcg_all_col].mean()),
                    "HitRate@10 (холодные)": float(group[hit_cold_col].mean()),
                    "NDCG@10 (холодные)": float(group[ndcg_cold_col].mean()),
                    "ExperimentID": exp_id,
                    "Runs": int(len(group)),
                }
            )

        ablation_df = exp_df[exp_df["experiment_id"] == "E9"]
        if not ablation_df.empty:
            best_idx = ablation_df[ndcg_cold_col].astype(float).idxmax()
            best_row = ablation_df.loc[best_idx]
            summary_rows.append(
                {
                    "Метод": f"{best_row['method_name']} (best ablation)",
                    "HitRate@10 (все)": float(best_row[hit_all_col]),
                    "NDCG@10 (все)": float(best_row[ndcg_all_col]),
                    "HitRate@10 (холодные)": float(best_row[hit_cold_col]),
                    "NDCG@10 (холодные)": float(best_row[ndcg_cold_col]),
                    "ExperimentID": "E9",
                    "Runs": int(len(ablation_df)),
                }
            )

        results_summary = pd.DataFrame(summary_rows)
        if not results_summary.empty:
            exp_order = (
                results_summary["ExperimentID"]
                .astype(str)
                .str.extract(r"E(\d+)", expand=False)
                .astype(float)
                .fillna(9999.0)
            )
            results_summary = (
                results_summary.assign(_exp_order=exp_order)
                .sort_values(["_exp_order", "ExperimentID"])
                .drop(columns=["_exp_order"])
                .reset_index(drop=True)
            )

        print('=' * 80)
        print('СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ')
        print('=' * 80)
        print(results_summary.to_string(index=False))

        state['results_summary'] = results_summary
        state['experiment_results_summary'] = results_summary
        return state

    pairwise_hit = pairwise_ndcg = np.nan
    if 'model_with_pairwise' in state:
        pairwise_hit, pairwise_ndcg = training.evaluate(
            state['model_with_pairwise'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            k=10,
        )

    implicit_hit = implicit_ndcg = np.nan
    if 'model_with_implicit_slim' in state:
        implicit_hit, implicit_ndcg = training.evaluate(
            state['model_with_implicit_slim'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            k=10,
        )

    letitgo_hit = letitgo_ndcg = np.nan
    letitgo_hit_cold = letitgo_ndcg_cold = np.nan
    if 'metrics_with_trainable_delta' in state:
        letitgo_hit = state['metrics_with_trainable_delta'].get('HitRate', np.nan)
        letitgo_ndcg = state['metrics_with_trainable_delta'].get('NDCG', np.nan)
        letitgo_hit_cold = state.get('hit_rate_cold_delta', np.nan)
        letitgo_ndcg_cold = state.get('ndcg_cold_delta', np.nan)

    metrics_baseline = state.get('metrics_baseline', {'HitRate': np.nan, 'NDCG': np.nan})
    metrics_with_embeddings = state.get(
        'metrics_with_embeddings',
        {'HitRate': np.nan, 'NDCG': np.nan},
    )
    hit_rate_cold_base = state.get('hit_rate_cold_base', np.nan)
    ndcg_cold_base = state.get('ndcg_cold_base', np.nan)
    hit_rate_cold_emb = state.get('hit_rate_cold_emb', np.nan)
    ndcg_cold_emb = state.get('ndcg_cold_emb', np.nan)

    results_summary = pd.DataFrame(
        {
            'Метод': [
                'Бейзлайн (SASRec)',
                'С контекстными эмбеддингами',
            ],
            'HitRate@10 (все)': [
                metrics_baseline.get('HitRate', np.nan),
                metrics_with_embeddings.get('HitRate', np.nan),
            ],
            'NDCG@10 (все)': [
                metrics_baseline.get('NDCG', np.nan),
                metrics_with_embeddings.get('NDCG', np.nan),
            ],
            'HitRate@10 (холодные)': [hit_rate_cold_base, hit_rate_cold_emb],
            'NDCG@10 (холодные)': [ndcg_cold_base, ndcg_cold_emb],
        }
    )

    if not np.isnan(letitgo_hit) or not np.isnan(letitgo_hit_cold):
        results_summary = pd.concat(
            [
                results_summary,
                pd.DataFrame(
                    {
                        'Метод': ['SASRec + Trainable Delta (let-it-go)'],
                        'HitRate@10 (все)': [letitgo_hit],
                        'NDCG@10 (все)': [letitgo_ndcg],
                        'HitRate@10 (холодные)': [letitgo_hit_cold],
                        'NDCG@10 (холодные)': [letitgo_ndcg_cold],
                    }
                ),
            ],
            ignore_index=True,
        )

    if not np.isnan(implicit_hit):
        results_summary = pd.concat(
            [
                results_summary,
                pd.DataFrame(
                    {
                        'Метод': ['С контекстными эмбеддингами + ImplicitSLIM'],
                        'HitRate@10 (все)': [implicit_hit],
                        'NDCG@10 (все)': [implicit_ndcg],
                        'HitRate@10 (холодные)': [state.get('hit_rate_cold_implicit', np.nan)],
                        'NDCG@10 (холодные)': [state.get('ndcg_cold_implicit', np.nan)],
                    }
                ),
            ],
            ignore_index=True,
        )

    pairwise_method = 'С попарным выравниванием'
    if state.get('pairwise_source_model_key') == 'model_with_implicit_slim':
        pairwise_method = 'С попарным выравниванием (на базе ImplicitSLIM)'

    pairwise_summary = pd.DataFrame(
        {
            'Метод': [pairwise_method],
            'HitRate@10 (все)': [pairwise_hit],
            'NDCG@10 (все)': [pairwise_ndcg],
            'HitRate@10 (холодные)': [state.get('hit_rate_cold_pairwise', np.nan)],
            'NDCG@10 (холодные)': [state.get('ndcg_cold_pairwise', np.nan)],
        }
    )
    results_summary = pd.concat([results_summary, pairwise_summary], ignore_index=True)

    if 'model_with_pairwise_transformer' in state:
        pairwise_transformer_hit, pairwise_transformer_ndcg = training.evaluate(
            state['model_with_pairwise_transformer'],
            state['test_loader'],
            state['device'],
            state['num_items'],
            k=10,
        )
        pairwise_transformer_method = 'С попарным выравниванием (Transformer)'
        if state.get('pairwise_transformer_source_model_key') == 'model_with_implicit_slim':
            pairwise_transformer_method = 'С попарным выравниванием (Transformer, база ImplicitSLIM)'

        transformed_summary = pd.DataFrame(
            {
                'Метод': [pairwise_transformer_method],
                'HitRate@10 (все)': [pairwise_transformer_hit],
                'NDCG@10 (все)': [pairwise_transformer_ndcg],
                'HitRate@10 (холодные)': [state.get('hit_rate_cold_pairwise_transformer', np.nan)],
                'NDCG@10 (холодные)': [state.get('ndcg_cold_pairwise_transformer', np.nan)],
            }
        )
        results_summary = pd.concat([results_summary, transformed_summary], ignore_index=True)

    print('=' * 80)
    print('СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ')
    print('=' * 80)
    print(results_summary.to_string(index=False))

    state['results_summary'] = results_summary
    return state


def run_seed_stability(state: dict, config: ExperimentConfig | None = None):
    config = _resolve_config(config)
    state = copy.copy(state)

    import random

    results_baseline_all = []
    results_with_embeddings_all = []

    for seed in config.seeds.seeds:
        print(f"\n{'='*50}")
        print(f"Эксперимент с seed={seed}")
        print(f"{'='*50}")

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        model_base = models.SASRec(
            num_items=state['num_items'],
            hidden_units=config.model.num_items_hidden,
            num_blocks=config.model.num_blocks,
            num_heads=config.model.num_heads,
            dropout_rate=config.model.dropout_rate,
            max_len=config.model.max_len,
        ).to(state['device'])
        optimizer_base = torch.optim.Adam(model_base.parameters(), lr=config.model.lr)

        for _ in range(config.seeds.epochs):
            training.train_epoch(
                model_base,
                state['train_loader'],
                optimizer_base,
                state['device'],
                state['num_items'],
                excluded_item_ids=_get_training_excluded_item_ids(state),
            )
        hit_rate_base, ndcg_base = training.evaluate(
            model_base,
            state['test_loader'],
            state['device'],
            state['num_items'],
            k=10,
        )
        results_baseline_all.append({'HitRate': hit_rate_base, 'NDCG': ndcg_base})

        model_emb = models.SASRec(
            num_items=state['num_items'],
            hidden_units=config.model.num_items_hidden,
            num_blocks=config.model.num_blocks,
            num_heads=config.model.num_heads,
            dropout_rate=config.model.dropout_rate,
            max_len=config.model.max_len,
        ).to(state['device'])
        optimizer_emb = torch.optim.Adam(model_emb.parameters(), lr=config.model.lr)

        for _ in range(config.seeds.epochs):
            training.train_epoch(
                model_emb,
                state['train_loader'],
                optimizer_emb,
                state['device'],
                state['num_items'],
                excluded_item_ids=_get_training_excluded_item_ids(state),
            )
        hit_rate_emb, ndcg_emb = training.evaluate(
            model_emb,
            state['test_loader'],
            state['device'],
            state['num_items'],
            k=10,
        )
        results_with_embeddings_all.append({'HitRate': hit_rate_emb, 'NDCG': ndcg_emb})

        print(f"HitRate: {hit_rate_emb:.4f} (с эмбеддингами) vs {hit_rate_base:.4f} (бейзлайн)")
        print(f"NDCG: {ndcg_emb:.4f} (с эмбеддингами) vs {ndcg_base:.4f} (бейзлайн)")

    state['results_baseline_all'] = results_baseline_all
    state['results_with_embeddings_all'] = results_with_embeddings_all
    state['df_base'] = pd.DataFrame(results_baseline_all)
    state['df_emb'] = pd.DataFrame(results_with_embeddings_all)
    return state


_E16_VARIANT_NAMES: dict[str, str] = {
    "E16_CLEAR": "E16 baseline (cold only)",
    "E16_DELTA": "E16 delta target",
    "E16_POPULAR": "E16 popular warm sampler",
    "E16_CLOSEST": "E16 closest-to-cold sampler",
    "E16_MIXED": "E16 mixed warm sampler",
    "E16_ALL": "E16 infer all items",
    "E16_ALL_LOW": "E16 infer all, low warm alpha",
    "E16_ALL_FREQ": "E16 infer all, freq-weighted alpha",
    "E16_CLOSEST_ALL_LOW": "E16 closest + all + low warm alpha",
    "E16_MIXED_ALL_LOW": "E16 mixed + all + low warm alpha",
    "E16_DELTA_MIXED_ALL": "E16 delta + mixed + all",
    "E16_CLOSEST_DELTA_FREQ": "E16 closest + delta + freq-weighted",
}


def _method_name_for_experiment(experiment_id: str) -> str:
    mapping = {
        "E0": "LetItGo baseline",
        "E11": "ImplicitSLIM baseline",
        "E1": "Procrustes alignment",
        "E2": "MLP + MSE",
        "E3": "Transformer + InfoNCE",
        "E3S": "Transformer + InfoNCE (E3* tuned)",
        "E4": "Transformer + Multi-Objective",
        "E5": "Procrustes init + contrastive fine-tune",
        "E6": "Contrastive + structure regularization",
        "E7": "Contrastive + hard negatives",
        "E8": "Confidence-weighted alignment",
        "E10": "MLP + Pairwise Ranking Distillation",
        "E12": "Transformer + Additive Delta",
        "E13": "Transformer + Teacher Residual Delta",
        "E14": "Transformer + Anchor Residual Fusion",
        "E15": "Transformer + Anchor Residual Delta Fusion",
        "E16": "Transformer + Norm-Aware Residual Fusion",
        "E9": "Ablation",
        **_E16_VARIANT_NAMES,
    }
    upper = str(experiment_id).upper()
    if upper in mapping:
        return mapping[upper]
    if upper.startswith("E16_"):
        return f"E16 variant ({upper})"
    return upper


def _validate_research_config(config: ExperimentConfig) -> None:
    topk_values = list(getattr(config.registry, "topk_values", [10]))
    if not topk_values or any(int(k) <= 0 for k in topk_values):
        raise ValueError("registry.topk_values must contain positive integers.")

    if config.align.pairwise_transformer_hard_negative_weight > 0 and config.align.pairwise_transformer_hard_negative_top_k <= 0:
        raise ValueError(
            "hard negative weight is enabled but pairwise_transformer_hard_negative_top_k <= 0."
        )

    if min(
        config.align.pairwise_transformer_loss_mse_weight,
        config.align.pairwise_transformer_loss_cosine_weight,
        config.align.pairwise_transformer_loss_nce_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative.")

    if (
        config.align.pairwise_transformer_loss_mse_weight
        + config.align.pairwise_transformer_loss_cosine_weight
        + config.align.pairwise_transformer_loss_nce_weight
    ) <= 0:
        raise ValueError("At least one alignment loss weight must be positive.")

    if config.ablation.method_id not in {"E1", "E2", "E3", "E3S", "E4", "E5", "E6", "E7", "E8", "E10", "E12", "E13", "E14", "E15", "E16"}:
        raise ValueError("ablation.method_id must be one of E1..E8,E10,E12,E13,E14,E15,E16,E3S.")
    if any(float(alpha) < 0.0 or float(alpha) > 1.0 for alpha in config.ablation.blend_alpha):
        raise ValueError("ablation.blend_alpha values must be in [0, 1].")
    if any(int(epochs) < 0 for epochs in config.ablation.adapt_epochs):
        raise ValueError("ablation.adapt_epochs values must be >= 0.")
    if any(int(rounds) <= 0 for rounds in config.ablation.distill_pair_rounds):
        raise ValueError("ablation.distill_pair_rounds values must be > 0.")
    if any(int(cands) <= 3 for cands in config.ablation.distill_candidate_count):
        raise ValueError("ablation.distill_candidate_count values must be > 3.")
    if any(float(margin) < 0.0 for margin in config.ablation.distill_teacher_margin):
        raise ValueError("ablation.distill_teacher_margin values must be >= 0.")

    source_pref = str(config.align.pairwise_source_model_preference).strip().lower()
    if source_pref not in {
        "model_with_embeddings",
        "embeddings",
        "content",
        "model_with_implicit_slim",
        "implicit",
        "implicit_slim",
        "auto",
    }:
        raise ValueError(
            "align.pairwise_source_model_preference must be one of "
            "{model_with_embeddings, embeddings, content, "
            "model_with_implicit_slim, implicit, implicit_slim, auto}."
        )

    if config.align.pairwise_distill_pair_rounds <= 0:
        raise ValueError("pairwise_distill_pair_rounds must be > 0.")
    if config.align.pairwise_distill_candidate_count <= 3:
        raise ValueError("pairwise_distill_candidate_count must be > 3.")
    if config.align.pairwise_distill_teacher_margin < 0:
        raise ValueError("pairwise_distill_teacher_margin must be >= 0.")

    target_mode = str(config.align.pairwise_target_mode).strip().lower()
    if target_mode not in {"full", "delta_from_content"}:
        raise ValueError("align.pairwise_target_mode must be one of {full, delta_from_content}.")

    warm_sampler = str(config.align.pairwise_warm_sampler).strip().lower()
    if warm_sampler not in {"all", "popular", "closest_to_cold", "mixed"}:
        raise ValueError("align.pairwise_warm_sampler must be one of {all, popular, closest_to_cold, mixed}.")

    warm_similarity = str(config.align.pairwise_warm_similarity).strip().lower()
    if warm_similarity not in {"cosine", "dot"}:
        raise ValueError("align.pairwise_warm_similarity must be one of {cosine, dot}.")

    infer_scope = str(config.align.pairwise_infer_scope).strip().lower()
    if infer_scope not in {"cold", "warm", "all"}:
        raise ValueError("align.pairwise_infer_scope must be one of {cold, warm, all}.")

    if int(config.align.pairwise_warm_sample_size) < 0:
        raise ValueError("align.pairwise_warm_sample_size must be >= 0.")
    if not (0.0 <= float(config.align.pairwise_warm_mix_ratio) <= 1.0):
        raise ValueError("align.pairwise_warm_mix_ratio must be in [0, 1].")

    item_role_mode = str(config.data.item_role_mode).strip().lower()
    if item_role_mode not in {"current", "strict_zero_vs_gt_k"}:
        raise ValueError("data.item_role_mode must be one of {current, strict_zero_vs_gt_k}.")
    if int(config.data.item_role_k) < 0:
        raise ValueError("data.item_role_k must be >= 0.")


def _evaluate_model_multi_k(
    model,
    state: dict,
    topk_values: list[int],
    config: ExperimentConfig | None = None,
) -> dict[str, float]:
    config = _resolve_config(config)
    metrics: dict[str, float] = {}
    model_num_items = _get_model_catalog_size(model)
    discarded_items = _sorted_positive_item_ids(state.get("discarded_items", []))
    all_candidate_items = None
    if discarded_items:
        all_candidate_items = _sorted_positive_item_ids(
            list(state.get("warm_items", [])) + list(state.get("cold_items", []))
        )

    paper_eval_enabled = bool(getattr(config.registry, "paper_eval_enabled", False))
    has_official_sequences = (
        isinstance(state.get("paper_eval_sequences_full"), list)
        and isinstance(state.get("paper_eval_sequences_warm_history"), list)
    )

    if paper_eval_enabled and has_official_sequences:
        primary_recommend_cold = bool(getattr(config.registry, "paper_eval_recommend_cold_items", True))
        primary_filter_cold_history = bool(getattr(config.registry, "paper_eval_filter_cold_history", False))
        report_all_modes = bool(getattr(config.registry, "paper_eval_report_all_modes", False))

        scenarios: list[tuple[bool, bool]] = [(primary_recommend_cold, primary_filter_cold_history)]
        if report_all_modes:
            scenarios = [
                (True, False),
                (True, True),
                (False, False),
                (False, True),
            ]

        primary_total_cold = 0.0
        for recommend_cold_items, filter_cold_history in scenarios:
            eval_sequences = (
                state["paper_eval_sequences_warm_history"]
                if filter_cold_history
                else state["paper_eval_sequences_full"]
            )
            eval_loader = _build_eval_loader_from_sequences(
                sequences=eval_sequences,
                batch_size=int(config.model.batch_size),
                max_len=int(config.model.max_len),
            )
            if recommend_cold_items:
                candidate_item_ids = all_candidate_items
            else:
                candidate_item_ids = state.get("warm_items", [])
            all_metrics_by_k, cold_metrics_by_k, total_cold = training.evaluate_all_and_cold_multi_k(
                model=model,
                dataloader=eval_loader,
                device=state['device'],
                num_items=model_num_items,
                topk_values=topk_values,
                cold_items=state['cold_items'],
                candidate_item_ids=candidate_item_ids,
            )

            if (
                recommend_cold_items == primary_recommend_cold
                and filter_cold_history == primary_filter_cold_history
            ):
                primary_total_cold = float(total_cold)
                for k in sorted(all_metrics_by_k.keys()):
                    hit_all, ndcg_all = all_metrics_by_k[k]
                    hit_cold, ndcg_cold = cold_metrics_by_k[k]
                    metrics[f"HitRate@{k} (все)"] = float(hit_all)
                    metrics[f"NDCG@{k} (все)"] = float(ndcg_all)
                    metrics[f"HitRate@{k} (холодные)"] = float(hit_cold)
                    metrics[f"NDCG@{k} (холодные)"] = float(ndcg_cold)

            if report_all_modes:
                mode_prefix = (
                    f"[paper rc={int(recommend_cold_items)} fc={int(filter_cold_history)}] "
                )
                for k in sorted(all_metrics_by_k.keys()):
                    hit_all, ndcg_all = all_metrics_by_k[k]
                    hit_cold, ndcg_cold = cold_metrics_by_k[k]
                    metrics[f"{mode_prefix}HitRate@{k} (все)"] = float(hit_all)
                    metrics[f"{mode_prefix}NDCG@{k} (все)"] = float(ndcg_all)
                    metrics[f"{mode_prefix}HitRate@{k} (холодные)"] = float(hit_cold)
                    metrics[f"{mode_prefix}NDCG@{k} (холодные)"] = float(ndcg_cold)
                metrics[f"{mode_prefix}TotalColdExamples"] = float(total_cold)

        metrics["TotalColdExamples"] = float(primary_total_cold)
        metrics["paper_eval_recommend_cold_items"] = bool(primary_recommend_cold)
        metrics["paper_eval_filter_cold_history"] = bool(primary_filter_cold_history)
    else:
        all_metrics_by_k, cold_metrics_by_k, total_cold = training.evaluate_all_and_cold_multi_k(
            model=model,
            dataloader=state['test_loader'],
            device=state['device'],
            num_items=model_num_items,
            topk_values=topk_values,
            cold_items=state['cold_items'],
            candidate_item_ids=all_candidate_items,
        )
        for k in sorted(all_metrics_by_k.keys()):
            hit_all, ndcg_all = all_metrics_by_k[k]
            hit_cold, ndcg_cold = cold_metrics_by_k[k]
            metrics[f"HitRate@{k} (все)"] = float(hit_all)
            metrics[f"NDCG@{k} (все)"] = float(ndcg_all)
            metrics[f"HitRate@{k} (холодные)"] = float(hit_cold)
            metrics[f"NDCG@{k} (холодные)"] = float(ndcg_cold)
        metrics["TotalColdExamples"] = float(total_cold)
    return metrics


def _clone_sasrec_model_from(source_model, num_items: int, device: torch.device):
    cloned = models.SASRec(
        num_items=num_items,
        hidden_units=source_model.hidden_units,
        num_blocks=source_model.num_blocks,
        num_heads=source_model.num_heads,
        dropout_rate=source_model.dropout_rate,
        max_len=source_model.max_len,
    ).to(device)
    cloned.load_state_dict(source_model.state_dict())
    return cloned


def _get_model_scoring_embeddings(model) -> torch.Tensor:
    if hasattr(model, "get_item_embeddings_for_scoring"):
        return model.get_item_embeddings_for_scoring()
    return model.item_emb.weight


def _mean_item_embedding_norm(model, item_indices: list[int] | None = None) -> float | None:
    with torch.no_grad():
        weights = _get_model_scoring_embeddings(model)
        if weights.ndim != 2 or weights.size(0) <= 1:
            return None

        if item_indices:
            valid_idx = [int(v) for v in item_indices if 0 < int(v) < weights.size(0)]
            if not valid_idx:
                return None
            index_tensor = torch.tensor(valid_idx, dtype=torch.long, device=weights.device)
            selected = weights.index_select(0, index_tensor)
        else:
            selected = weights[1:]

        norms = selected.norm(dim=1)
        finite = norms[torch.isfinite(norms)]
        if finite.numel() == 0:
            return None
        return float(finite.mean().item())


def _rescale_embedding_to_norm(vec: torch.Tensor, target_norm: float | None) -> torch.Tensor:
    if target_norm is None or not np.isfinite(target_norm) or target_norm <= 0:
        return vec
    vec_norm = vec.norm(p=2)
    if not torch.isfinite(vec_norm) or float(vec_norm.item()) <= 0:
        return vec
    return vec * (float(target_norm) / vec_norm.clamp_min(1e-8))


def _apply_predicted_embeddings(
    target_model,
    predicted_embeddings: dict[int, np.ndarray],
    blend_alpha: float,
    device: torch.device,
    update_mode: str = "blend",
    target_norm: float | None = None,
    normalize_output: bool = True,
    warm_item_indices: set[int] | None = None,
    blend_alpha_warm: float | None = None,
    item_frequencies: dict[int, int] | None = None,
    freq_decay_k: int | None = None,
) -> tuple[int, list[int]]:
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    mode = str(update_mode).strip().lower()
    if mode not in {"blend", "delta_add"}:
        raise ValueError(f"Unsupported embedding update_mode: {update_mode}")
    _warm_set = set(warm_item_indices) if warm_item_indices else set()
    _has_warm_alpha = blend_alpha_warm is not None and len(_warm_set) > 0
    _freq_decay_k = int(freq_decay_k) if freq_decay_k is not None else None
    updated = 0
    updated_ids: list[int] = []
    with torch.no_grad():
        for item_idx, pred_emb in predicted_embeddings.items():
            item_idx = int(item_idx)
            if item_idx <= 0 or item_idx >= target_model.item_emb.weight.size(0):
                continue
            effective_alpha = blend_alpha
            if _has_warm_alpha and item_idx in _warm_set:
                effective_alpha = float(blend_alpha_warm)
                if _freq_decay_k is not None and item_frequencies is not None:
                    freq = max(item_frequencies.get(item_idx, 1), 1)
                    effective_alpha *= min(1.0, _freq_decay_k / freq)
            pred_t = torch.tensor(
                pred_emb,
                dtype=target_model.item_emb.weight.dtype,
                device=device,
            )
            pred_t = _rescale_embedding_to_norm(pred_t, target_norm)
            current = target_model.item_emb.weight[item_idx]
            if mode == "delta_add":
                mixed = current + effective_alpha * pred_t
            else:
                mixed = (1.0 - effective_alpha) * current + effective_alpha * pred_t
            if target_norm is not None:
                mixed = _rescale_embedding_to_norm(mixed, target_norm)
            elif normalize_output:
                mixed = torch.nn.functional.normalize(mixed.unsqueeze(0), p=2, dim=1).squeeze(0)
            elif not torch.isfinite(mixed).all() or float(mixed.norm(p=2).item()) <= 0:
                continue
            target_model.item_emb.weight[item_idx].copy_(mixed)
            updated += 1
            updated_ids.append(int(item_idx))
    return updated, updated_ids


def _predict_anchor_embeddings_from_content_knn(
    item_embeddings: dict,
    idx_to_item: dict[int, int],
    warm_item_indices: list[int],
    target_item_indices: list[int],
    source_model,
    top_k: int = 64,
    temperature: float = 0.07,
    batch_size: int = 512,
    normalize_collab: bool = True,
    normalize_output: bool = True,
    prediction_mode: str = "full",
    projector_bundle=None,
    output_dim: int | None = None,
    exclude_self: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Build target anchors as weighted kNN averages of warm collaborative embeddings."""
    prediction_mode = str(prediction_mode).strip().lower()
    if prediction_mode not in {"full", "delta"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")
    warm_idx: list[int] = []
    warm_content: list[np.ndarray] = []
    for item_idx in warm_item_indices:
        item_idx = int(item_idx)
        if item_idx <= 0:
            continue
        item_id = idx_to_item.get(item_idx)
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        norm = np.linalg.norm(vec)
        if norm <= 0:
            continue
        warm_idx.append(item_idx)
        warm_content.append(vec / norm)

    if not warm_idx:
        return {}, {}

    warm_lookup = {int(item_idx): pos for pos, item_idx in enumerate(warm_idx)}
    warm_content_mat = np.asarray(warm_content, dtype=np.float32)
    warm_collab_mat = source_model.item_emb.weight[torch.tensor(warm_idx, dtype=torch.long, device=source_model.item_emb.weight.device)]
    warm_collab = warm_collab_mat.detach().cpu().numpy().astype(np.float32)
    if normalize_collab:
        warm_collab = warm_collab / (np.linalg.norm(warm_collab, axis=1, keepdims=True) + 1e-8)

    target_idx: list[int] = []
    target_content: list[np.ndarray] = []
    target_base: list[np.ndarray] | None = [] if prediction_mode == "delta" else None
    for item_idx in target_item_indices:
        item_idx = int(item_idx)
        if item_idx <= 0:
            continue
        item_id = idx_to_item.get(item_idx)
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        norm = np.linalg.norm(vec)
        if norm <= 0:
            continue
        target_idx.append(item_idx)
        target_content.append(vec / norm)
        if target_base is not None:
            if output_dim is None:
                raise ValueError("output_dim must be provided when prediction_mode='delta'.")
            target_base.append(
                pairwise._resolve_content_base_vector(
                    vec,
                    projector_bundle=projector_bundle,
                    output_dim=int(output_dim),
                )
            )

    if not target_idx:
        return {}, {}

    target_content_mat = np.asarray(target_content, dtype=np.float32)
    target_base_mat = np.asarray(target_base, dtype=np.float32) if target_base is not None else None
    anchors: dict[int, np.ndarray] = {}
    confidence: dict[int, float] = {}
    safe_temp = float(max(1e-3, temperature))
    k = max(1, min(int(top_k), warm_content_mat.shape[0]))
    bsz = max(1, int(batch_size))

    for start in range(0, target_content_mat.shape[0], bsz):
        end = min(target_content_mat.shape[0], start + bsz)
        batch = target_content_mat[start:end]  # [B, D]
        sims = np.matmul(batch, warm_content_mat.T)  # [B, W]
        batch_target_idx = target_idx[start:end]
        if exclude_self:
            for row_idx, item_idx in enumerate(batch_target_idx):
                warm_pos = warm_lookup.get(int(item_idx))
                if warm_pos is not None:
                    sims[row_idx, warm_pos] = float("-inf")

        top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        valid_mask = np.isfinite(top_scores).any(axis=1)
        safe_scores = top_scores.copy()
        safe_scores[~np.isfinite(safe_scores)] = -1e9
        scaled = safe_scores / safe_temp
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        weights = np.exp(scaled) * np.isfinite(top_scores)
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)

        selected_collab = warm_collab[top_idx]  # [B, K, H]
        batch_anchors = np.sum(weights[:, :, None] * selected_collab, axis=1)  # [B, H]
        if target_base_mat is not None:
            batch_anchors = batch_anchors - target_base_mat[start:end]
        if normalize_output:
            batch_anchors = batch_anchors / (np.linalg.norm(batch_anchors, axis=1, keepdims=True) + 1e-8)

        # Map avg top-k cosine into [0, 1] confidence.
        conf_scores = np.where(np.isfinite(top_scores), top_scores, -1.0)
        batch_conf = np.clip((np.mean(conf_scores, axis=1) + 1.0) * 0.5, 0.0, 1.0)

        for i, item_idx in enumerate(batch_target_idx):
            if not bool(valid_mask[i]):
                continue
            anchors[int(item_idx)] = batch_anchors[i].astype(np.float32)
            confidence[int(item_idx)] = float(batch_conf[i])

    return anchors, confidence


def _merge_anchor_and_mapper_predictions(
    mapper_predictions: dict[int, np.ndarray],
    anchor_predictions: dict[int, np.ndarray],
    anchor_confidence: dict[int, float],
    alpha: float,
    normalize_output: bool = True,
) -> dict[int, np.ndarray]:
    """Blend anchor and mapper outputs with confidence-aware gating."""
    merged: dict[int, np.ndarray] = {}
    alpha = float(np.clip(alpha, 0.0, 1.0))
    all_keys = set(anchor_predictions.keys()) | set(mapper_predictions.keys())
    for item_idx in all_keys:
        anchor = anchor_predictions.get(item_idx)
        pred = mapper_predictions.get(item_idx)
        if anchor is None and pred is None:
            continue
        if anchor is None:
            vec = pred
        elif pred is None:
            vec = anchor
        else:
            conf = float(np.clip(anchor_confidence.get(item_idx, 0.0), 0.0, 1.0))
            alpha_eff = alpha * conf
            vec = (1.0 - alpha_eff) * anchor + alpha_eff * pred
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            continue
        merged[int(item_idx)] = (vec / norm) if normalize_output else vec
    return merged


def _ensure_output_dir(config: ExperimentConfig) -> Path:
    out_dir = Path(config.registry.save_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_single_alignment_experiment(
    state: dict,
    config: ExperimentConfig,
    experiment_id: str,
    override: dict | None = None,
) -> tuple[dict, dict]:
    local_state = copy.copy(state)
    exp_id = str(experiment_id).upper()
    override = override or {}
    topk_values = list(config.registry.topk_values)
    started_at = time.perf_counter()

    if exp_id == "E0":
        if 'model_with_trainable_delta' not in local_state:
            local_state = run_let_it_go_step(local_state, config=config)
        model_for_eval = local_state['model_with_trainable_delta']
        metrics = _evaluate_model_multi_k(model_for_eval, local_state, topk_values, config=config)
        elapsed = time.perf_counter() - started_at
        record = {
            "experiment_id": exp_id,
            "method_name": _method_name_for_experiment(exp_id),
            "source_model_key": "model_with_trainable_delta",
            "source_model_label": "LetItGo",
            "mapper_type": "delta_finetune",
            "num_predicted_cold": 0,
            "num_updated_total": np.nan,
            "num_updated_warm": np.nan,
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
            "seed": int(override.get("random_state", 42)),
            "runtime_sec": float(elapsed),
            "override_json": json.dumps(override, ensure_ascii=False, sort_keys=True),
        }
        record.update(metrics)
        return local_state, record

    if exp_id == "E11":
        if "model_with_implicit_slim" not in local_state:
            raise KeyError(
                "Experiment E11 requires `model_with_implicit_slim`. "
                "Run run_implicit_slim_step before run_experiment_grid."
            )
        model_for_eval = local_state["model_with_implicit_slim"]
        metrics = _evaluate_model_multi_k(model_for_eval, local_state, topk_values, config=config)
        elapsed = time.perf_counter() - started_at
        record = {
            "experiment_id": exp_id,
            "method_name": _method_name_for_experiment(exp_id),
            "source_model_key": "model_with_implicit_slim",
            "source_model_label": "ImplicitSLIM",
            "mapper_type": "implicit_slim",
            "num_predicted_cold": 0,
            "num_updated_total": np.nan,
            "num_updated_warm": np.nan,
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
            "seed": int(override.get("random_state", 42)),
            "runtime_sec": float(elapsed),
            "override_json": json.dumps(override, ensure_ascii=False, sort_keys=True),
        }
        record.update(metrics)
        return local_state, record

    _KNOWN_PAIRWISE_IDS = {"E1", "E2", "E3", "E3S", "E4", "E5", "E6", "E7", "E8", "E10", "E12", "E13", "E14", "E15", "E16"}
    if exp_id not in _KNOWN_PAIRWISE_IDS and not exp_id.startswith("E16_"):
        raise ValueError(f"Unsupported experiment_id: {experiment_id}")

    source_pref = override.get("pairwise_source_model_preference", config.align.pairwise_source_model_preference)
    source_model, source_key, source_label = _select_pairwise_source_model(
        local_state,
        preference=str(source_pref),
    )
    item_frequencies = _collect_item_interaction_frequencies(local_state.get('train_df'))
    content_projector_bundle = _fit_or_get_content_projector_bundle(local_state, config)
    default_min_warm = int(config.align.pairwise_transformer_min_warm_interactions)
    if exp_id in {"E15", "E16"} or exp_id.startswith("E16_"):
        default_min_warm = min(default_min_warm, 5)
    warm_for_mapper, warm_sampler_info = _select_warm_items_for_mapper(
        local_state,
        config,
        item_frequencies=item_frequencies,
        override=override,
        default_min_warm_interactions=default_min_warm,
    )
    min_warm = int(warm_sampler_info["min_warm_interactions"])

    target_mode = str(override.get("pairwise_target_mode", config.align.pairwise_target_mode)).strip().lower()
    selected_heads = int(override.get("pairwise_transformer_heads", config.align.pairwise_transformer_heads))
    hidden_dim = int(override.get("pairwise_transformer_hidden_dim", config.align.pairwise_transformer_hidden_dim))
    if hidden_dim % int(selected_heads) != 0:
        hidden_dim = int(selected_heads) * int(
            np.ceil(hidden_dim / max(int(selected_heads), 1))
        )

    bundle = pairwise.train_mapper_for_experiment(
        experiment_id=exp_id,
        model_for_projection=source_model,
        reference_model=local_state.get("model_baseline"),
        item_embeddings=local_state['item_embeddings'],
        idx_to_item=local_state['idx_to_item'],
        warm_item_indices=warm_for_mapper,
        content_dim=len(next(iter(local_state['item_embeddings'].values()))),
        device=local_state['device'],
        hidden_dim=local_state['model_with_embeddings'].hidden_units,
        n_layers=int(override.get("pairwise_transformer_layers", config.align.pairwise_transformer_layers)),
        n_heads=selected_heads,
        transformer_dim=hidden_dim,
        epochs=int(override.get("pairwise_transformer_epochs", config.align.pairwise_transformer_epochs)),
        lr=float(override.get("pairwise_transformer_lr", config.align.pairwise_transformer_lr)),
        batch_size=int(override.get("pairwise_transformer_batch_size", config.align.pairwise_transformer_batch_size)),
        item_frequencies=item_frequencies,
        val_fraction=float(override.get("pairwise_transformer_val_fraction", config.align.pairwise_transformer_val_fraction)),
        patience=int(override.get("pairwise_transformer_patience", config.align.pairwise_transformer_patience)),
        min_delta=float(override.get("pairwise_transformer_min_delta", config.align.pairwise_transformer_min_delta)),
        mse_weight=float(override.get("pairwise_transformer_loss_mse_weight", config.align.pairwise_transformer_loss_mse_weight)),
        cosine_weight=float(override.get("pairwise_transformer_loss_cosine_weight", config.align.pairwise_transformer_loss_cosine_weight)),
        nce_weight=float(override.get("pairwise_transformer_loss_nce_weight", config.align.pairwise_transformer_loss_nce_weight)),
        nce_temperature=float(override.get("pairwise_transformer_nce_temperature", config.align.pairwise_transformer_nce_temperature)),
        sample_weight_power=float(override.get("pairwise_transformer_sample_weight_power", config.align.pairwise_transformer_sample_weight_power)),
        weight_decay=float(override.get("pairwise_transformer_weight_decay", config.align.pairwise_transformer_weight_decay)),
        grad_clip=float(override.get("pairwise_transformer_grad_clip", config.align.pairwise_transformer_grad_clip)),
        token_count=int(override.get("pairwise_transformer_token_count", config.align.pairwise_transformer_token_count)),
        dropout=float(override.get("pairwise_transformer_dropout", config.align.pairwise_transformer_dropout)),
        structure_weight=float(override.get("pairwise_transformer_structure_weight", config.align.pairwise_transformer_structure_weight)),
        hard_negative_weight=float(override.get("pairwise_transformer_hard_negative_weight", config.align.pairwise_transformer_hard_negative_weight)),
        hard_negative_top_k=int(override.get("pairwise_transformer_hard_negative_top_k", config.align.pairwise_transformer_hard_negative_top_k)),
        distill_pair_rounds=int(
            override.get("pairwise_distill_pair_rounds", config.align.pairwise_distill_pair_rounds)
        ),
        distill_candidate_count=int(
            override.get("pairwise_distill_candidate_count", config.align.pairwise_distill_candidate_count)
        ),
        distill_teacher_margin=float(
            override.get("pairwise_distill_teacher_margin", config.align.pairwise_distill_teacher_margin)
        ),
        min_warm_interactions=min_warm,
        random_state=int(override.get("random_state", 42)),
        projector_bundle=content_projector_bundle,
        target_mode=target_mode,
    )

    postprocess_mode = str(bundle.get("postprocess", "")).strip().lower()
    embedding_update_mode = str(bundle.get("embedding_update_mode", "blend"))
    infer_scope = str(override.get("pairwise_infer_scope", config.align.pairwise_infer_scope)).strip().lower()
    target_item_indices = _resolve_pairwise_target_item_indices(local_state, infer_scope)
    preserve_norms = bool(bundle.get("preserve_norms", False))
    prediction_mode = _resolve_pairwise_prediction_mode(bundle.get("target_mode", target_mode), embedding_update_mode)
    predicted = pairwise.predict_embeddings_for_experiment(
        bundle,
        item_embeddings=local_state['item_embeddings'],
        idx_to_item=local_state['idx_to_item'],
        target_item_indices=target_item_indices,
        device=local_state['device'],
        batch_size=int(override.get("pairwise_transformer_batch_size", config.align.pairwise_transformer_batch_size)),
        prediction_mode=prediction_mode,
        projector_bundle=content_projector_bundle,
        output_dim=int(local_state['model_with_embeddings'].hidden_units),
    )
    if postprocess_mode == "anchor_residual":
        anchor_top_k = int(override.get("pairwise_anchor_top_k", bundle.get("anchor_top_k", 64)))
        anchor_temperature = float(override.get("pairwise_anchor_temperature", bundle.get("anchor_temperature", 0.07)))
        anchor_batch_size = int(override.get("pairwise_anchor_batch_size", bundle.get("anchor_batch_size", 512)))
        anchor_alpha = float(override.get("pairwise_anchor_blend_alpha", bundle.get("anchor_blend_alpha", 0.5)))
        anchors, anchor_conf = _predict_anchor_embeddings_from_content_knn(
            item_embeddings=local_state["item_embeddings"],
            idx_to_item=local_state["idx_to_item"],
            warm_item_indices=warm_for_mapper,
            target_item_indices=target_item_indices,
            source_model=source_model,
            top_k=anchor_top_k,
            temperature=anchor_temperature,
            batch_size=anchor_batch_size,
            normalize_collab=not preserve_norms,
            normalize_output=not preserve_norms,
            prediction_mode=prediction_mode,
            projector_bundle=content_projector_bundle,
            output_dim=int(local_state['model_with_embeddings'].hidden_units),
            exclude_self=True,
        )
        predicted = _merge_anchor_and_mapper_predictions(
            mapper_predictions=predicted,
            anchor_predictions=anchors,
            anchor_confidence=anchor_conf,
            alpha=anchor_alpha,
            normalize_output=not preserve_norms,
        )

    aligned_model = _clone_sasrec_model_from(
        source_model=source_model,
        num_items=local_state['num_items'],
        device=local_state['device'],
    )
    if "pairwise_transformer_blend_alpha" in override:
        blend_alpha = float(override["pairwise_transformer_blend_alpha"])
    elif config.data.use_letitgo_official_splits and embedding_update_mode == "blend":
        # On official splits cold items are unseen in train; blend with train-updated
        # base embeddings harms cold ranking. Prefer full replacement.
        blend_alpha = 1.0
    else:
        blend_alpha = float(bundle.get("default_blend_alpha", config.align.pairwise_transformer_blend_alpha))
    target_norm = None
    if config.data.use_letitgo_official_splits and not preserve_norms and prediction_mode == "full":
        target_norm = _mean_item_embedding_norm(aligned_model, local_state.get("warm_items", []))
    _blend_alpha_warm = override.get("blend_alpha_warm", None)
    if _blend_alpha_warm is not None:
        _blend_alpha_warm = float(_blend_alpha_warm)
    _freq_decay_k = override.get("blend_alpha_freq_decay_k", None)
    if _freq_decay_k is not None:
        _freq_decay_k = int(_freq_decay_k)
    updated, updated_ids = _apply_predicted_embeddings(
        aligned_model,
        predicted_embeddings=predicted,
        blend_alpha=blend_alpha,
        device=local_state['device'],
        update_mode=embedding_update_mode,
        target_norm=target_norm,
        normalize_output=not preserve_norms,
        warm_item_indices=set(local_state.get("warm_items", [])),
        blend_alpha_warm=_blend_alpha_warm,
        item_frequencies=item_frequencies,
        freq_decay_k=_freq_decay_k,
    )

    default_adapt_epochs = int(bundle.get("default_adapt_epochs", config.align.pairwise_transformer_adapt_epochs))
    adapt_epochs = int(override.get("pairwise_transformer_adapt_epochs", default_adapt_epochs))
    if adapt_epochs > 0 and updated_ids:
        _adapt_selected_item_embeddings(
            model=aligned_model,
            train_loader=local_state['train_loader'],
            device=local_state['device'],
            num_items=local_state['num_items'],
            target_item_indices=updated_ids,
            epochs=adapt_epochs,
            lr=float(override.get("pairwise_transformer_adapt_lr", config.align.pairwise_transformer_adapt_lr)),
            excluded_item_ids=_get_training_excluded_item_ids(local_state),
        )

    metrics = _evaluate_model_multi_k(aligned_model, local_state, topk_values, config=config)
    elapsed = time.perf_counter() - started_at
    updated_cold = _count_item_group_members(updated_ids, local_state.get("cold_items", []))
    updated_warm = _count_item_group_members(updated_ids, local_state.get("warm_items", []))

    local_state[f"experiment_{exp_id}_model"] = aligned_model
    local_state[f"experiment_{exp_id}_bundle"] = bundle

    record = {
        "experiment_id": exp_id,
        "method_name": _method_name_for_experiment(exp_id),
        "source_model_key": source_key,
        "source_model_label": source_label,
        "mapper_type": bundle.get("mapper_type", "unknown"),
        "postprocess_mode": postprocess_mode or np.nan,
        "embedding_update_mode": embedding_update_mode,
        "num_predicted_cold": int(updated_cold),
        "num_updated_total": int(updated),
        "num_updated_warm": int(updated_warm),
        "num_warm_for_mapper": int(len(warm_for_mapper)),
        "pairwise_target_mode": target_mode,
        "pairwise_prediction_mode": prediction_mode,
        "pairwise_infer_scope": infer_scope,
        "pairwise_warm_sampler": warm_sampler_info["sampler"],
        "pairwise_warm_sample_size": int(warm_sampler_info["sample_size"]),
        "pairwise_warm_similarity": warm_sampler_info["similarity"],
        "pairwise_warm_mix_ratio": float(warm_sampler_info["mix_ratio"]),
        "pairwise_item_role_mode": str(config.data.item_role_mode),
        "pairwise_item_role_k": int(config.data.item_role_k),
        "seed": int(override.get("random_state", 42)),
        "runtime_sec": float(elapsed),
        "blend_alpha": float(blend_alpha),
        "override_json": json.dumps(override, ensure_ascii=False, sort_keys=True),
        "training_kwargs_json": json.dumps(bundle.get("training_kwargs", {}), ensure_ascii=False, sort_keys=True),
    }
    record.update(metrics)
    return local_state, record


def _ablation_override_iter(config: ExperimentConfig, method_id: str):
    """Yield method-aware override dicts for ablation trials."""
    method_id = str(method_id).upper()
    ab = config.ablation

    blend_grid = list(ab.blend_alpha) or [float(config.align.pairwise_transformer_blend_alpha)]
    adapt_grid = list(ab.adapt_epochs) or [int(config.align.pairwise_transformer_adapt_epochs)]

    # Focused grid for E3/E3*: only knobs that affect Transformer+InfoNCE behavior.
    if method_id in {"E3", "E3S"}:
        for proj_dim in ab.projection_dims:
            for temp in ab.temperatures:
                for batch_size in ab.batch_sizes:
                    for sw_power in ab.sample_weight_power:
                        for blend_alpha in blend_grid:
                            for adapt_epochs in adapt_grid:
                                yield {
                                    "pairwise_transformer_hidden_dim": int(proj_dim),
                                    "pairwise_transformer_nce_temperature": float(temp),
                                    "pairwise_transformer_batch_size": int(batch_size),
                                    "pairwise_transformer_sample_weight_power": float(sw_power),
                                    "pairwise_transformer_blend_alpha": float(blend_alpha),
                                    "pairwise_transformer_adapt_epochs": int(adapt_epochs),
                                }
        return

    # Focused grid for E14: mapper knobs + anchor fusion strength.
    if method_id == "E14":
        for proj_dim in ab.projection_dims:
            for temp in ab.temperatures:
                for batch_size in ab.batch_sizes:
                    for sw_power in ab.sample_weight_power:
                        for anchor_alpha in blend_grid:
                            for adapt_epochs in adapt_grid:
                                yield {
                                    "pairwise_transformer_hidden_dim": int(proj_dim),
                                    "pairwise_transformer_nce_temperature": float(temp),
                                    "pairwise_transformer_batch_size": int(batch_size),
                                    "pairwise_transformer_sample_weight_power": float(sw_power),
                                    "pairwise_anchor_blend_alpha": float(anchor_alpha),
                                    "pairwise_transformer_adapt_epochs": int(adapt_epochs),
                                }
        return

    # Focused grid for E10: distillation-specific knobs + shared inference knobs.
    if method_id == "E10":
        for proj_dim in ab.projection_dims:
            for temp in ab.temperatures:
                for batch_size in ab.batch_sizes:
                    for sw_power in ab.sample_weight_power:
                        for distill_rounds in ab.distill_pair_rounds:
                            for distill_cands in ab.distill_candidate_count:
                                for distill_margin in ab.distill_teacher_margin:
                                    for blend_alpha in blend_grid:
                                        for adapt_epochs in adapt_grid:
                                            yield {
                                                "pairwise_transformer_hidden_dim": int(proj_dim),
                                                "pairwise_transformer_nce_temperature": float(temp),
                                                "pairwise_transformer_batch_size": int(batch_size),
                                                "pairwise_transformer_sample_weight_power": float(sw_power),
                                                "pairwise_distill_pair_rounds": int(distill_rounds),
                                                "pairwise_distill_candidate_count": int(distill_cands),
                                                "pairwise_distill_teacher_margin": float(distill_margin),
                                                "pairwise_transformer_blend_alpha": float(blend_alpha),
                                                "pairwise_transformer_adapt_epochs": int(adapt_epochs),
                                            }
        return

    # Generic fallback grid for other methods (keeps previous behavior).
    for proj_dim in ab.projection_dims:
        for temp in ab.temperatures:
            for hneg_topk in ab.hard_negative_top_k:
                for loss_set in ab.loss_weight_sets:
                    for batch_size in ab.batch_sizes:
                        for sw_power in ab.sample_weight_power:
                            for blend_alpha in blend_grid:
                                for adapt_epochs in adapt_grid:
                                    yield {
                                        "pairwise_transformer_hidden_dim": int(proj_dim),
                                        "pairwise_transformer_nce_temperature": float(temp),
                                        "pairwise_transformer_hard_negative_top_k": int(hneg_topk),
                                        "pairwise_transformer_loss_mse_weight": float(loss_set["mse"]),
                                        "pairwise_transformer_loss_cosine_weight": float(loss_set["cosine"]),
                                        "pairwise_transformer_loss_nce_weight": float(loss_set["nce"]),
                                        "pairwise_transformer_batch_size": int(batch_size),
                                        "pairwise_transformer_sample_weight_power": float(sw_power),
                                        "pairwise_transformer_blend_alpha": float(blend_alpha),
                                        "pairwise_transformer_adapt_epochs": int(adapt_epochs),
                                    }


def run_ablation_grid(
    state: dict,
    config: ExperimentConfig | None = None,
    base_method_id: str | None = None,
) -> pd.DataFrame:
    config = _resolve_config(config)
    local_state = copy.copy(state)
    method_id = str(base_method_id or config.ablation.method_id).upper()
    max_trials = config.ablation.max_trials

    trials: list[dict[str, Any]] = []
    for trial_idx, base_override in enumerate(_ablation_override_iter(config, method_id), start=1):
        if max_trials is not None and trial_idx > int(max_trials):
            break
        override = dict(base_override)
        override["random_state"] = int(config.ablation.random_state + trial_idx)

        local_state, record = _run_single_alignment_experiment(
            local_state,
            config,
            experiment_id=method_id,
            override=override,
        )
        record["experiment_id"] = str(config.registry.ablation_experiment_id).upper()
        record["ablation_parent_method"] = method_id
        record["ablation_trial_id"] = int(trial_idx)
        trials.append(record)

    return pd.DataFrame(trials)


def compute_significance_table(
    experiment_results: pd.DataFrame,
    baseline_id: str = "E0",
    metric_col: str = "NDCG@10 (холодные)",
    test: str = "wilcoxon",
    alpha: float = 0.05,
) -> pd.DataFrame:
    if experiment_results is None or experiment_results.empty:
        return pd.DataFrame()

    df = experiment_results.copy()
    baseline_id = str(baseline_id).upper()
    df["experiment_id"] = df["experiment_id"].astype(str).str.upper()

    baseline = df[df["experiment_id"] == baseline_id]
    if baseline.empty or metric_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for exp_id in sorted(df["experiment_id"].unique()):
        if exp_id == baseline_id:
            continue
        exp_df = df[df["experiment_id"] == exp_id]
        if "seed" in baseline.columns and "seed" in exp_df.columns:
            paired = baseline[["seed", metric_col]].merge(
                exp_df[["seed", metric_col]],
                on="seed",
                how="inner",
                suffixes=("_base", "_exp"),
            )
            x = paired[f"{metric_col}_base"].to_numpy(dtype=float)
            y = paired[f"{metric_col}_exp"].to_numpy(dtype=float)
        else:
            x = baseline[metric_col].to_numpy(dtype=float)
            y = exp_df[metric_col].to_numpy(dtype=float)
        n = int(min(len(x), len(y)))
        if n < 2:
            rows.append(
                {
                    "experiment_id": exp_id,
                    "metric": metric_col,
                    "baseline_mean": float(np.nanmean(x)) if len(x) > 0 else np.nan,
                    "experiment_mean": float(np.nanmean(y)) if len(y) > 0 else np.nan,
                    "delta": float(np.nanmean(y) - np.nanmean(x)) if len(x) > 0 and len(y) > 0 else np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "test": test,
                }
            )
            continue

        p_value = np.nan
        test_lower = str(test).lower()
        try:
            from scipy import stats

            x_cut = x[:n]
            y_cut = y[:n]
            if test_lower == "ttest":
                _, p_value = stats.ttest_rel(y_cut, x_cut, nan_policy="omit")
            else:
                _, p_value = stats.wilcoxon(y_cut, x_cut, zero_method="wilcox")
        except Exception:
            p_value = np.nan

        rows.append(
            {
                "experiment_id": exp_id,
                "metric": metric_col,
                "baseline_mean": float(np.nanmean(x[:n])),
                "experiment_mean": float(np.nanmean(y[:n])),
                "delta": float(np.nanmean(y[:n]) - np.nanmean(x[:n])),
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "significant": bool(np.isfinite(p_value) and p_value < float(alpha)),
                "test": test,
            }
        )
    return pd.DataFrame(rows)


def run_experiment_grid(
    state: dict,
    config: ExperimentConfig | None = None,
    experiment_ids: list[str] | None = None,
    run_ablation: bool | None = None,
    experiment_overrides: dict[str, dict] | None = None,
) -> dict:
    """Run E0..E14/E3S experiment grid with normalized logging."""
    config = _resolve_config(config)
    _validate_research_config(config)
    local_state = copy.copy(state)

    chosen_experiments = [str(v).upper() for v in (experiment_ids or config.registry.enabled_experiments)]
    run_ablation_flag = bool(config.ablation.enabled if run_ablation is None else run_ablation)
    normalized_overrides: dict[str, dict] = {
        str(exp_id).upper(): dict(override or {})
        for exp_id, override in (experiment_overrides or {}).items()
    }

    records: list[dict[str, Any]] = []
    for exp_id in chosen_experiments:
        override = normalized_overrides.get(exp_id, {})
        local_state, record = _run_single_alignment_experiment(
            local_state,
            config,
            experiment_id=exp_id,
            override=override,
        )
        records.append(record)

    results_df = pd.DataFrame(records)

    if run_ablation_flag:
        ablation_df = run_ablation_grid(local_state, config=config, base_method_id=config.ablation.method_id)
        if not ablation_df.empty:
            results_df = pd.concat([results_df, ablation_df], ignore_index=True)

    if not results_df.empty:
        results_df["timestamp_utc"] = pd.Timestamp.utcnow()
        metric_cols = [c for c in results_df.columns if c.startswith("HitRate@") or c.startswith("NDCG@")]
        preferred_cols = [
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
        ] + metric_cols
        ordered_cols = [col for col in preferred_cols if col in results_df.columns] + [
            c
            for c in results_df.columns
            if c not in {
                *preferred_cols,
            }
        ]
        results_df = results_df.loc[:, ordered_cols]

    local_state["experiment_results"] = results_df

    if config.registry.run_significance and not results_df.empty:
        significance_df = compute_significance_table(
            results_df,
            baseline_id=config.registry.primary_baseline_id,
            metric_col=config.registry.significance_metric,
            test=config.registry.significance_test,
            alpha=config.registry.significance_alpha,
        )
        local_state["experiment_significance"] = significance_df
    else:
        local_state["experiment_significance"] = pd.DataFrame()

    out_dir = _ensure_output_dir(config)
    if config.registry.save_csv and not results_df.empty:
        results_df.to_csv(out_dir / "experiment_results.csv", index=False)
        if isinstance(local_state.get("experiment_significance"), pd.DataFrame) and not local_state["experiment_significance"].empty:
            local_state["experiment_significance"].to_csv(out_dir / "experiment_significance.csv", index=False)
    if config.registry.save_json and not results_df.empty:
        results_df.to_json(out_dir / "experiment_results.json", orient="records", force_ascii=False, indent=2)
        if isinstance(local_state.get("experiment_significance"), pd.DataFrame) and not local_state["experiment_significance"].empty:
            local_state["experiment_significance"].to_json(
                out_dir / "experiment_significance.json",
                orient="records",
                force_ascii=False,
                indent=2,
            )

    print("=" * 80)
    print("EXPERIMENT GRID COMPLETED")
    print("=" * 80)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    if isinstance(local_state.get("experiment_significance"), pd.DataFrame) and not local_state["experiment_significance"].empty:
        print("\nSignificance summary:")
        print(local_state["experiment_significance"].to_string(index=False))

    return local_state


def run_full_pipeline(config: ExperimentConfig | None = None) -> dict:
    if config is None:
        config = ExperimentConfig()

    state: dict[str, Any] = {}
    state = run_data_pipeline(config)
    state = run_encoding_pipeline(state, config)
    state = run_split_and_sequences(state, config)
    state = run_baseline_training(state, config)
    state = run_embedding_training(state, config)
    state = run_implicit_slim_step(state, config)
    state = run_pairwise_step(state, config)
    state = run_let_it_go_step(state, config)
    state = run_cold_evaluation_step(state, config)
    state = build_results_step(state, config)
    return state
