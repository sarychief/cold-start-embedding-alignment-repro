"""Training and evaluation helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import SequentialDataset


def _get_model_catalog_size(model) -> int:
    if hasattr(model, "get_item_embeddings_for_scoring"):
        weights = model.get_item_embeddings_for_scoring()
    else:
        weights = model.item_emb.weight
    return max(int(weights.size(0)) - 1, 0)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device,
    num_items,
    excluded_item_ids: list[int] | None = None,
):
    model.train()
    total_loss = 0.0
    catalog_size = _get_model_catalog_size(model)
    excluded_idx_tensor = None
    if excluded_item_ids:
        excluded_idx = sorted({int(v) - 1 for v in excluded_item_ids if 0 < int(v) <= int(catalog_size)})
        if excluded_idx:
            excluded_idx_tensor = torch.tensor(excluded_idx, device=device, dtype=torch.long)

    for seq, pos in tqdm(dataloader, desc='Training'):
        seq, pos = seq.to(device), pos.to(device)
        optimizer.zero_grad(set_to_none=True)

        input_seq = seq[:, :-1]
        input_pos = pos[:, :-1]
        target_items = seq[:, 1:]

        seq_emb = model(input_seq, input_pos)
        if hasattr(model, "get_item_embeddings_for_scoring"):
            item_emb = model.get_item_embeddings_for_scoring()[1:]
        else:
            item_emb = model.item_emb.weight[1:]
        scores = torch.matmul(seq_emb, item_emb.t())
        if excluded_idx_tensor is not None:
            # Keep positives valid even if they belong to masked ids.
            pos_idx = (target_items - 1).clamp(min=0, max=int(catalog_size) - 1)
            pos_scores = scores.gather(2, pos_idx.unsqueeze(-1))
            scores = scores.index_fill(dim=2, index=excluded_idx_tensor, value=float("-inf"))
            scores = scores.scatter(2, pos_idx.unsqueeze(-1), pos_scores)

        targets = target_items - 1
        targets = targets.masked_fill((target_items <= 0) | (target_items > int(catalog_size)), -100)
        loss = F.cross_entropy(
            scores.reshape(-1, scores.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()
        if hasattr(model, "constrain_delta_"):
            model.constrain_delta_()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def _normalize_topk_values(topk_values, num_items: int):
    unique = sorted({int(v) for v in topk_values if int(v) > 0})
    if not unique:
        unique = [10]
    max_items = max(1, int(num_items))
    effective = {k: max(1, min(int(k), max_items)) for k in unique}
    return unique, effective


def _mask_seen_items(scores: torch.Tensor, seq: torch.Tensor, num_items: int):
    """Mask already seen items except the target item in the last position."""
    pos_items = seq[:, -1]
    batch_size = seq.size(0)
    seen_mask = torch.zeros((batch_size, num_items + 1), dtype=torch.bool, device=scores.device)
    valid = (seq > 0) & (seq <= num_items)
    clipped_seq = seq.clamp(min=0, max=num_items)
    seen_mask.scatter_(1, clipped_seq, valid)

    # Keep target item unmasked (even if it appeared earlier in the sequence),
    # matching the previous evaluation behavior.
    clipped_pos = pos_items.clamp(min=0, max=num_items).unsqueeze(1)
    seen_mask.scatter_(1, clipped_pos, False)

    return scores.masked_fill(seen_mask[:, 1:], float("-inf")), pos_items


def evaluate_all_and_cold_multi_k(
    model,
    dataloader,
    device,
    num_items,
    topk_values,
    cold_items,
    candidate_item_ids: list[int] | None = None,
):
    """Evaluate all and cold metrics for multiple K in one forward pass."""
    model.eval()
    catalog_size = _get_model_catalog_size(model)
    candidate_ids = None
    if candidate_item_ids is not None:
        candidate_ids = sorted({int(v) for v in candidate_item_ids if 0 < int(v) <= int(catalog_size)})

    eval_item_count = len(candidate_ids) if candidate_ids else int(catalog_size)
    requested_k, effective_k = _normalize_topk_values(topk_values, num_items=eval_item_count)
    max_k = max(effective_k.values())

    hit_all = {k: 0.0 for k in requested_k}
    ndcg_all = {k: 0.0 for k in requested_k}
    hit_cold = {k: 0.0 for k in requested_k}
    ndcg_cold = {k: 0.0 for k in requested_k}
    total_all = 0
    total_cold = 0

    cold_indicator = torch.zeros((catalog_size + 1,), dtype=torch.bool, device=device)
    if cold_items:
        cold_idx = torch.tensor([int(v) for v in cold_items if 0 < int(v) <= catalog_size], device=device, dtype=torch.long)
        if cold_idx.numel() > 0:
            cold_indicator[cold_idx] = True

    candidate_indicator = None
    if candidate_ids:
        candidate_indicator = torch.zeros((catalog_size + 1,), dtype=torch.bool, device=device)
        candidate_tensor = torch.tensor(candidate_ids, device=device, dtype=torch.long)
        candidate_indicator[candidate_tensor] = True

    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
    )  # [1/log2(2), ..., 1/log2(max_k+1)]

    with torch.inference_mode():
        for seq, pos in dataloader:
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            scores = model.predict(seq[:, :-1], pos[:, :-1])
            scores, pos_items = _mask_seen_items(scores, seq, num_items=catalog_size)
            valid_targets = (pos_items > 0) & (pos_items <= int(catalog_size))

            target_is_candidate = None
            if candidate_indicator is not None:
                allowed_scores_mask = candidate_indicator[1:].unsqueeze(0)
                scores = scores.masked_fill(~allowed_scores_mask, float("-inf"))
                safe_pos_items = pos_items.clamp(min=0, max=int(catalog_size))
                target_is_candidate = candidate_indicator[safe_pos_items] & valid_targets

            if not bool(valid_targets.any()):
                continue

            topk_items = torch.topk(scores, k=max_k, dim=1).indices + 1
            matches = topk_items.eq(pos_items.unsqueeze(1))
            if target_is_candidate is not None:
                matches = matches & target_is_candidate.unsqueeze(1)

            matches = matches[valid_targets]
            pos_items = pos_items[valid_targets]

            batch_size = int(valid_targets.sum().item())
            total_all += batch_size

            cold_mask = cold_indicator[pos_items]
            batch_cold = int(cold_mask.sum().item())
            total_cold += batch_cold

            cold_matches = matches[cold_mask] if batch_cold > 0 else None

            for k in requested_k:
                k_eff = effective_k[k]
                k_matches = matches[:, :k_eff]
                hit_all[k] += float(k_matches.any(dim=1).sum().item())
                ndcg_all[k] += float((k_matches.float() * discounts[:k_eff]).sum().item())

                if batch_cold > 0 and cold_matches is not None:
                    k_cold_matches = cold_matches[:, :k_eff]
                    hit_cold[k] += float(k_cold_matches.any(dim=1).sum().item())
                    ndcg_cold[k] += float((k_cold_matches.float() * discounts[:k_eff]).sum().item())

    all_metrics = {}
    cold_metrics = {}
    for k in requested_k:
        all_metrics[k] = (
            hit_all[k] / total_all if total_all > 0 else 0.0,
            ndcg_all[k] / total_all if total_all > 0 else 0.0,
        )
        cold_metrics[k] = (
            hit_cold[k] / total_cold if total_cold > 0 else 0.0,
            ndcg_cold[k] / total_cold if total_cold > 0 else 0.0,
        )

    return all_metrics, cold_metrics, total_cold


def evaluate(model, dataloader, device, num_items, k=10):
    all_metrics, _, _ = evaluate_all_and_cold_multi_k(
        model=model,
        dataloader=dataloader,
        device=device,
        num_items=num_items,
        topk_values=[k],
        cold_items=[],
    )
    return all_metrics[int(k)]


def evaluate_cold_items(model, dataloader, device, num_items, cold_items, k=10):
    _, cold_metrics, total_cold = evaluate_all_and_cold_multi_k(
        model=model,
        dataloader=dataloader,
        device=device,
        num_items=num_items,
        topk_values=[k],
        cold_items=cold_items,
    )
    hit_rate, ndcg_val = cold_metrics[int(k)]
    return hit_rate, ndcg_val, int(total_cold)


def build_dataloaders(train_sequences, test_sequences, batch_size=256, max_len=50):
    train_dataset = SequentialDataset(train_sequences, max_len=max_len)
    test_dataset = SequentialDataset(test_sequences, max_len=max_len)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
