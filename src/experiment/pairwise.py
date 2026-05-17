"""Pairwise alignment and implicit fusion helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def implicit_slim(W: np.ndarray, X: np.ndarray, lam: float, alpha: float) -> np.ndarray:
    A = W.copy()
    D = 1 / (X.sum(0) + lam)
    M = (lam * A + A @ X.T @ X) * D * D
    AinvC = lam * M + M @ X.T @ X
    AinvCAt = AinvC @ A.T
    K = np.eye(A.shape[0]) / alpha + AinvCAt
    Xsol = np.linalg.solve(K, AinvC)
    AC = AinvC - AinvCAt @ Xsol
    return alpha * W @ A.T @ AC


def build_context_matrix(item_embeddings, idx_to_item, embedding_dim, num_items):
    X = np.zeros((embedding_dim, num_items), dtype=np.float32)
    for encoded_idx, item_id in idx_to_item.items():
        if encoded_idx == 0:
            continue
        if item_id in item_embeddings:
            X[:, encoded_idx - 1] = item_embeddings[item_id]
    return X


def build_interaction_features_sparse(
    train_df,
    num_users,
    num_items,
    user_col='user_id_encoded',
    item_col='item_id_encoded',
    n_components=64,
    random_state=42,
):
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    rows = train_df[user_col].to_numpy() - 1
    cols = train_df[item_col].to_numpy() - 1
    data = np.ones_like(rows, dtype=np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    n_components = int(min(n_components, num_users - 1, num_items - 1))
    if n_components <= 0:
        raise ValueError("Not enough users/items to build interaction features.")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    item_features = svd.fit_transform(mat.T)  # (num_items, n_components)
    return item_features.T


def compute_textual_similarity_matrix(item_embeddings, item_to_idx, top_k=None, use_sparse=False):
    items_with_embeddings = sorted(set(item_embeddings.keys()) & set(item_to_idx.keys()))
    if len(items_with_embeddings) == 0:
        raise ValueError("No common items between embeddings and item mapping.")

    num_items = len(items_with_embeddings)
    embedding_dim = len(list(item_embeddings.values())[0])
    embedding_to_idx = {item_id: idx for idx, item_id in enumerate(items_with_embeddings)}

    X = np.zeros((num_items, embedding_dim), dtype=np.float32)
    for item_id in tqdm(items_with_embeddings, desc='Загрузка эмбеддингов'):
        X[embedding_to_idx[item_id]] = item_embeddings[item_id]

    X = X / np.where(np.linalg.norm(X, axis=1, keepdims=True) == 0, 1, np.linalg.norm(X, axis=1, keepdims=True))
    sim = np.dot(X, X.T)

    if top_k is not None:
        sparse_sim = np.zeros_like(sim)
        for i in tqdm(range(num_items), desc='Фильтрация top-K'):
            top_indices = np.argsort(sim[i])[-top_k - 1:-1]
            sparse_sim[i, top_indices] = sim[i, top_indices]
        sim = sparse_sim

    if use_sparse:
        from scipy.sparse import csr_matrix
        sim = csr_matrix(sim)

    return sim, embedding_to_idx


def regularize_cold_items_embeddings(
    model,
    similarity_matrix,
    cold_item_indices,
    warm_item_indices,
    alpha=0.5,
    embedding_to_idx=None,
    idx_to_item=None,
):
    if hasattr(similarity_matrix, 'toarray'):
        similarity_matrix = similarity_matrix.toarray()

    use_mapping = embedding_to_idx is not None and idx_to_item is not None

    for cold_idx in tqdm(cold_item_indices, desc='Регуляризация холодных товаров'):
        if cold_idx in warm_item_indices:
            continue

        if idx_to_item is not None:
            cold_item_id = idx_to_item[cold_idx]
        else:
            cold_item_id = cold_idx

        if use_mapping and cold_item_id not in embedding_to_idx:
            continue

        cold_local_idx = embedding_to_idx[cold_item_id] if use_mapping else cold_idx - 1
        if cold_local_idx >= similarity_matrix.shape[0]:
            continue

        sims = similarity_matrix[cold_local_idx, :]
        warm_similarities = []
        warm_embeddings = []

        for warm_idx in warm_item_indices:
            if idx_to_item is not None:
                warm_item_id = idx_to_item[warm_idx]
            else:
                warm_item_id = warm_idx

            if use_mapping and warm_item_id not in embedding_to_idx:
                continue

            warm_local_idx = embedding_to_idx[warm_item_id] if use_mapping else warm_idx - 1
            if warm_local_idx >= similarity_matrix.shape[1]:
                continue

            sim_val = sims[warm_local_idx]
            if sim_val > 0:
                warm_similarities.append(sim_val)
                warm_embeddings.append(model.item_emb.weight[warm_idx].detach().cpu().numpy())

        if not warm_embeddings:
            continue

        warm_similarities = np.array(warm_similarities, dtype=np.float32)
        warm_similarities = warm_similarities / (warm_similarities.sum() + 1e-8)

        regularized = np.zeros_like(warm_embeddings[0], dtype=np.float32)
        for i, emb in enumerate(warm_embeddings):
            regularized += warm_similarities[i] * emb

        current = model.item_emb.weight[cold_idx].detach().cpu().numpy()
        new_emb = (1 - alpha) * current + alpha * regularized
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        new_emb_torch = torch.tensor(
            new_emb,
            dtype=model.item_emb.weight.dtype,
            device=model.item_emb.weight.device,
        )
        with torch.no_grad():
            model.item_emb.weight[cold_idx].copy_(new_emb_torch)


def _normalize_np_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec.astype(np.float32, copy=True)
    return (vec / norm).astype(np.float32, copy=False)


def _prepare_frequency_weight(freq: float, power: float) -> float:
    base = np.log1p(max(freq, 1.0))
    return float(base ** max(power, 0.0))


def _project_content_base_vector(
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


def _resolve_content_base_vector(
    vec: np.ndarray,
    projector_bundle,
    output_dim: int,
) -> np.ndarray:
    return _project_content_base_vector(
        vec,
        projector_bundle=projector_bundle,
        output_dim=output_dim,
    )


def _alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    mse_weight: float,
    cosine_weight: float,
    nce_weight: float,
    nce_temperature: float,
    content_batch: torch.Tensor | None = None,
    structure_weight: float = 0.0,
    hard_negative_weight: float = 0.0,
    hard_negative_top_k: int = 0,
):
    if sample_weights.ndim != 1:
        sample_weights = sample_weights.view(-1)
    normalized_weights = sample_weights / sample_weights.sum().clamp_min(1e-8)

    mse_per = torch.mean((pred - target) ** 2, dim=1)
    mse_loss = torch.sum(normalized_weights * mse_per)

    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cosine_per = 1.0 - torch.sum(pred_norm * target_norm, dim=1)
    cosine_loss = torch.sum(normalized_weights * cosine_per)

    nce_loss = pred.new_tensor(0.0)
    if nce_weight > 0 and pred.size(0) > 1:
        temperature = max(float(nce_temperature), 1e-6)
        logits = torch.matmul(pred_norm, target_norm.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        nce_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    structure_loss = pred.new_tensor(0.0)
    if structure_weight > 0 and pred.size(0) > 1:
        pred_sim = torch.matmul(pred_norm, pred_norm.t())
        target_sim = torch.matmul(target_norm, target_norm.t())
        structure_loss = F.mse_loss(pred_sim, target_sim)

    hard_negative_loss = pred.new_tensor(0.0)
    if (
        hard_negative_weight > 0
        and hard_negative_top_k > 0
        and pred.size(0) > 2
        and content_batch is not None
    ):
        content_norm = F.normalize(content_batch, p=2, dim=1)
        content_sim = torch.matmul(content_norm, content_norm.t())
        target_sim = torch.matmul(target_norm, target_norm.t())
        pred_sim = torch.matmul(pred_norm, pred_norm.t())

        discrepancy = content_sim - target_sim
        eye = torch.eye(discrepancy.size(0), device=discrepancy.device, dtype=torch.bool)
        discrepancy = discrepancy.masked_fill(eye, float("-inf"))
        pred_gap = (pred_sim - target_sim).masked_fill(eye, 0.0)
        pred_gap = torch.relu(pred_gap)

        flat_scores = discrepancy.view(-1)
        k = int(min(hard_negative_top_k, max(flat_scores.numel() - discrepancy.size(0), 1)))
        top_vals, top_idx = torch.topk(flat_scores, k=k, largest=True)
        valid = torch.isfinite(top_vals) & (top_vals > 0)
        if torch.any(valid):
            valid_idx = top_idx[valid]
            hard_negative_loss = pred_gap.view(-1)[valid_idx].mean()

    total_loss = (
        mse_weight * mse_loss
        + cosine_weight * cosine_loss
        + nce_weight * nce_loss
        + structure_weight * structure_loss
        + hard_negative_weight * hard_negative_loss
    )
    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "mse": float(mse_loss.detach().item()),
        "cosine": float(cosine_loss.detach().item()),
        "nce": float(nce_loss.detach().item()),
        "structure": float(structure_loss.detach().item()),
        "hard_negative": float(hard_negative_loss.detach().item()),
    }


def _pairwise_distillation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    anchor_indices: torch.Tensor,
    teacher_embeddings: torch.Tensor | None,
    mse_weight: float,
    cosine_weight: float,
    ranking_weight: float,
    temperature: float,
    pair_rounds: int = 2,
    candidate_count: int = 64,
    teacher_margin: float = 0.01,
):
    """Pairwise ranking distillation from teacher collaborative geometry."""
    if sample_weights.ndim != 1:
        sample_weights = sample_weights.view(-1)
    normalized_weights = sample_weights / sample_weights.sum().clamp_min(1e-8)

    mse_per = torch.mean((pred - target) ** 2, dim=1)
    mse_loss = torch.sum(normalized_weights * mse_per)

    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cosine_per = 1.0 - torch.sum(pred_norm * target_norm, dim=1)
    cosine_loss = torch.sum(normalized_weights * cosine_per)

    ranking_loss = pred.new_tensor(0.0)
    if (
        ranking_weight > 0
        and teacher_embeddings is not None
        and teacher_embeddings.ndim == 2
        and teacher_embeddings.size(0) > 2
    ):
        num_items = int(teacher_embeddings.size(0))
        rounds = int(max(1, pair_rounds))
        cand_count = int(min(max(4, candidate_count), num_items - 1))
        margin = float(max(0.0, teacher_margin))
        temp = float(max(temperature, 1e-6))

        anchor_indices = anchor_indices.long().to(pred.device)
        anchor_indices = anchor_indices.clamp(min=0, max=num_items - 1)
        teacher_anchor = teacher_embeddings[anchor_indices]
        batch_size = pred.size(0)
        batch_range = torch.arange(batch_size, device=pred.device)

        accum_loss = pred.new_tensor(0.0)
        valid_rounds = 0
        for _ in range(rounds):
            candidates = torch.randint(
                low=0,
                high=num_items,
                size=(batch_size, cand_count),
                device=pred.device,
            )
            anchor_grid = anchor_indices.unsqueeze(1).expand_as(candidates)
            candidates = torch.where(candidates == anchor_grid, (candidates + 1) % num_items, candidates)

            candidate_emb = teacher_embeddings[candidates]  # [B, C, D]
            teacher_scores = torch.einsum("bd,bcd->bc", teacher_anchor, candidate_emb)

            pos_c = torch.argmax(teacher_scores, dim=1)
            neg_c = torch.argmin(teacher_scores, dim=1)
            pos_teacher = teacher_scores[batch_range, pos_c]
            neg_teacher = teacher_scores[batch_range, neg_c]
            teacher_gap = pos_teacher - neg_teacher
            valid_mask = teacher_gap > margin

            if not torch.any(valid_mask):
                continue

            pos_emb = candidate_emb[batch_range, pos_c]
            neg_emb = candidate_emb[batch_range, neg_c]

            student_pos = torch.sum(pred_norm * pos_emb, dim=1)
            student_neg = torch.sum(pred_norm * neg_emb, dim=1)
            rank_per = F.softplus(-(student_pos - student_neg) / temp)
            confidence = 1.0 + torch.relu(teacher_gap)
            weighted_rank = rank_per * confidence

            mask_weights = normalized_weights * valid_mask.float()
            denom = mask_weights.sum().clamp_min(1e-8)
            accum_loss = accum_loss + torch.sum(mask_weights * weighted_rank) / denom
            valid_rounds += 1

        if valid_rounds > 0:
            ranking_loss = accum_loss / float(valid_rounds)

    total_loss = (
        mse_weight * mse_loss
        + cosine_weight * cosine_loss
        + ranking_weight * ranking_loss
    )
    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "mse": float(mse_loss.detach().item()),
        "cosine": float(cosine_loss.detach().item()),
        "nce": float(ranking_loss.detach().item()),
        "structure": 0.0,
        "hard_negative": 0.0,
    }


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {
            "loss": 0.0,
            "mse": 0.0,
            "cosine": 0.0,
            "nce": 0.0,
            "structure": 0.0,
            "hard_negative": 0.0,
        }
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def fit_procrustes_mapping(
    model_for_projection,
    item_embeddings: dict,
    idx_to_item: dict,
    warm_item_indices: list[int],
    content_dim: int,
    item_frequencies: Mapping[int, int] | None = None,
    min_interactions: int = 1,
):
    """Fit an orthogonal (or rectangular semi-orthogonal) Procrustes mapping XW≈Y."""
    x_data: list[np.ndarray] = []
    y_data: list[np.ndarray] = []
    weights: list[float] = []

    for item_idx in warm_item_indices:
        item_idx = int(item_idx)
        item_id = idx_to_item.get(item_idx)
        if item_id is None or item_id not in item_embeddings:
            continue

        freq = int(item_frequencies.get(item_idx, 1)) if item_frequencies is not None else 1
        if freq < int(max(1, min_interactions)):
            continue

        x = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if x.shape[0] != int(content_dim):
            continue
        y = model_for_projection.item_emb.weight[item_idx].detach().cpu().numpy().astype(np.float32)
        x_data.append(_normalize_np_vector(x))
        y_data.append(_normalize_np_vector(y))
        weights.append(np.log1p(max(freq, 1)))

    if len(x_data) < 2:
        raise RuntimeError("Недостаточно warm-пар для Procrustes alignment.")

    X = np.asarray(x_data, dtype=np.float32)
    Y = np.asarray(y_data, dtype=np.float32)
    W = np.asarray(weights, dtype=np.float32)
    W = W / (W.mean() + 1e-8)

    Xw = X * W[:, None]
    Yw = Y * W[:, None]
    cross = Xw.T @ Yw
    U, _, Vt = np.linalg.svd(cross, full_matrices=False)
    mapping = (U @ Vt).astype(np.float32)
    return mapping


def apply_procrustes_mapping(
    mapping: np.ndarray,
    item_embeddings: dict,
    idx_to_item: dict,
    target_item_indices: list[int],
):
    """Project content embeddings into collaborative space with a Procrustes map."""
    out: dict[int, np.ndarray] = {}
    for item_idx in target_item_indices:
        item_idx = int(item_idx)
        if item_idx <= 0:
            continue
        item_id = idx_to_item.get(item_idx)
        if item_id is None or item_id not in item_embeddings:
            continue
        x = np.asarray(item_embeddings[item_id], dtype=np.float32)
        pred = x @ mapping
        out[item_idx] = _normalize_np_vector(pred)
    return out


class ContentToCollaborativeMLP(torch.nn.Module):
    """MLP mapper with optional linear Procrustes residual branch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual_linear: np.ndarray | None = None,
        residual_trainable: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(max(1, num_layers))

        layers = []
        if self.num_layers == 1:
            layers.append(torch.nn.Linear(self.input_dim, self.output_dim))
        else:
            layers.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout))
            for _ in range(self.num_layers - 2):
                layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(torch.nn.GELU())
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(self.hidden_dim, self.output_dim))
        self.main = torch.nn.Sequential(*layers)

        self.residual = None
        if residual_linear is not None:
            residual_linear = np.asarray(residual_linear, dtype=np.float32)
            if residual_linear.shape != (self.input_dim, self.output_dim):
                raise ValueError(
                    "residual_linear has invalid shape: "
                    f"expected {(self.input_dim, self.output_dim)}, got {residual_linear.shape}"
                )
            self.residual = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
            with torch.no_grad():
                self.residual.weight.copy_(torch.from_numpy(residual_linear.T))
            self.residual.weight.requires_grad = bool(residual_trainable)

    def forward(self, content_embeddings: torch.Tensor) -> torch.Tensor:
        x = content_embeddings.float()
        out = self.main(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out


def _build_alignment_mapper(
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    token_count: int,
    dropout: float,
    procrustes_matrix: np.ndarray | None = None,
):
    normalized_model_type = str(model_type).strip().lower()
    if normalized_model_type == "transformer":
        return ContentToCollaborativeTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            num_heads=n_heads,
            token_count=token_count,
            dropout=dropout,
        )
    if normalized_model_type == "mlp":
        return ContentToCollaborativeMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            residual_linear=procrustes_matrix,
            residual_trainable=True,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_content_to_collab_transformer(
    model_for_projection,
    item_embeddings: dict,
    idx_to_item: dict,
    warm_item_indices: list[int],
    content_dim: int,
    device: torch.device,
    hidden_dim: int,
    n_layers: int = 2,
    n_heads: int = 2,
    transformer_dim: int | None = None,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 128,
    item_frequencies: Mapping[int, int] | None = None,
    val_fraction: float = 0.1,
    patience: int = 4,
    min_delta: float = 1e-4,
    mse_weight: float = 0.2,
    cosine_weight: float = 0.6,
    nce_weight: float = 0.2,
    nce_temperature: float = 0.07,
    sample_weight_power: float = 0.5,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    token_count: int = 8,
    dropout: float = 0.1,
    model_type: str = "transformer",
    structure_weight: float = 0.0,
    hard_negative_weight: float = 0.0,
    hard_negative_top_k: int = 0,
    procrustes_init_matrix: np.ndarray | None = None,
    objective: str = "alignment",
    distill_pair_rounds: int = 2,
    distill_candidate_count: int = 64,
    distill_teacher_margin: float = 0.01,
    random_state: int = 42,
    normalize_targets: bool = True,
    projector_bundle=None,
    target_mode: str = "full",
):
    """Train a mapper from content space to collaborative embedding space."""
    content_dim = int(content_dim)
    transformer_dim = int(transformer_dim or hidden_dim)
    objective = str(objective).strip().lower()
    if objective not in {"alignment", "pairwise_distill"}:
        raise ValueError(f"Unsupported objective: {objective}")
    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"full", "delta_from_content"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    use_delta_targets = target_mode == "delta_from_content"
    rng = np.random.RandomState(random_state)

    if content_dim <= 0:
        raise ValueError("content_dim must be > 0")
    if not warm_item_indices:
        raise ValueError("Warm items list is empty; cannot train content-to-collab mapper.")

    x_data: list[np.ndarray] = []
    y_data: list[np.ndarray] = []
    sample_weights: list[float] = []
    for item_idx in warm_item_indices:
        item_idx = int(item_idx)
        item_id = idx_to_item.get(item_idx)
        if item_id is None or item_id not in item_embeddings:
            continue

        content_vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if content_vec.size == 0:
            continue
        if content_vec.shape[0] != content_dim:
            raise ValueError(f"Inconsistent content dim: expected {content_dim}, got {content_vec.shape[0]}")

        target_vec = model_for_projection.item_emb.weight[item_idx].detach().cpu().numpy().astype(np.float32)
        x_data.append(_normalize_np_vector(content_vec))
        if use_delta_targets:
            content_base = _resolve_content_base_vector(
                content_vec,
                projector_bundle=projector_bundle,
                output_dim=hidden_dim,
            )
        if normalize_targets:
            target_for_loss = _normalize_np_vector(target_vec)
        else:
            target_for_loss = target_vec.astype(np.float32, copy=False)
        if use_delta_targets:
            y_data.append(np.asarray(target_for_loss - content_base, dtype=np.float32))
        else:
            y_data.append(np.asarray(target_for_loss, dtype=np.float32))

        freq = float(item_frequencies.get(item_idx, 1)) if item_frequencies is not None else 1.0
        sample_weights.append(_prepare_frequency_weight(freq, sample_weight_power))

    if len(x_data) == 0:
        raise RuntimeError("Не найдено ни одного тёплого айтема с доступным контентным эмбеддингом.")

    X = np.asarray(x_data, dtype=np.float32)
    Y = np.asarray(y_data, dtype=np.float32)
    W = np.asarray(sample_weights, dtype=np.float32)
    if not np.isfinite(W).all() or np.allclose(W.mean(), 0.0):
        W = np.ones_like(W, dtype=np.float32)
    else:
        W = W / (W.mean() + 1e-8)

    n_samples = X.shape[0]
    min_val_size = 256
    if val_fraction > 0 and n_samples >= min_val_size * 2:
        val_size = max(min_val_size, int(n_samples * float(val_fraction)))
        val_size = min(val_size, n_samples - min_val_size)
        permutation = rng.permutation(n_samples)
        val_idx = permutation[:val_size]
        train_idx = permutation[val_size:]
    else:
        train_idx = np.arange(n_samples)
        val_idx = None

    train_idx_tensor = torch.from_numpy(np.asarray(train_idx, dtype=np.int64))
    train_tensors = [
        torch.from_numpy(X[train_idx]),
        torch.from_numpy(Y[train_idx]),
        torch.from_numpy(W[train_idx]),
        train_idx_tensor,
    ]
    train_dataset = TensorDataset(*train_tensors)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_idx is not None:
        val_idx_tensor = torch.from_numpy(np.asarray(val_idx, dtype=np.int64))
        val_tensors = [
            torch.from_numpy(X[val_idx]),
            torch.from_numpy(Y[val_idx]),
            torch.from_numpy(W[val_idx]),
            val_idx_tensor,
        ]
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

    mapper = _build_alignment_mapper(
        model_type=model_type,
        input_dim=X.shape[1],
        output_dim=hidden_dim,
        hidden_dim=transformer_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        token_count=token_count,
        dropout=dropout,
        procrustes_matrix=procrustes_init_matrix,
    ).to(device)

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    teacher_embeddings = None
    if objective == "pairwise_distill":
        teacher_embeddings = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)

    best_state = copy.deepcopy(mapper.state_dict())
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        mapper.train()
        train_metrics: list[dict[str, float]] = []

        for batch in train_loader:
            xb, yb, wb, idxb = batch
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            wb = wb.to(device, non_blocking=pin_memory)
            idxb = idxb.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            pred = mapper(xb)
            if objective == "pairwise_distill":
                loss, loss_metrics = _pairwise_distillation_loss(
                    pred,
                    yb,
                    wb,
                    anchor_indices=idxb,
                    teacher_embeddings=teacher_embeddings,
                    mse_weight=mse_weight,
                    cosine_weight=cosine_weight,
                    ranking_weight=nce_weight,
                    temperature=nce_temperature,
                    pair_rounds=distill_pair_rounds,
                    candidate_count=distill_candidate_count,
                    teacher_margin=distill_teacher_margin,
                )
            else:
                loss, loss_metrics = _alignment_loss(
                    pred,
                    yb,
                    wb,
                    mse_weight=mse_weight,
                    cosine_weight=cosine_weight,
                    nce_weight=nce_weight,
                    nce_temperature=nce_temperature,
                    content_batch=xb,
                    structure_weight=structure_weight,
                    hard_negative_weight=hard_negative_weight,
                    hard_negative_top_k=hard_negative_top_k,
                )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_metrics.append(loss_metrics)

        scheduler.step()
        avg_train = _mean_metrics(train_metrics)

        if val_loader is None:
            if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"Эпоха {model_type}-мэппинга "
                    f"{epoch + 1}/{epochs}: train_loss={avg_train['loss']:.6f}, "
                    f"mse={avg_train['mse']:.6f}, cos={1.0 - avg_train['cosine']:.6f}, "
                    f"nce={avg_train['nce']:.6f}, struct={avg_train['structure']:.6f}, "
                    f"hneg={avg_train['hard_negative']:.6f}"
                )
            continue

        mapper.eval()
        val_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                xb, yb, wb, idxb = batch
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                wb = wb.to(device, non_blocking=pin_memory)
                idxb = idxb.to(device, non_blocking=pin_memory)
                pred = mapper(xb)
                if objective == "pairwise_distill":
                    _, loss_metrics = _pairwise_distillation_loss(
                        pred,
                        yb,
                        wb,
                        anchor_indices=idxb,
                        teacher_embeddings=teacher_embeddings,
                        mse_weight=mse_weight,
                        cosine_weight=cosine_weight,
                        ranking_weight=nce_weight,
                        temperature=nce_temperature,
                        pair_rounds=distill_pair_rounds,
                        candidate_count=distill_candidate_count,
                        teacher_margin=distill_teacher_margin,
                    )
                else:
                    _, loss_metrics = _alignment_loss(
                        pred,
                        yb,
                        wb,
                        mse_weight=mse_weight,
                        cosine_weight=cosine_weight,
                        nce_weight=nce_weight,
                        nce_temperature=nce_temperature,
                        content_batch=xb,
                        structure_weight=structure_weight,
                        hard_negative_weight=hard_negative_weight,
                        hard_negative_top_k=hard_negative_top_k,
                    )
                val_metrics.append(loss_metrics)

        avg_val = _mean_metrics(val_metrics)
        improved = avg_val["loss"] < (best_val_loss - float(min_delta))
        if improved:
            best_val_loss = avg_val["loss"]
            best_state = copy.deepcopy(mapper.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"Эпоха {model_type}-мэппинга "
            f"{epoch + 1}/{epochs}: train_loss={avg_train['loss']:.6f}, "
            f"val_loss={avg_val['loss']:.6f}, val_cos={1.0 - avg_val['cosine']:.6f}, "
            f"val_nce={avg_val['nce']:.6f}, val_struct={avg_val['structure']:.6f}, "
            f"val_hneg={avg_val['hard_negative']:.6f}"
        )

        if stale_epochs >= int(max(1, patience)):
            print(f"Ранняя остановка {model_type}-мэппинга на эпохе {epoch + 1}")
            break

    if val_loader is not None:
        mapper.load_state_dict(best_state)

    return mapper


def predict_embeddings_with_mapper(
    mapper,
    item_embeddings: dict,
    idx_to_item: dict,
    target_item_indices: list[int],
    device: torch.device,
    batch_size: int = 128,
    normalize_predictions: bool = True,
    prediction_mode: str = "full",
    projector_bundle=None,
    output_dim: int | None = None,
    target_mode: str = "full",
):
    """Predict mapper outputs for the provided item indices."""
    prediction_mode = str(prediction_mode).strip().lower()
    if prediction_mode not in {"full", "delta"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")
    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"full", "delta_from_content"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    if output_dim is None:
        output_dim = getattr(mapper, "output_dim", None)
    item_indices_with_content = []
    content_vectors = []
    need_base_vectors = (target_mode == "delta_from_content") != (prediction_mode == "delta")
    base_vectors = [] if need_base_vectors else None

    for item_idx in target_item_indices:
        if item_idx <= 0:
            continue
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None or item_id not in item_embeddings:
            continue
        vec = np.asarray(item_embeddings[item_id], dtype=np.float32)
        if vec.size == 0:
            continue
        item_indices_with_content.append(int(item_idx))
        content_vectors.append(_normalize_np_vector(vec))
        if base_vectors is not None:
            if output_dim is None:
                raise ValueError("output_dim must be provided when prediction_mode='delta'.")
            base_vectors.append(
                _resolve_content_base_vector(
                    vec,
                    projector_bundle=projector_bundle,
                    output_dim=int(output_dim),
                )
            )

    if not item_indices_with_content:
        return {}

    X = torch.from_numpy(np.asarray(content_vectors, dtype=np.float32))
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    base_tensor = torch.from_numpy(np.asarray(base_vectors, dtype=np.float32)) if base_vectors is not None else None

    mapper.eval()
    out = {}
    cursor = 0
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            pred = mapper(batch_x)
            if base_tensor is not None:
                batch_size_actual = pred.shape[0]
                batch_base = base_tensor[cursor : cursor + batch_size_actual].to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                if target_mode == "delta_from_content":
                    pred = pred + batch_base
                else:
                    pred = pred - batch_base
            if normalize_predictions:
                pred = F.normalize(pred, p=2, dim=1)
            pred = pred.detach().cpu().numpy()

            batch_size_actual = pred.shape[0]
            batch_indices = item_indices_with_content[cursor : cursor + batch_size_actual]
            for i, item_idx in enumerate(batch_indices):
                out[item_idx] = pred[i]
            cursor += batch_size_actual

    return out


def predict_cold_embeddings_with_mapper(
    mapper,
    item_embeddings: dict,
    idx_to_item: dict,
    cold_item_indices: list[int],
    device: torch.device,
    batch_size: int = 128,
    normalize_predictions: bool = True,
):
    """Backward-compatible wrapper for cold-only full-vector inference."""
    return predict_embeddings_with_mapper(
        mapper,
        item_embeddings=item_embeddings,
        idx_to_item=idx_to_item,
        target_item_indices=cold_item_indices,
        device=device,
        batch_size=batch_size,
        normalize_predictions=normalize_predictions,
        prediction_mode="full",
    )


def train_mapper_for_experiment(
    experiment_id: str,
    model_for_projection,
    item_embeddings: dict,
    idx_to_item: dict,
    warm_item_indices: list[int],
    content_dim: int,
    device: torch.device,
    hidden_dim: int,
    n_layers: int = 2,
    n_heads: int = 2,
    transformer_dim: int | None = None,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 128,
    item_frequencies: Mapping[int, int] | None = None,
    val_fraction: float = 0.1,
    patience: int = 4,
    min_delta: float = 1e-4,
    mse_weight: float = 0.2,
    cosine_weight: float = 0.6,
    nce_weight: float = 0.2,
    nce_temperature: float = 0.07,
    sample_weight_power: float = 0.5,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    token_count: int = 8,
    dropout: float = 0.1,
    structure_weight: float = 0.0,
    hard_negative_weight: float = 0.0,
    hard_negative_top_k: int = 0,
    distill_pair_rounds: int = 2,
    distill_candidate_count: int = 64,
    distill_teacher_margin: float = 0.01,
    min_warm_interactions: int = 1,
    random_state: int = 42,
    reference_model=None,
    projector_bundle=None,
    target_mode: str = "full",
):
    """Train a mapper for E1-E8/E10/E12/E13/E14/E15/E16/E3S experiments and return model metadata."""
    exp_id = str(experiment_id).strip().upper()
    training_kwargs = {
        "model_for_projection": model_for_projection,
        "item_embeddings": item_embeddings,
        "idx_to_item": idx_to_item,
        "warm_item_indices": warm_item_indices,
        "content_dim": content_dim,
        "device": device,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "transformer_dim": transformer_dim,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "item_frequencies": item_frequencies,
        "val_fraction": val_fraction,
        "patience": patience,
        "min_delta": min_delta,
        "mse_weight": mse_weight,
        "cosine_weight": cosine_weight,
        "nce_weight": nce_weight,
        "nce_temperature": nce_temperature,
        "sample_weight_power": sample_weight_power,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "token_count": token_count,
        "dropout": dropout,
        "structure_weight": structure_weight,
        "hard_negative_weight": hard_negative_weight,
        "hard_negative_top_k": hard_negative_top_k,
        "distill_pair_rounds": distill_pair_rounds,
        "distill_candidate_count": distill_candidate_count,
        "distill_teacher_margin": distill_teacher_margin,
        "random_state": random_state,
        "projector_bundle": projector_bundle,
        "target_mode": target_mode,
    }

    if exp_id == "E1":
        mapping = fit_procrustes_mapping(
            model_for_projection=model_for_projection,
            item_embeddings=item_embeddings,
            idx_to_item=idx_to_item,
            warm_item_indices=warm_item_indices,
            content_dim=content_dim,
            item_frequencies=item_frequencies,
            min_interactions=min_warm_interactions,
        )
        return {
            "experiment_id": exp_id,
            "mapper_type": "procrustes",
            "mapper": None,
            "procrustes_matrix": mapping,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "min_warm_interactions": min_warm_interactions,
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E2":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "mlp",
                "mse_weight": 1.0,
                "cosine_weight": 0.0,
                "nce_weight": 0.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "mlp",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "mse",
                "model_type": "mlp",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E3":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "mse_weight": 0.0,
                "cosine_weight": 0.0,
                "nce_weight": 1.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "contrastive",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E3S":
        # Best-performing focused ablation setup for E3 from prior runs.
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "transformer_dim": 128,
                "batch_size": 256,
                "nce_temperature": 0.05,
                "sample_weight_power": 0.8,
                "mse_weight": 0.0,
                "cosine_weight": 0.0,
                "nce_weight": 1.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_tuned_e3",
            "mapper": mapper,
            "procrustes_matrix": None,
            "default_blend_alpha": 0.50,
            "default_adapt_epochs": 0,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "contrastive_tuned_e3_star",
                "model_type": "transformer",
                "transformer_dim": 128,
                "batch_size": 256,
                "nce_temperature": 0.05,
                "sample_weight_power": 0.8,
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E4":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update({"model_type": "transformer"})
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "multi_objective",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E5":
        mapping = fit_procrustes_mapping(
            model_for_projection=model_for_projection,
            item_embeddings=item_embeddings,
            idx_to_item=idx_to_item,
            warm_item_indices=warm_item_indices,
            content_dim=content_dim,
            item_frequencies=item_frequencies,
            min_interactions=min_warm_interactions,
        )
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "mlp",
                "procrustes_init_matrix": mapping,
                "mse_weight": 0.2,
                "cosine_weight": 0.3,
                "nce_weight": 0.5,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "mlp",
            "mapper": mapper,
            "procrustes_matrix": mapping,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "procrustes_init_then_contrastive",
                "model_type": "mlp",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E6":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "structure_weight": max(float(structure_weight), 0.2),
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "multi_objective_plus_structure",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E7":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "hard_negative_weight": max(float(hard_negative_weight), 0.2),
                "hard_negative_top_k": max(int(hard_negative_top_k), 512),
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "multi_objective_plus_hard_negative",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E8":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "sample_weight_power": max(float(sample_weight_power), 0.8),
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "confidence_weighted_multi_objective",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E10":
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "mlp",
                "objective": "pairwise_distill",
                "mse_weight": max(float(mse_weight), 0.1),
                "cosine_weight": max(float(cosine_weight), 0.1),
                "nce_weight": max(float(nce_weight), 0.7),
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
                "distill_pair_rounds": max(int(distill_pair_rounds), 1),
                "distill_candidate_count": max(int(distill_candidate_count), 8),
                "distill_teacher_margin": max(float(distill_teacher_margin), 0.0),
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "mlp_distill",
            "mapper": mapper,
            "procrustes_matrix": None,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "pairwise_ranking_distillation",
                "model_type": "mlp",
                "distill_pair_rounds": int(exp_kwargs["distill_pair_rounds"]),
                "distill_candidate_count": int(exp_kwargs["distill_candidate_count"]),
                "distill_teacher_margin": float(exp_kwargs["distill_teacher_margin"]),
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E12":
        # User-requested variant: treat mapper output as an additive delta at inference.
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "mse_weight": 0.0,
                "cosine_weight": 0.0,
                "nce_weight": 1.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_delta_add",
            "mapper": mapper,
            "procrustes_matrix": None,
            "embedding_update_mode": "delta_add",
            "default_blend_alpha": 0.20,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "contrastive_delta_add",
                "model_type": "transformer",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E13":
        # Proposed variant: learn residual direction from content model toward a stronger teacher.
        if reference_model is None:
            raise ValueError("Experiment E13 requires `reference_model` (e.g. model_baseline).")
        if reference_model.item_emb.weight.shape != model_for_projection.item_emb.weight.shape:
            raise ValueError("E13 requires matching embedding shapes between source and reference models.")

        residual_weight = (
            reference_model.item_emb.weight.detach() - model_for_projection.item_emb.weight.detach()
        )
        residual_teacher = SimpleNamespace(
            item_emb=SimpleNamespace(weight=residual_weight)
        )

        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_for_projection": residual_teacher,
                "model_type": "transformer",
                "mse_weight": max(float(mse_weight), 0.7),
                "cosine_weight": max(float(cosine_weight), 0.2),
                "nce_weight": 0.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_teacher_residual",
            "mapper": mapper,
            "procrustes_matrix": None,
            "embedding_update_mode": "delta_add",
            "default_blend_alpha": 0.20,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "teacher_residual_delta",
                "model_type": "transformer",
                "teacher": "reference_model_minus_source_model",
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E14":
        # MARec-style inspired variant:
        # train a transformer mapper, then fuse it with content-kNN anchor in collab space.
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "transformer_dim": 128,
                "batch_size": 256,
                "nce_temperature": 0.05,
                "sample_weight_power": 0.8,
                "mse_weight": 0.0,
                "cosine_weight": 0.0,
                "nce_weight": 1.0,
                "structure_weight": 0.0,
                "hard_negative_weight": 0.0,
                "hard_negative_top_k": 0,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_anchor_residual",
            "mapper": mapper,
            "procrustes_matrix": None,
            "postprocess": "anchor_residual",
            "anchor_top_k": 64,
            "anchor_temperature": 0.07,
            "anchor_blend_alpha": 0.50,
            "anchor_batch_size": 512,
            "embedding_update_mode": "blend",
            "default_blend_alpha": 1.0,
            "default_adapt_epochs": 0,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "contrastive_anchor_residual",
                "model_type": "transformer",
                "transformer_dim": 128,
                "batch_size": 256,
                "nce_temperature": 0.05,
                "sample_weight_power": 0.8,
                "anchor_top_k": 64,
                "anchor_temperature": 0.07,
                "anchor_blend_alpha": 0.50,
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E15":
        # Stronger pairwise-alignment variant:
        # preserve content-based cold init and learn a collaborative residual
        # with a richer transformer loss stack, then fuse with a content-kNN anchor.
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "transformer_dim": max(int(transformer_dim), 256),
                "n_layers": max(int(n_layers), 4),
                "n_heads": max(int(n_heads), 4),
                "token_count": max(int(token_count), 16),
                "batch_size": max(int(batch_size), 256),
                "nce_temperature": min(float(nce_temperature), 0.05),
                "sample_weight_power": max(float(sample_weight_power), 0.8),
                "mse_weight": max(float(mse_weight), 0.20),
                "cosine_weight": max(float(cosine_weight), 0.30),
                "nce_weight": max(float(nce_weight), 0.50),
                "structure_weight": max(float(structure_weight), 0.10),
                "hard_negative_weight": max(float(hard_negative_weight), 0.10),
                "hard_negative_top_k": max(int(hard_negative_top_k), 256),
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_anchor_residual_delta",
            "mapper": mapper,
            "procrustes_matrix": None,
            "postprocess": "anchor_residual",
            "anchor_top_k": 128,
            "anchor_temperature": 0.05,
            "anchor_blend_alpha": 0.60,
            "anchor_batch_size": 512,
            "embedding_update_mode": "delta_add",
            "default_blend_alpha": 0.35,
            "default_adapt_epochs": 0,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "multi_objective_anchor_residual_delta",
                "model_type": "transformer",
                "transformer_dim": max(int(transformer_dim), 256),
                "layers": max(int(n_layers), 4),
                "heads": max(int(n_heads), 4),
                "token_count": max(int(token_count), 16),
                "batch_size": max(int(batch_size), 256),
                "nce_temperature": min(float(nce_temperature), 0.05),
                "sample_weight_power": max(float(sample_weight_power), 0.8),
                "mse_weight": max(float(mse_weight), 0.20),
                "cosine_weight": max(float(cosine_weight), 0.30),
                "nce_weight": max(float(nce_weight), 0.50),
                "structure_weight": max(float(structure_weight), 0.10),
                "hard_negative_weight": max(float(hard_negative_weight), 0.10),
                "hard_negative_top_k": max(int(hard_negative_top_k), 256),
                "anchor_top_k": 128,
                "anchor_temperature": 0.05,
                "anchor_blend_alpha": 0.60,
                "embedding_update_mode": "delta_add",
                "blend_alpha": 0.35,
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    if exp_id == "E16" or exp_id.startswith("E16_"):
        # Norm-aware pairwise alignment:
        # learn raw collaborative vectors (not only directions), keep anchor fusion,
        # and add the result as a residual over the content-based cold initialization.
        exp_kwargs = dict(training_kwargs)
        exp_kwargs.update(
            {
                "model_type": "transformer",
                "transformer_dim": max(int(transformer_dim), 256),
                "n_layers": max(int(n_layers), 4),
                "n_heads": max(int(n_heads), 4),
                "token_count": max(int(token_count), 16),
                "batch_size": max(int(batch_size), 256),
                "nce_temperature": min(float(nce_temperature), 0.05),
                "sample_weight_power": max(float(sample_weight_power), 0.8),
                "mse_weight": max(float(mse_weight), 0.40),
                "cosine_weight": max(float(cosine_weight), 0.30),
                "nce_weight": max(float(nce_weight), 0.30),
                "structure_weight": max(float(structure_weight), 0.10),
                "hard_negative_weight": max(float(hard_negative_weight), 0.05),
                "hard_negative_top_k": max(int(hard_negative_top_k), 256),
                "normalize_targets": False,
            }
        )
        mapper = train_content_to_collab_transformer(**exp_kwargs)
        return {
            "experiment_id": exp_id,
            "mapper_type": "transformer_norm_aware_residual",
            "mapper": mapper,
            "procrustes_matrix": None,
            "postprocess": "anchor_residual",
            "anchor_top_k": 128,
            "anchor_temperature": 0.05,
            "anchor_blend_alpha": 0.60,
            "anchor_batch_size": 512,
            "embedding_update_mode": "delta_add",
            "default_blend_alpha": 0.35,
            "default_adapt_epochs": 0,
            "preserve_norms": True,
            "target_mode": str(target_mode).strip().lower(),
            "training_kwargs": {
                "objective": "norm_aware_anchor_residual_delta",
                "model_type": "transformer",
                "transformer_dim": max(int(transformer_dim), 256),
                "layers": max(int(n_layers), 4),
                "heads": max(int(n_heads), 4),
                "token_count": max(int(token_count), 16),
                "batch_size": max(int(batch_size), 256),
                "normalize_targets": False,
                "nce_temperature": min(float(nce_temperature), 0.05),
                "sample_weight_power": max(float(sample_weight_power), 0.8),
                "mse_weight": max(float(mse_weight), 0.40),
                "cosine_weight": max(float(cosine_weight), 0.30),
                "nce_weight": max(float(nce_weight), 0.30),
                "structure_weight": max(float(structure_weight), 0.10),
                "hard_negative_weight": max(float(hard_negative_weight), 0.05),
                "hard_negative_top_k": max(int(hard_negative_top_k), 256),
                "anchor_top_k": 128,
                "anchor_temperature": 0.05,
                "anchor_blend_alpha": 0.60,
                "embedding_update_mode": "delta_add",
                "blend_alpha": 0.35,
                "target_mode": str(target_mode).strip().lower(),
            },
        }

    raise ValueError(
        f"Unsupported experiment_id for alignment mapper: {experiment_id}. "
        "Expected one of E1..E8,E10,E12,E13,E14,E15,E16,E16_*,E3S."
    )


def predict_embeddings_for_experiment(
    experiment_bundle: dict,
    item_embeddings: dict,
    idx_to_item: dict,
    target_item_indices: list[int],
    device: torch.device,
    batch_size: int = 128,
    prediction_mode: str = "full",
    projector_bundle=None,
    output_dim: int | None = None,
):
    mapper_type = experiment_bundle.get("mapper_type")
    preserve_norms = bool(experiment_bundle.get("preserve_norms", False))
    target_mode = str(experiment_bundle.get("target_mode", "full")).strip().lower()
    prediction_mode = str(prediction_mode).strip().lower()
    if prediction_mode not in {"full", "delta"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")
    if mapper_type == "procrustes":
        matrix = experiment_bundle.get("procrustes_matrix")
        if matrix is None:
            return {}
        full_predictions = apply_procrustes_mapping(
            matrix,
            item_embeddings=item_embeddings,
            idx_to_item=idx_to_item,
            target_item_indices=target_item_indices,
        )
        if prediction_mode == "full":
            return full_predictions
        if output_dim is None:
            output_dim = len(next(iter(full_predictions.values()))) if full_predictions else 0
        out: dict[int, np.ndarray] = {}
        for item_idx, pred in full_predictions.items():
            item_id = idx_to_item.get(int(item_idx))
            if item_id is None or item_id not in item_embeddings:
                continue
            base_vec = _resolve_content_base_vector(
                item_embeddings[item_id],
                projector_bundle=projector_bundle,
                output_dim=int(output_dim),
            )
            out[int(item_idx)] = np.asarray(pred - base_vec, dtype=np.float32)
        return out

    mapper = experiment_bundle.get("mapper")
    if mapper is None:
        return {}
    return predict_embeddings_with_mapper(
        mapper,
        item_embeddings=item_embeddings,
        idx_to_item=idx_to_item,
        target_item_indices=target_item_indices,
        device=device,
        batch_size=batch_size,
        normalize_predictions=not preserve_norms,
        prediction_mode=prediction_mode,
        projector_bundle=projector_bundle,
        output_dim=output_dim,
        target_mode=target_mode,
    )


class ContentToCollaborativeTransformer(torch.nn.Module):
    """Transformer regressor that maps content embeddings to collaborative embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        token_count: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_count = int(token_count)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.token_count <= 0:
            raise ValueError("token_count must be > 0")
        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim должен делиться на num_heads")

        self.input_projection = torch.nn.Linear(self.input_dim, self.hidden_dim * self.token_count)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, self.token_count + 1, self.hidden_dim))
        self.dropout = torch.nn.Dropout(dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_output = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, content_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = content_embeddings.size(0)
        x = self.input_projection(content_embeddings.float())
        x = x.view(batch_size, self.token_count, self.hidden_dim)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embedding[:, : x.size(1), :]
        x = self.dropout(x)

        encoded = self.encoder(x)
        pooled = encoded[:, 0, :]
        return self.to_output(pooled)
