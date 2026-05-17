"""Model definitions used in experiments."""

from __future__ import annotations

from typing import List
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


class SASRec(nn.Module):
    """SASRec model for sequential recommendation."""

    def __init__(
        self,
        num_items,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        max_len: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, hidden_units, padding_idx=0)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_units,
                    nhead=num_heads,
                    dim_feedforward=hidden_units * 4,
                    dropout=dropout_rate,
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def generate_square_subsequent_mask(self, sz: int):
        return torch.triu(
            torch.ones(sz, sz, device=self.item_emb.weight.device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, seq: torch.Tensor, pos: torch.Tensor):
        seq_emb = self.item_emb(seq)
        pos_emb = self.pos_emb(pos)
        hidden = self.layer_norm(self.dropout(seq_emb * (self.hidden_units ** 0.5) + pos_emb))

        mask = self.generate_square_subsequent_mask(seq.size(1))
        padding_mask = seq.eq(0)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask, src_key_padding_mask=padding_mask)
        return hidden

    def get_item_embeddings_for_scoring(self) -> torch.Tensor:
        return self.item_emb.weight

    def score_sequence_embeddings(self, seq_emb: torch.Tensor) -> torch.Tensor:
        item_emb = self.get_item_embeddings_for_scoring()[1:]
        return torch.matmul(seq_emb, item_emb.t())

    def predict(self, seq: torch.Tensor, pos: torch.Tensor):
        seq_emb = self.forward(seq, pos)
        last_emb = seq_emb[:, -1, :]
        return self.score_sequence_embeddings(last_emb)


class SASRecWithTrainableDelta(SASRec):
    """SASRec with additive trainable delta on a masked subset of items."""

    def __init__(
        self,
        num_items,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        max_len: int = 50,
        cold_item_indices: Sequence[int] | None = None,
        apply_to_all_items: bool = False,
        max_delta_norm: float | None = None,
    ):
        super().__init__(
            num_items=num_items,
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_len=max_len,
        )
        if apply_to_all_items:
            delta_indices = list(range(1, int(num_items) + 1))
        else:
            delta_indices = [
                int(item_idx)
                for item_idx in (cold_item_indices or [])
                if int(item_idx) > 0 and int(item_idx) <= num_items
            ]
            delta_indices = sorted(set(delta_indices))

        self.register_buffer(
            "_delta_mask",
            torch.zeros((num_items + 1, 1), dtype=torch.float32),
            persistent=False,
        )
        if delta_indices:
            self._delta_mask[torch.tensor(delta_indices, dtype=torch.long)] = 1.0

        # Trainable correction (delta) per-item, applied only where the mask allows it.
        self.delta = nn.Parameter(torch.zeros(num_items + 1, hidden_units, dtype=torch.float32))
        self.max_delta_norm = float(max_delta_norm) if max_delta_norm is not None else None

    def _embedding_with_delta(self):
        return self.item_emb.weight + self.delta * self._delta_mask

    def get_item_embeddings_for_scoring(self) -> torch.Tensor:
        return self._embedding_with_delta()

    def constrain_delta_(self) -> None:
        if self.max_delta_norm is None or self.max_delta_norm <= 0:
            return
        with torch.no_grad():
            delta = self.delta.data
            row_norms = delta.norm(dim=1, keepdim=True)
            scale = torch.clamp(self.max_delta_norm / (row_norms + 1e-8), max=1.0)
            delta.mul_(scale)
            delta.mul_(self._delta_mask)
            delta[0].zero_()

    def forward(self, seq: torch.Tensor, pos: torch.Tensor):
        seq_emb = F.embedding(seq, self._embedding_with_delta())
        pos_emb = self.pos_emb(pos)
        hidden = self.layer_norm(self.dropout(seq_emb * (self.hidden_units ** 0.5) + pos_emb))

        mask = self.generate_square_subsequent_mask(seq.size(1))
        padding_mask = seq.eq(0)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask, src_key_padding_mask=padding_mask)
        return hidden

    def predict(self, seq: torch.Tensor, pos: torch.Tensor):
        seq_emb = self.forward(seq, pos)
        last_emb = seq_emb[:, -1, :]
        return self.score_sequence_embeddings(last_emb)


class SequentialDataset(TorchDataset):
    """Torch dataset for fixed-length user sequences."""

    def __init__(self, sequences: List[List[int]], max_len: int = 50):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = list(self.sequences[idx])
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
            actual_len = self.max_len
        else:
            actual_len = len(seq)
            seq = [0] * (self.max_len - len(seq)) + seq

        padding_start = self.max_len - actual_len
        pos = []
        for i in range(self.max_len):
            if i < padding_start:
                pos.append(0)
            else:
                pos.append(i - padding_start + 1)

        return torch.LongTensor(seq), torch.LongTensor(pos)


def prepare_sequences(df, user_col='user_id', item_col='item_id_encoded', time_col='timestamp'):
    df_sorted = df.sort_values([user_col, time_col])
    sequences = []

    for _, group in df_sorted.groupby(user_col):
        seq = group[item_col].tolist()
        if len(seq) > 1:
            sequences.append(seq)

    return sequences
