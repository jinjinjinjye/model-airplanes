"""Small, causal (decoder-only) Transformer language model."""
from __future__ import annotations

import torch
from torch import nn


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 100,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 400,
        context_length: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, length = token_ids.shape
        if length > self.context_length:
            raise ValueError("Input exceeds context_length")
        positions = torch.arange(length, device=token_ids.device)
        x = self.embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        # True masks out attention to future positions.
        causal_mask = torch.ones(length, length, device=token_ids.device, dtype=torch.bool).triu(1)
        # Right-side padding needs no key-padding mask: real tokens cannot
        # attend to future padding due to the causal mask. Padded outputs
        # are ignored by cross-entropy loss.
        return self.norm(self.transformer(x, mask=causal_mask))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode(token_ids))
