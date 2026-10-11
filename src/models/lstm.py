"""Standard, unidirectional LSTM language model."""
from __future__ import annotations

import torch
from torch import nn


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        hidden_states, _ = self.lstm(embeddings)
        return self.dropout(hidden_states)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode(token_ids))
