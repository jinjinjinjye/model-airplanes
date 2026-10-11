"""Reuse the existing ASRS tokenizer/vocabulary and document-level splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import yaml

from dataset import build_vocab, tokenize
from models.rnn import RNNLanguageModel
from models.lstm import LSTMLanguageModel
from models.transformer import TransformerLanguageModel

ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict:
    with project_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_texts(config: dict, split: str) -> list[str]:
    filename = project_path(config["data"]["processed_dir"]) / f"{split}.csv"
    data = pd.read_csv(filename)
    column = config["data"]["processed_text_column"]
    return data[column].fillna("").astype(str).tolist()


def build_neural_vocab(train_texts: list[str], min_frequency: int) -> list[str]:
    vocab, _ = build_vocab(train_texts, min_frequency=min_frequency)
    vocab = list(vocab)
    for special in ("<unk>", "<bos>", "<eos>"):
        if special not in vocab:
            raise ValueError(f"Existing dataset.build_vocab() omitted {special}")
    if "<pad>" not in vocab:
        vocab.append("<pad>")
    return vocab


def encode_reports(texts: list[str], vocab: list[str]) -> list[list[int]]:
    word_to_id = {word: i for i, word in enumerate(vocab)}
    unk_id = word_to_id["<unk>"]
    bos_id = word_to_id["<bos>"]
    eos_id = word_to_id["<eos>"]
    encoded = []
    for text in texts:
        token_ids = [word_to_id.get(token, unk_id) for token in tokenize(text)]
        if token_ids:
            encoded.append([bos_id, *token_ids, eos_id])
    return encoded


class NextTokenWindows(Dataset):
    """Non-overlapping target windows: predict every token once, including EOS.

    Each window has context length <= block_size; recurrent states are reset
    between windows. No training example crosses a report boundary.
    """

    def __init__(self, reports: list[list[int]], block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.reports = reports
        self.block_size = block_size
        self.locations = [
            (report_index, start)
            for report_index, report in enumerate(reports)
            for start in range(0, len(report) - 1, block_size)
        ]

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        report_index, start = self.locations[index]
        tokens = self.reports[report_index]
        stop = min(start + self.block_size + 1, len(tokens))
        window = tokens[start:stop]
        return (
            torch.tensor(window[:-1], dtype=torch.long),
            torch.tensor(window[1:], dtype=torch.long),
        )


def collate_next_tokens(batch, pad_id: int):
    xs, ys = zip(*batch)
    return (
        pad_sequence(xs, batch_first=True, padding_value=pad_id),
        pad_sequence(ys, batch_first=True, padding_value=pad_id),
    )


def create_model(name: str, vocab_size: int, pad_id: int, model_config: dict):
    if name == "rnn":
        return RNNLanguageModel(vocab_size=vocab_size, pad_id=pad_id, **model_config)
    if name == "lstm":
        return LSTMLanguageModel(vocab_size=vocab_size, pad_id=pad_id, **model_config)
    if name == "transformer":
        return TransformerLanguageModel(vocab_size=vocab_size, pad_id=pad_id, **model_config)
    raise ValueError(f"Unknown model {name!r}")


def model_config_for(name: str, context_length: int, dropout: float = 0.2) -> dict:
    if name in ("rnn", "lstm"):
        return dict(embedding_dim=100, hidden_size=128, num_layers=2, dropout=dropout)
    if name == "transformer":
        return dict(
            embedding_dim=100,
            num_layers=2,
            num_heads=4,
            feedforward_dim=400,
            context_length=context_length,
            dropout=dropout,
        )
    raise ValueError(name)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
