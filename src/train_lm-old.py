"""Train an RNN, LSTM, or causal Transformer on the existing ASRS splits.

Example: python src/train_lm.py --model lstm
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_data import (
    ROOT,
    NextTokenWindows,
    build_neural_vocab,
    collate_next_tokens,
    create_model,
    encode_reports,
    load_config,
    load_texts,
    model_config_for,
    parameter_count,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def epoch_pass(model, loader, pad_id, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    nll_total = 0.0
    token_total = 0
    start = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_sum = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            valid = (y != pad_id).sum()
            if training:
                (loss_sum / valid).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            nll_total += loss_sum.detach().item()
            token_total += valid.item()
    sync_device(device)
    return nll_total / token_total, token_total, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("rnn", "lstm", "transformer"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Device:", device)

    cfg = load_config(args.config)
    train_texts = load_texts(cfg, "train")
    val_texts = load_texts(cfg, "validation")
    vocab = build_neural_vocab(train_texts, cfg["tokenizer"]["min_frequency"])
    pad_id = vocab.index("<pad>")
    print(f"Training reports: {len(train_texts):,}; validation reports: {len(val_texts):,}")
    print(f"Vocabulary: {len(vocab):,} (including padding)")
    train_sequences = encode_reports(train_texts, vocab)
    val_sequences = encode_reports(val_texts, vocab)

    collate = partial(collate_next_tokens, pad_id=pad_id)
    train_loader = DataLoader(
        NextTokenWindows(train_sequences, args.context_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NextTokenWindows(val_sequences, args.context_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model_config = model_config_for(args.model, args.context_length)
    model = create_model(args.model, len(vocab), pad_id, model_config).to(device)
    lr = args.lr if args.lr is not None else (0.0005 if args.model == "transformer" else 0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_parameters = parameter_count(model)
    print(f"Model parameters: {n_parameters:,}; learning rate: {lr}")
    print(f"Train windows: {len(train_loader.dataset):,}; validation windows: {len(val_loader.dataset):,}")

    checkpoint_dir = ROOT / "checkpoints"
    result_dir = ROOT / "results" / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.model}.pt"

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    train_seconds = 0.0
    val_seconds = 0.0
    history = []
    sync_device(device)
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_ce, train_n, epoch_train_seconds = epoch_pass(model, train_loader, pad_id, device, optimizer)
        val_ce, val_n, epoch_val_seconds = epoch_pass(model, val_loader, pad_id, device)
        train_seconds += epoch_train_seconds
        val_seconds += epoch_val_seconds
        history.append({
            "epoch": epoch, "train_ce": train_ce, "validation_ce": val_ce,
            "train_tokens": train_n, "validation_tokens": val_n,
            "training_seconds": epoch_train_seconds,
            "validation_seconds": epoch_val_seconds,
        })
        print(f"Epoch {epoch:02d} | train CE {train_ce:.4f} | val CE {val_ce:.4f} | "
              f"val PPL {math.exp(val_ce):.2f} | train {epoch_train_seconds:.1f}s")
        if val_ce < best_val:
            best_val = val_ce
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_name": args.model,
                "model_config": model_config,
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "pad_id": pad_id,
                "best_epoch": epoch,
                "validation_cross_entropy_nats": val_ce,
                "context_length": args.context_length,
                "min_frequency": cfg["tokenizer"]["min_frequency"],
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation loss has not improved.")
                break

    sync_device(device)
    total_seconds = time.perf_counter() - run_start
    summary = {
        "model": args.model,
        "model_config": model_config,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_ce_nats": best_val,
        "best_validation_perplexity": math.exp(best_val),
        "embedding_dim": model_config["embedding_dim"],
        "vocabulary_size": len(vocab),
        "min_frequency": cfg["tokenizer"]["min_frequency"],
        "parameters": n_parameters,
        "training_seconds": train_seconds,
        "validation_seconds": val_seconds,
        "training_plus_validation_seconds": total_seconds,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "history": history,
    }
    with (result_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot([entry["epoch"] for entry in history], [entry["train_ce"] for entry in history], label="Train")
        plt.plot([entry["epoch"] for entry in history], [entry["validation_ce"] for entry in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Next-token cross-entropy (nats)")
        plt.title(f"{args.model.upper()}: learning curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / "loss_curves.png", dpi=160)
        plt.close()
    except ImportError:
        print("matplotlib not installed; skipping loss_curves.png")

    print(f"Saved best checkpoint: {checkpoint_path} (epoch {best_epoch})")
    print(f"Saved training metrics and curves: {result_dir}")


if __name__ == "__main__":
    main()
