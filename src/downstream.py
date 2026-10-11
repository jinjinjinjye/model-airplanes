"""Part III: frozen LSTM vs Transformer representations for ASRS incident classification.

Put this file in src/downstream.py. Run from the repository root:

    python src/downstream.py --prepare-only
    python src/downstream.py --device auto --batch-size 32

Expected existing repository components:
    src/dataset.py: tokenize
    src/neural_data.py: ROOT, load_config, create_model
    data/processed/{train,validation,test}.csv: acn, text
    checkpoints/{lstm,transformer}.pt: Part I state_dict, model_config, vocab, pad_id
    config.yaml: data.raw_path, data.processed_dir

This implementation uses one identical extraction procedure for both architectures:
mean-pool the context-dependent hidden states for non-special tokens, using
nonoverlapping windows of common length, then pool all windows for a report.
The language models stay frozen. Only simple logistic-regression classifiers
are fitted; no test labels are used for model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import tokenize
from neural_data import ROOT, create_model, load_config

CLASSES = ("Aircraft", "Human Factors", "Procedure", "Weather", "Airport")
SPLITS = ("train", "validation", "test")
LABEL_COLUMN = "Assessments | Primary Problem"
ID_COLUMN = "ACN"
C_VALUES = (0.01, 0.1, 1.0, 10.0)


def file_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(ROOT) / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_acn(series: pd.Series) -> pd.Series:
    # Numeric IDs avoid CSV int/float/string differences; disallow invalid identifiers.
    numbers = pd.to_numeric(series, errors="raise")
    if numbers.isna().any() or not (numbers % 1 == 0).all():
        raise ValueError("ACNs must be present and integer-valued")
    return numbers.astype("int64")


def load_labeled_splits(config: dict, raw_csv: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Labeled ASRS CSV not found: {raw_csv}")

    labels = pd.read_csv(
        raw_csv, usecols=[ID_COLUMN, LABEL_COLUMN], low_memory=False
    )
    labels["acn"] = normalize_acn(labels[ID_COLUMN])
    if labels["acn"].duplicated().any():
        raise ValueError("Duplicate ACNs in labeled ASRS CSV; cannot safely merge")
    labels = labels[["acn", LABEL_COLUMN]].rename(columns={LABEL_COLUMN: "label"})
    raw_counts = labels["label"].value_counts(dropna=False).to_dict()

    processed_dir = file_path(config["data"]["processed_dir"])
    text_col = config["data"].get("processed_text_column", "text")
    id_col = config["data"].get("processed_id_column", "acn")
    merged: dict[str, pd.DataFrame] = {}
    missing_labels = {}

    for split in SPLITS:
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Part I split file: {csv_path}")
        frame = pd.read_csv(csv_path, usecols=[id_col, text_col])
        frame = frame.rename(columns={id_col: "acn", text_col: "text"})
        frame["acn"] = normalize_acn(frame["acn"])
        if frame["acn"].duplicated().any():
            raise ValueError(f"Duplicate ACNs in {split}.csv")
        frame = frame.merge(labels, on="acn", how="left", validate="one_to_one", indicator=True)
        unmatched = frame["_merge"] != "both"
        missing_labels[split] = int(unmatched.sum())
        if unmatched.any():
            raise ValueError(
                f"{split}.csv has {unmatched.sum()} ACNs absent from the labeled CSV. "
                f"First unmatched: {frame.loc[unmatched, 'acn'].head().tolist()}. "
                "Provide the same original merged ASRS data used for Part I."
            )
        frame = frame.drop(columns="_merge")
        frame["text"] = frame["text"].fillna("").astype(str).str.strip()
        merged[split] = frame

    # Absolutely preserve Part I's original train/validation/test partitions.
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = set(merged[left]["acn"]) & set(merged[right]["acn"])
            if overlap:
                raise ValueError(f"ACN overlap between {left} and {right}: {len(overlap)}")

    n_before = {s: len(merged[s]) for s in SPLITS}
    dropped_other = {}
    for split in SPLITS:
        frame = merged[split]
        keep = frame["label"].isin(CLASSES) & frame["text"].ne("")
        dropped_other[split] = int((~keep).sum())
        merged[split] = frame.loc[keep].reset_index(drop=True)

    # Detect exact text leakage, including across different ACNs and whitespace.
    # Conflicting-label duplicate texts are excluded completely rather than arbitrarily relabeled.
    for split in SPLITS:
        merged[split]["_hash"] = merged[split]["text"].map(
            lambda t: hashlib.sha256(" ".join(t.lower().split()).encode("utf-8")).hexdigest()
        )
    all_labeled = pd.concat(merged.values(), ignore_index=True)
    conflicting_hashes = set(
        all_labeled.groupby("_hash")["label"].nunique().loc[lambda s: s > 1].index
    )
    removed_conflicts = {}
    removed_duplicates = {}
    seen: set[str] = set()
    for split in SPLITS:
        frame = merged[split]
        mask_conflict = frame["_hash"].isin(conflicting_hashes)
        removed_conflicts[split] = int(mask_conflict.sum())
        frame = frame.loc[~mask_conflict].copy()
        mask_duplicate = frame["_hash"].isin(seen) | frame["_hash"].duplicated()
        removed_duplicates[split] = int(mask_duplicate.sum())
        frame = frame.loc[~mask_duplicate].copy()
        seen.update(frame["_hash"].tolist())
        merged[split] = frame.drop(columns="_hash").reset_index(drop=True)

    for split in SPLITS:
        missing_classes = set(CLASSES) - set(merged[split]["label"])
        if missing_classes:
            raise ValueError(f"No examples for {sorted(missing_classes)} in {split} split")

    audit = {
        "raw_label_distribution": {str(k): int(v) for k, v in raw_counts.items()},
        "processed_original_rows": n_before,
        "removed_outside_top_five_or_empty_text": dropped_other,
        "removed_conflicting_duplicate_texts": removed_conflicts,
        "removed_repeated_texts": removed_duplicates,
        "missing_raw_ACNs": missing_labels,
        "usable_rows": {s: len(merged[s]) for s in SPLITS},
        "usable_class_counts": {
            s: {c: int((merged[s]["label"] == c).sum()) for c in CLASSES}
            for s in SPLITS
        },
    }
    return merged, audit


def dataset_fingerprint(splits: dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for split in SPLITS:
        h.update(split.encode("utf-8"))
        for row in splits[split].itertuples(index=False):
            h.update(f"{row.acn}\x00{row.label}\x00{row.text}\n".encode("utf-8"))
    return h.hexdigest()


def load_checkpoint(path: Path, architecture: str, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {architecture} checkpoint: {path}\n"
            f"Train it first using: python src/train_lm.py --model {architecture}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != architecture:
        raise ValueError(f"Checkpoint {path} is not a {architecture} language model")
    vocab = list(checkpoint["vocab"])
    if len(set(vocab)) != len(vocab):
        raise ValueError("Checkpoint vocabulary has duplicate tokens")
    token_to_id = {token: i for i, token in enumerate(vocab)}
    for special in ("<pad>", "<unk>", "<bos>", "<eos>"):
        if special not in token_to_id:
            raise ValueError(f"Checkpoint {path} missing special token {special}")
    pad_id = int(checkpoint["pad_id"])
    if pad_id != token_to_id["<pad>"]:
        raise ValueError("Checkpoint padding ID and vocabulary disagree")
    context_length = int(checkpoint["context_length"])
    model = create_model(architecture, len(vocab), pad_id, checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, token_to_id, context_length, checkpoint


class HiddenStateReader:
    """Prefer architecture's encode(); otherwise intercept states before its LM head."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hidden = None
        self._hook = None
        if callable(getattr(model, "encode", None)):
            self.method = "encode"
        else:
            head = None
            for name in ("output", "lm_head", "head", "output_layer", "fc_out", "fc"):
                layer = getattr(model, name, None)
                if isinstance(layer, torch.nn.Linear):
                    head = layer
                    break
            if head is None:
                raise RuntimeError(
                    "Model has no .encode() and no recognized final output Linear layer. "
                    "Add an encode(token_ids) method returning [batch, time, hidden]."
                )
            self.method = f"pre-hook:{name}"
            self._hook = head.register_forward_pre_hook(self._capture)

    def _capture(self, _module, args):
        self._hidden = args[0]

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.method == "encode":
            states = self.model.encode(token_ids)
        else:
            self._hidden = None
            _ = self.model(token_ids)
            states = self._hidden
        if isinstance(states, (tuple, list)):
            states = states[0]
        if not isinstance(states, torch.Tensor) or states.ndim != 3:
            raise RuntimeError("Expected contextual states with shape [batch, tokens, features]")
        if states.shape[:2] != token_ids.shape:
            raise RuntimeError(f"Shape mismatch: hidden={states.shape}, tokens={token_ids.shape}")
        return states

    def close(self):
        if self._hook is not None:
            self._hook.remove()


def iter_batches(
    texts: list[str], token_to_id: dict[str, int], chunk_length: int,
    max_chunks: int, batch_size: int, pad_id: int,
):
    """Emit (padded token IDs, lexical masks, source report IDs), without mixing reports."""
    unk = token_to_id["<unk>"]
    bos = token_to_id["<bos>"]
    eos = token_to_id["<eos>"]
    tokens_batch = []
    mask_batch = []
    owner_batch = []
    max_lexical = max_chunks * chunk_length - 2
    if max_lexical < 1:
        raise ValueError("max_chunks * chunk_length must exceed 2")

    for owner, text in enumerate(texts):
        lexical = tokenize(text)
        lexical = lexical[:max_lexical]
        seq = [bos] + [token_to_id.get(tok, unk) for tok in lexical] + [eos]
        for begin in range(0, len(seq), chunk_length):
            chunk = seq[begin:begin + chunk_length]
            lexical_mask = [int(token not in (bos, eos, pad_id)) for token in chunk]
            tokens_batch.append(chunk)
            mask_batch.append(lexical_mask)
            owner_batch.append(owner)
            if len(tokens_batch) == batch_size:
                yield pad_batch(tokens_batch, mask_batch, owner_batch, pad_id)
                tokens_batch, mask_batch, owner_batch = [], [], []
    if tokens_batch:
        yield pad_batch(tokens_batch, mask_batch, owner_batch, pad_id)


def pad_batch(token_seqs, masks, owners, pad_id):
    length = max(map(len, token_seqs))
    tokens = torch.full((len(token_seqs), length), pad_id, dtype=torch.long)
    lexical = torch.zeros((len(token_seqs), length), dtype=torch.bool)
    for i, (seq, mask) in enumerate(zip(token_seqs, masks)):
        tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        lexical[i, :len(mask)] = torch.tensor(mask, dtype=torch.bool)
    return tokens, lexical, np.asarray(owners, dtype=np.int64)


def extract_features(
    model: torch.nn.Module, vocab: dict[str, int], split: pd.DataFrame,
    chunk_length: int, max_chunks: int, batch_size: int, device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, dict]:
    texts = split["text"].tolist()
    n_docs = len(texts)
    if not n_docs:
        raise ValueError(f"Empty split: {split_name}")
    max_lexical = chunk_length * max_chunks - 2
    lengths = np.asarray([len(tokenize(t)) for t in texts], dtype=np.int64)
    if (lengths == 0).any():
        raise ValueError(f"Empty tokenization in {split_name}")
    reader = HiddenStateReader(model)
    accum = None
    counts = np.zeros(n_docs, dtype=np.float64)
    n_windows = 0
    start = time.perf_counter()
    print(f"  {split_name}: {n_docs:,} reports; extractor={reader.method}", flush=True)
    try:
        with torch.inference_mode():
            for tokens, lexical, owners in iter_batches(
                texts, vocab, chunk_length, max_chunks, batch_size, vocab["<pad>"]
            ):
                token_tensor = tokens.to(device, non_blocking=True)
                lexical_tensor = lexical.to(device, non_blocking=True)
                hidden = reader(token_tensor)
                masked_sum = (hidden.float() * lexical_tensor.unsqueeze(-1)).sum(dim=1)
                sums = masked_sum.detach().cpu().numpy().astype(np.float64, copy=False)
                valid_counts = lexical.sum(dim=1).numpy().astype(np.float64, copy=False)
                if accum is None:
                    accum = np.zeros((n_docs, sums.shape[1]), dtype=np.float64)
                np.add.at(accum, owners, sums)
                np.add.at(counts, owners, valid_counts)
                n_windows += len(owners)
                if n_windows % (batch_size * 100) < batch_size:
                    print(f"    processed {n_windows:,} windows", flush=True)
    finally:
        reader.close()
    if accum is None or (counts <= 0).any():
        raise RuntimeError(f"Zero valid hidden-state tokens in split {split_name}")
    features = (accum / counts[:, None]).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError(f"Non-finite representations in {split_name}")
    duration = time.perf_counter() - start
    print(f"    done: {n_windows:,} windows, {duration:.1f}s, {features.shape[1]} features")
    details = {
        "reports": n_docs,
        "windows": n_windows,
        "feature_dimension": features.shape[1],
        "truncated_reports": int((lengths > max_lexical).sum()),
        "fraction_truncated": float((lengths > max_lexical).mean()),
        "retained_token_fraction": float(np.minimum(lengths, max_lexical).sum() / lengths.sum()),
        "max_lexical_tokens": max_lexical,
        "extraction_seconds": duration,
        "extractor_method": reader.method,
    }
    return features, details


def get_all_features(
    model, vocab, splits, ckpt_path: Path, data_fingerprint: str,
    chunk_length, max_chunks, batch_size, device, output_dir, arch: str, no_cache: bool
):
    stamp = (
        f"{arch}_{sha256_file(ckpt_path)[:16]}_{data_fingerprint[:16]}"
        f"_L{chunk_length}_C{max_chunks}"
    )
    cache_path = output_dir / "cache" / f"{stamp}.npz"
    if not no_cache and cache_path.exists():
        print(f"Using feature cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as f:
            all_features = {s: f[s].copy() for s in SPLITS}
            details = json.loads(str(f["details"].item()))
        for split in SPLITS:
            if len(all_features[split]) != len(splits[split]):
                raise RuntimeError(f"Outdated cache for {arch}; rerun with --no-cache")
        return all_features, details
    all_features, details = {}, {}
    for split in SPLITS:
        all_features[split], details[split] = extract_features(
            model, vocab, splits[split], chunk_length, max_chunks,
            batch_size, device, split
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, **all_features, details=json.dumps(details)
    )
    print(f"Cached features: {cache_path}")
    return all_features, details


def train_classifier(features, splits, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    y = {split: splits[split]["label"].to_numpy() for split in SPLITS}
    best_c = None
    best_score = -1.0
    validation_results = {}

    for c in C_VALUES:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed),
        )
        classifier.fit(features["train"], y["train"])
        prediction = classifier.predict(features["validation"])
        score = float(f1_score(y["validation"], prediction, labels=list(CLASSES), average="macro", zero_division=0))
        validation_results[str(c)] = score
        print(f"    C={c:g}: validation macro-F1={score:.4f}")
        if score > best_score + 1e-12:
            best_score, best_c = score, c

    # Validation is used for choosing C; refit on train+validation, test once.
    trainval_x = np.concatenate([features["train"], features["validation"]])
    trainval_y = np.concatenate([y["train"], y["validation"]])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(trainval_x, trainval_y)
    y_pred = classifier.predict(features["test"])
    probability = classifier.predict_proba(features["test"])
    model_classes = classifier.named_steps["logisticregression"].classes_.tolist()
    assert set(model_classes) == set(CLASSES)
    metrics = {
        "selected_C": best_c,
        "validation_macro_f1_by_C": validation_results,
        "best_validation_macro_f1": best_score,
        "test_accuracy": float(accuracy_score(y["test"], y_pred)),
        "test_macro_f1": float(f1_score(y["test"], y_pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "test_report": classification_report(
            y["test"], y_pred, labels=list(CLASSES), output_dict=True, zero_division=0
        ),
        "confusion_matrix_classes": list(CLASSES),
        "confusion_matrix": confusion_matrix(y["test"], y_pred, labels=list(CLASSES)).tolist(),
        "classifier_probabilities_order": model_classes,
    }
    return metrics, y_pred, probability


def write_predictions(splits, prediction, proba, metrics, arch: str, output: Path):
    test = splits["test"].copy()
    test["predicted"] = prediction
    test["confidence"] = np.max(proba, axis=1)
    test["correct"] = test["label"] == test["predicted"]
    prediction_cols = ["acn", "label", "predicted", "correct", "confidence", "text"]
    test[prediction_cols].to_csv(output / f"{arch}_test_predictions.csv", index=False)
    wrong = test.loc[~test["correct"]].copy()
    wrong = wrong.sort_values("confidence", ascending=False)
    wrong[prediction_cols].to_csv(output / f"{arch}_all_errors.csv", index=False)

    # Representative high-confidence mistakes for each true class, not just majority classes.
    sampled = (
        wrong.groupby("label", sort=False, group_keys=False)
        .head(3)
        .copy()
    )
    sampled["text_excerpt"] = sampled["text"].map(
        lambda text: " ".join(text.split())[:700]
    )
    sampled[["acn", "label", "predicted", "confidence", "text_excerpt"]].to_csv(
        output / f"{arch}_representative_errors.csv", index=False
    )
    return len(wrong)


def plot_results(metrics_by_model: dict, output: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping figures")
        return

    for arch, metrics in metrics_by_model.items():
        confusion = np.asarray(metrics["confusion_matrix"])
        normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
        ax.set_xlabel("Predicted Primary Problem")
        ax.set_ylabel("True Primary Problem")
        ax.set_title(f"{arch.upper()} normalized confusion matrix")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                        color="white" if normalized[i, j] > 0.6 else "black")
        fig.colorbar(im, ax=ax, label="Fraction of true class")
        fig.tight_layout()
        fig.savefig(output / f"{arch}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    names = list(metrics_by_model)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(names))
    width = 0.36
    ax.bar(positions - width / 2, [metrics_by_model[n]["test_accuracy"] for n in names], width, label="Accuracy")
    ax.bar(positions + width / 2, [metrics_by_model[n]["test_macro_f1"] for n in names], width, label="Macro-F1")
    ax.set_xticks(positions, [n.upper() for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "part3_comparison.png", dpi=160)
    plt.close(fig)


def load_lm_test_metrics(architecture: str) -> dict | None:
    path = Path(ROOT) / "results" / architecture / "metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if "test_perplexity" not in data:
        return None
    return {
        "test_perplexity": data["test_perplexity"],
        "test_cross_entropy_nats": data.get("test_cross_entropy_nats"),
        "source": str(path.relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-csv", default=None,
                        help="Original labeled ASRS CSV; defaults to data.raw_path in config.yaml")
    parser.add_argument("--lstm-checkpoint", default="checkpoints/lstm.pt")
    parser.add_argument("--transformer-checkpoint", default="checkpoints/transformer.pt")
    parser.add_argument("--output-dir", default="results/part3")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=None,
                        help="Tokens per window; defaults to the minimum of both checkpoint context lengths")
    parser.add_argument("--max-chunks", type=int, default=4,
                        help="Maximum number of nonoverlapping windows per report (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate labels, ACNs, partitions, and class counts without checkpoints")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_chunks < 1:
        parser.error("batch-size and max-chunks must be >= 1")
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but PyTorch CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    print("Device:", device)
    config = load_config(str(file_path(args.config)))
    raw_csv = file_path(args.raw_csv or config["data"]["raw_path"])
    splits, audit = load_labeled_splits(config, raw_csv)
    out = file_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Labeled CSV: {raw_csv}")
    print("Top-five incident categories (after duplicate removal):")
    print(pd.DataFrame(audit["usable_class_counts"]).to_string())
    print("Rows per split:", audit["usable_rows"])
    with (out / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    if args.prepare_only:
        print(f"Preparation completed. Audit: {out / 'dataset_audit.json'}")
        return

    ckpt_paths = {
        "lstm": file_path(args.lstm_checkpoint),
        "transformer": file_path(args.transformer_checkpoint),
    }
    context_lengths = {}
    for arch, path in ckpt_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        context_lengths[arch] = int(ckpt["context_length"])
    chunk_length = args.chunk_length or min(context_lengths.values())
    if not 4 <= chunk_length <= min(context_lengths.values()):
        raise ValueError(
            "chunk-length must be >= 4 and no greater than either checkpoint's context_length: "
            f"{context_lengths}"
        )
    print(f"Common context window: {chunk_length}; max chunks/report: {args.max_chunks}")
    fingerprint = dataset_fingerprint(splits)
    all_results = {}
    representations = {}
    for arch in ("lstm", "transformer"):
        print(f"\n=== {arch.upper()} ===", flush=True)
        model, vocab, _, checkpoint = load_checkpoint(ckpt_paths[arch], arch, device)
        print(f"Loaded {len(vocab):,}-token vocabulary from: {ckpt_paths[arch]}")
        feature_dict, extraction = get_all_features(
            model, vocab, splits, ckpt_paths[arch], fingerprint,
            chunk_length, args.max_chunks, args.batch_size, device, out, arch, args.no_cache
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("Logistic-regression validation grid search:")
        metrics, pred, prob = train_classifier(feature_dict, splits, args.seed)
        n_errors = write_predictions(splits, pred, prob, metrics, arch, out)
        metrics["extraction"] = extraction
        metrics["lm_test_metrics"] = load_lm_test_metrics(arch)
        metrics["test_errors"] = n_errors
        metrics["checkpoint"] = str(ckpt_paths[arch])
        all_results[arch] = metrics
        representations[arch] = {split: list(feature_dict[split].shape) for split in SPLITS}
        print(f"Held-out test: accuracy={metrics['test_accuracy']:.4f}, "
              f"macro-F1={metrics['test_macro_f1']:.4f}", flush=True)

    majority_class = Counter(splits["train"]["label"]).most_common(1)[0][0]
    y_test = splits["test"]["label"].tolist()
    baseline = {
        "majority_class": majority_class,
        "test_accuracy": float(np.mean([x == majority_class for x in y_test])),
        "test_macro_f1": float(f1_score(
            y_test, [majority_class] * len(y_test), labels=list(CLASSES),
            average="macro", zero_division=0
        )),
    }
    summary = {
        "task": "ASRS five-class primary problem classification",
        "classes": list(CLASSES),
        "seed": args.seed,
        "data": audit,
        "data_fingerprint": fingerprint,
        "context_window": chunk_length,
        "max_chunks_per_report": args.max_chunks,
        "pooling": "token-weighted mean of lexical hidden states across nonoverlapping windows",
        "freeze_language_models": True,
        "classifier": "StandardScaler + class-balanced multinomial logistic regression",
        "C_grid": list(C_VALUES),
        "selection_metric": "validation macro-F1",
        "refit": "train+validation after selecting C",
        "majority_baseline": baseline,
        "results": all_results,
        "representations": representations,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    flat_rows = [
        {"model": "Majority baseline", "test_accuracy": baseline["test_accuracy"],
         "test_macro_f1": baseline["test_macro_f1"], "selected_C": np.nan,
         "lm_test_perplexity": np.nan}
    ]
    for arch, result in all_results.items():
        flat_rows.append({
            "model": arch, "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"], "selected_C": result["selected_C"],
            "lm_test_perplexity": (result["lm_test_metrics"] or {}).get("test_perplexity", np.nan),
        })
    pd.DataFrame(flat_rows).to_csv(out / "comparison.csv", index=False)
    plot_results(all_results, out)
    print("\n=== COMPARISON ===")
    print(pd.DataFrame(flat_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
