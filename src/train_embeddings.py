"""Train ASRS CBOW vectors and prepare frozen input embeddings for Part II.

Run from the project root:
    python src/train_embeddings.py --gensim-cache E:\\gensim-data

Uses the EXACT Part I vocabulary from checkpoints/lstm.pt and the same
training texts / tokenization as src/train_lm.py (via neural_data.py).
Only the training split is used to train CBOW. Test/validation data are unused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset import SPECIAL_TOKENS, tokenize
from neural_data import ROOT, build_neural_vocab, load_config, load_texts


def vocab_digest(vocab: list[str]) -> str:
    payload = json.dumps(vocab, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TokenizedReports:
    """Re-iterable token stream for Gensim's multiple training epochs."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __iter__(self):
        for text in self.texts:
            tokens = tokenize(text)
            if tokens:
                yield tokens


def make_embedding_matrix(
    vocab: list[str], vectors, dimension: int, seed: int
) -> tuple[torch.Tensor, set[str]]:
    """Align word vectors to the exact LM vocabulary, with fixed OOV rows."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab), dimension)).astype(np.float32)
    available = set()
    for idx, token in enumerate(vocab):
        if token in SPECIAL_TOKENS:
            continue
        if token in vectors:
            vector = np.asarray(vectors[token], dtype=np.float32)
            if vector.shape != (dimension,):
                raise ValueError(f"Wrong vector size for {token!r}: {vector.shape}")
            matrix[idx] = vector
            available.add(token)
    matrix[vocab.index("<pad>")] = 0
    return torch.from_numpy(matrix), available


def coverage(vocab: list[str], counts: Counter, available: set[str]) -> dict:
    words = [word for word in vocab if word not in SPECIAL_TOKENS]
    matches = [word for word in words if word in available]
    matched_count = sum(counts[word] for word in matches)
    eligible_count = sum(counts[word] for word in words)
    missing = sorted((w for w in words if w not in available),
                     key=lambda w: (-counts[w], w))
    return {
        "lexical_types": len(words),
        "covered_types": len(matches),
        "missing_types": len(missing),
        "type_coverage": len(matches) / len(words) if words else 0.0,
        "eligible_training_tokens": eligible_count,
        "covered_training_tokens": matched_count,
        "token_weighted_coverage": matched_count / eligible_count if eligible_count else 0.0,
        "most_frequent_missing_words": missing[:25],
    }


def save_embeddings(
    path: Path, weight: torch.Tensor, vocab: list[str], variant: str,
    fallback_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weight": weight,
        "vocab": vocab,
        "embedding_dim": weight.shape[1],
        "variant": variant,
        "vocab_sha256": vocab_digest(vocab),
        "oov_initialization": "independent N(0, 0.1^2), fixed seed",
        "oov_seed": fallback_seed,
        "padding_row": "zeros",
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--baseline", default="checkpoints/lstm.pt",
                        help="Existing Part I LSTM checkpoint; needed for vocabulary alignment")
    parser.add_argument("--mode", choices=("both", "cbow", "glove"), default="both")
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gensim-cache", default=None)
    parser.add_argument("--output-dir", default="checkpoints/embeddings")
    parser.add_argument("--results-dir", default="results/part2")
    args = parser.parse_args()

    def project_path(value: str) -> Path:
        value = Path(value)
        return value if value.is_absolute() else ROOT / value

    if args.gensim_cache:
        # Must be set before importing gensim.downloader.
        os.environ["GENSIM_DATA_DIR"] = str(project_path(args.gensim_cache))

    baseline_path = project_path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Part I checkpoint not found: {baseline_path}")
    ckpt = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if ckpt.get("model_name") != "lstm":
        raise ValueError("Part I checkpoint is not an LSTM")
    baseline_vocab = list(ckpt["vocab"])
    baseline_dim = int(ckpt["model_config"]["embedding_dim"])
    if args.dimension != baseline_dim:
        raise ValueError(f"Requested {args.dimension}d but Part I LSTM uses {baseline_dim}d")
    if baseline_dim != 100:
        raise ValueError("This script's GloVe 6B variant requires 100-dimensional embeddings")

    config = load_config(str(project_path(args.config)))
    texts = load_texts(config, "train")
    vocab = list(build_neural_vocab(texts, config["tokenizer"]["min_frequency"]))
    if vocab != baseline_vocab:
        raise ValueError(
            "Reconstructed vocabulary does not match Part I checkpoint. "
            "Do not train on a different split or token ordering."
        )
    if ckpt["pad_id"] != vocab.index("<pad>"):
        raise ValueError("Part I padding ID does not match")

    counts = Counter()
    for tokens in TokenizedReports(texts):
        counts.update(tokens)

    out_dir = project_path(args.output_dir)
    result_dir = project_path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training reports: {len(texts):,}; vocabulary: {len(vocab):,}")
    print(f"Matched Part I LSTM vocabulary (SHA-256: {vocab_digest(vocab)[:16]}...)")

    summary_path = result_dir / "embedding_coverage.json"
    summary = {
        "source": "ASRS processed training split only",
        "baseline_checkpoint": str(baseline_path.relative_to(ROOT)) if baseline_path.is_relative_to(ROOT) else str(baseline_path),
        "vocab_sha256": vocab_digest(vocab),
        "vocabulary_size": len(vocab),
        "dimension": args.dimension,
        "fallback_seed": args.seed,
        "variants": {},
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("vocab_sha256") == summary["vocab_sha256"]:
            summary["variants"] = previous.get("variants", {})

    if args.mode in ("both", "cbow"):
        from gensim.models import Word2Vec
        start = time.perf_counter()
        cbow = Word2Vec(
            sentences=TokenizedReports(texts),
            vector_size=args.dimension,
            window=args.window,
            min_count=int(config["tokenizer"]["min_frequency"]),
            sg=0, cbow_mean=1, negative=args.negative, hs=0,
            epochs=args.epochs, sample=1e-3,
            seed=args.seed, workers=1,
        )
        seconds = time.perf_counter() - start
        cbow.save(str(out_dir / "cbow_100d.model"))
        matrix, matched = make_embedding_matrix(vocab, cbow.wv, args.dimension, args.seed)
        save_embeddings(out_dir / "cbow_100d.pt", matrix, vocab, "cbow", args.seed)
        summary["variants"]["cbow"] = {
            **coverage(vocab, counts, matched),
            "training_seconds": seconds,
            "epochs": args.epochs,
            "context_window": args.window,
            "negative_samples": args.negative,
            "min_count": int(config["tokenizer"]["min_frequency"]),
            "sg": 0,
            "workers": 1,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"CBOW trained in {seconds:.1f}s; matched {len(matched):,} word types")

    if args.mode in ("both", "glove"):
        import gensim.downloader as api
        print("Loading GloVe 6B 100d (may download on first use)...")
        start = time.perf_counter()
        glove = api.load("glove-wiki-gigaword-100")
        seconds = time.perf_counter() - start
        matrix, matched = make_embedding_matrix(vocab, glove, args.dimension, args.seed)
        save_embeddings(out_dir / "glove_100d.pt", matrix, vocab, "glove", args.seed)
        summary["variants"]["glove"] = {
            **coverage(vocab, counts, matched),
            "loading_seconds": seconds,
            "pretrained_source": "glove-wiki-gigaword-100",
        }
        print(f"GloVe loaded in {seconds:.1f}s; matched {len(matched):,} word types")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved matrices under {out_dir} and coverage report to {summary_path}")


if __name__ == "__main__":
    main()
