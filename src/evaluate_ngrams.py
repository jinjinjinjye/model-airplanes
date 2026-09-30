from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import re
import statistics
import time
from pathlib import Path

import pandas as pd
import yaml

from dataset import build_vocab, tokenize
from models.ngram import NGramLanguageModel


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return ROOT / path


def load_split(
    processed_dir: str | Path,
    split: str,
) -> pd.DataFrame:
    path = project_path(processed_dir) / f"{split}.csv"
    return pd.read_csv(path)


def make_sequences(
    texts: list[str],
    vocab: list[str],
) -> list[list[str]]:
    """
    Tokenize documents and map tokens outside the vocabulary to <unk>.

    Each report remains separate so n-gram context never crosses report
    boundaries.
    """
    vocab_set = set(vocab)
    sequences = []

    for text in texts:
        tokens = tokenize(text)

        tokens = [
            token if token in vocab_set else "<unk>"
            for token in tokens
        ]

        if not tokens:
            continue

        sequences.append(
            ["<bos>", *tokens, "<eos>"]
        )

    return sequences


def count_prediction_tokens(
    sequences: list[list[str]],
) -> int:
    """
    Number of next-token predictions in a collection of sequences.
    """
    return sum(
        max(len(sequence) - 1, 0)
        for sequence in sequences
    )


def evaluate_cross_entropy(
    model: NGramLanguageModel,
    sequences: list[list[str]],
) -> tuple[float, float, int]:
    """
    Evaluate average next-token cross-entropy in nats.

    Perplexity = exp(cross_entropy).
    """
    total_negative_log_likelihood = 0.0
    token_count = 0
    context_length = model.context_length

    for sequence in sequences:
        for i in range(1, len(sequence)):
            target = sequence[i]

            start = max(0, i - context_length)
            context = sequence[start:i]

            probability = model.probability(
                target=target,
                context=context,
            )

            if probability <= 0:
                raise RuntimeError(
                    f"Model assigned zero probability to {target!r}"
                )

            total_negative_log_likelihood -= math.log(
                probability
            )

            token_count += 1

    if token_count == 0:
        raise RuntimeError(
            "No prediction tokens were available for evaluation."
        )

    cross_entropy = (
        total_negative_log_likelihood / token_count
    )

    perplexity = math.exp(cross_entropy)

    return cross_entropy, perplexity, token_count


def build_benchmark_workload(
    sequences: list[list[str]],
    context_length: int,
    max_tokens: int,
) -> list[tuple[list[str], str]]:
    """
    Build a fixed inference workload without copying complete report prefixes.
    """
    workload = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            start = max(0, i - context_length)

            workload.append(
                (
                    sequence[start:i],
                    sequence[i],
                )
            )

            if len(workload) >= max_tokens:
                return workload

    return workload


def benchmark_inference(
    model: NGramLanguageModel,
    workload: list[tuple[list[str], str]],
    repeats: int,
) -> dict:
    """
    Time exactly the same next-token probability workload repeatedly.

    Tokenization and disk I/O are excluded.
    """
    if not workload:
        raise RuntimeError(
            "Inference benchmark workload is empty."
        )

    warmup_size = min(100, len(workload))

    for context, target in workload[:warmup_size]:
        model.probability(
            target=target,
            context=context,
        )

    elapsed_times = []

    for _ in range(repeats):
        start = time.perf_counter()

        for context, target in workload:
            model.probability(
                target=target,
                context=context,
            )

        elapsed_times.append(
            time.perf_counter() - start
        )

    median_seconds = statistics.median(
        elapsed_times
    )

    throughput = (
        len(workload) / median_seconds
    )

    return {
        "tokens": len(workload),
        "repeats": repeats,
        "times_seconds": elapsed_times,
        "median_seconds": median_seconds,
        "tokens_per_second": throughput,
    }


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)

    text = re.sub(
        r"\s+([.,!?;:%)\]}])",
        r"\1",
        text,
    )

    text = re.sub(
        r"([(\[{])\s+",
        r"\1",
        text,
    )

    return text


def generate_samples(
    model: NGramLanguageModel,
    prompts: list[str],
    vocab: list[str],
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    vocab_set = set(vocab)
    outputs = []

    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenize(prompt)

        prompt_tokens = [
            token if token in vocab_set else "<unk>"
            for token in prompt_tokens
        ]

        generated_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            seed=seed + i,
        )

        outputs.append(
            {
                "prompt": prompt,
                "generated_tokens": generated_tokens,
                "continuation": detokenize(
                    generated_tokens
                ),
            }
        )

    return outputs


def run_order_search(
    train_texts: list[str],
    validation_texts: list[str],
    min_frequency: int,
    candidate_orders: list[int],
    discount: float,
    order_search_fraction: float,
    order_search_seed: int,
) -> tuple[int, list[dict], dict]:
    """
    Compare candidate n-gram orders on a fixed training subset.

    Returns:
        selected_order
        candidate results
        search metadata
    """
    search_size = max(
        1,
        int(len(train_texts) * order_search_fraction),
    )

    rng = random.Random(order_search_seed)

    search_indices = rng.sample(
        range(len(train_texts)),
        search_size,
    )

    search_train_texts = [
        train_texts[i]
        for i in search_indices
    ]

    search_vocab, _ = build_vocab(
        search_train_texts,
        min_frequency=min_frequency,
    )

    search_train_sequences = make_sequences(
        search_train_texts,
        search_vocab,
    )

    search_validation_sequences = make_sequences(
        validation_texts,
        search_vocab,
    )

    search_train_tokens = count_prediction_tokens(
        search_train_sequences
    )

    search_validation_tokens = count_prediction_tokens(
        search_validation_sequences
    )

    print(
        f"N-gram order search will use "
        f"{len(search_train_sequences):,} / "
        f"{len(train_texts):,} training reports "
        f"({order_search_fraction:.0%})."
    )

    print(
        f"Order-search vocabulary size: "
        f"{len(search_vocab):,}"
    )

    print(
        f"Order-search training tokens: "
        f"{search_train_tokens:,}"
    )

    print(
        f"Order-search validation tokens: "
        f"{search_validation_tokens:,}"
    )

    print(
        "Candidate n-gram orders: "
        + ", ".join(
            str(order)
            for order in candidate_orders
        )
    )

    print()
    print("Selecting n-gram order on validation set...")
    print()

    candidates = []

    for order in candidate_orders:
        print(f"Training {order}-gram...")

        model = NGramLanguageModel(
            order=order,
            discount=discount,
        )

        start = time.perf_counter()

        model.fit(search_train_sequences)

        training_seconds = (
            time.perf_counter() - start
        )

        validation_ce, validation_ppl, validation_tokens = (
            evaluate_cross_entropy(
                model,
                search_validation_sequences,
            )
        )

        result = {
            "order": order,
            "context_length": order - 1,
            "training_seconds": training_seconds,
            "validation_cross_entropy": validation_ce,
            "validation_perplexity": validation_ppl,
            "validation_tokens": validation_tokens,
        }

        candidates.append(result)

        print(
            f"  validation CE: "
            f"{validation_ce:.4f}"
        )

        print(
            f"  validation PPL: "
            f"{validation_ppl:.2f}"
        )

        print(
            f"  training time: "
            f"{training_seconds:.2f} s"
        )

        print()

        del model
        gc.collect()

    selected = min(
        candidates,
        key=lambda item: item[
            "validation_cross_entropy"
        ],
    )

    selected_order = selected["order"]

    print(
        f"Selected order: {selected_order}-gram "
        f"(context length = {selected_order - 1})"
    )

    metadata = {
        "fraction": order_search_fraction,
        "seed": order_search_seed,
        "training_reports": len(
            search_train_sequences
        ),
        "training_tokens": search_train_tokens,
        "validation_tokens": search_validation_tokens,
        "vocabulary_size": len(search_vocab),
    }

    return selected_order, candidates, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the n-gram language-model baseline."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project configuration file.",
    )

    parser.add_argument(
        "--select-order",
        action="store_true",
        help=(
            "Rerun validation-based order selection using "
            "candidate orders 2, 3, and 5. By default, "
            "the previously selected 3-gram is used."
        ),
    )

    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help=(
            "Fixed n-gram order used when --select-order "
            "is not supplied. Default: 3."
        ),
    )

    args = parser.parse_args()

    if args.order < 2:
        raise ValueError(
            "--order must be at least 2."
        )

    config = load_config(
        project_path(args.config)
    )

    text_column = config["data"][
        "processed_text_column"
    ]

    processed_dir = config["data"][
        "processed_dir"
    ]

    min_frequency = config["tokenizer"][
        "min_frequency"
    ]

    discount = config["ngram"]["discount"]

    order_search_fraction = config["ngram"].get(
        "order_search_fraction",
        0.20,
    )

    order_search_seed = config["ngram"].get(
        "order_search_seed",
        42,
    )

    benchmark_tokens = config["evaluation"][
        "benchmark_tokens"
    ]

    benchmark_repeats = config["evaluation"][
        "benchmark_repeats"
    ]

    generation_config = config["evaluation"][
        "generation"
    ]

    candidate_orders = [2, 3, 5]

    print("Loading data...")

    train_df = load_split(
        processed_dir,
        "train",
    )

    validation_df = load_split(
        processed_dir,
        "validation",
    )

    test_df = load_split(
        processed_dir,
        "test",
    )

    print(train_df.columns.tolist())

    train_texts = (
        train_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    validation_texts = (
        validation_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    test_texts = (
        test_df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("Building vocabulary from full training data...")

    vocab, _ = build_vocab(
        train_texts,
        min_frequency=min_frequency,
    )

    print(
        f"Vocabulary size: {len(vocab):,}"
    )

    print("Preparing full-data token sequences...")

    train_sequences = make_sequences(
        train_texts,
        vocab,
    )

    test_sequences = make_sequences(
        test_texts,
        vocab,
    )

    print(
        f"Training reports: "
        f"{len(train_sequences):,}"
    )

    print(
        f"Training prediction tokens: "
        f"{count_prediction_tokens(train_sequences):,}"
    )

    print(
        f"Test reports: "
        f"{len(test_sequences):,}"
    )

    print(
        f"Test prediction tokens: "
        f"{count_prediction_tokens(test_sequences):,}"
    )

    selection_results = []
    search_metadata = None

    if args.select_order:
        print()

        selected_order, selection_results, search_metadata = (
            run_order_search(
                train_texts=train_texts,
                validation_texts=validation_texts,
                min_frequency=min_frequency,
                candidate_orders=candidate_orders,
                discount=discount,
                order_search_fraction=order_search_fraction,
                order_search_seed=order_search_seed,
            )
        )

    else:
        selected_order = args.order

        print()
        print(
            "Skipping n-gram order search."
        )

        print(
            f"Using fixed order: "
            f"{selected_order}-gram"
        )

    print()
    print(
        f"Training final {selected_order}-gram "
        "on the full training set..."
    )

    model = NGramLanguageModel(
        order=selected_order,
        discount=discount,
    )

    start = time.perf_counter()

    model.fit(train_sequences)

    training_seconds = (
        time.perf_counter() - start
    )

    print(
        f"Full-data training time: "
        f"{training_seconds:.2f} s "
        f"({training_seconds / 60:.2f} min)"
    )

    print()
    print("Evaluating final model on test set...")

    test_cross_entropy, test_perplexity, test_tokens = (
        evaluate_cross_entropy(
            model,
            test_sequences,
        )
    )

    print(
        f"Test cross-entropy: "
        f"{test_cross_entropy:.4f} nats/token"
    )

    print(
        f"Test perplexity: "
        f"{test_perplexity:.2f}"
    )

    print(
        f"Test prediction tokens: "
        f"{test_tokens:,}"
    )

    output_dir = (
        ROOT
        / "results"
        / "ngram"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        ROOT
        / "checkpoints"
        / "ngram.pkl"
    )

    model.save(
        checkpoint_path
    )

    model_size_bytes = (
        checkpoint_path.stat().st_size
    )

    model_size_mb = (
        model_size_bytes / 1024**2
    )

    print(
        f"Serialized model size: "
        f"{model_size_mb:.2f} MB"
    )

    workload = build_benchmark_workload(
        sequences=test_sequences,
        context_length=model.context_length,
        max_tokens=benchmark_tokens,
    )

    benchmark = benchmark_inference(
        model=model,
        workload=workload,
        repeats=benchmark_repeats,
    )

    print(
        f"Inference throughput: "
        f"{benchmark['tokens_per_second']:.1f} "
        "tokens/s"
    )

    generations = generate_samples(
        model=model,
        prompts=generation_config["prompts"],
        vocab=vocab,
        max_new_tokens=generation_config[
            "max_new_tokens"
        ],
        seed=generation_config["seed"],
    )

    print()
    print("Generated continuations:")
    print()

    for sample in generations:
        print(
            f"PROMPT: {sample['prompt']}"
        )

        print(
            f"CONTINUATION: "
            f"{sample['continuation']}"
        )

        print()

    metrics = {
        "model": "ngram",
        "smoothing": "interpolated_kneser_ney",
        "discount": discount,
        "selected_order": model.order,
        "context_length": model.context_length,
        "selection_mode": (
            "validation_search"
            if args.select_order
            else "fixed_order"
        ),
        "candidate_orders": (
            candidate_orders
            if args.select_order
            else None
        ),
        "vocabulary_size": len(vocab),
        "min_frequency": min_frequency,
        "training_reports": len(
            train_sequences
        ),
        "training_tokens": count_prediction_tokens(
            train_sequences
        ),
        "test_cross_entropy_nats": (
            test_cross_entropy
        ),
        "test_perplexity": (
            test_perplexity
        ),
        "test_tokens": test_tokens,
        "training_seconds": training_seconds,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,
        "inference_benchmark": benchmark,
        "order_selection": selection_results,
        "order_search": search_metadata,
        "generation_settings": {
            "max_new_tokens": generation_config[
                "max_new_tokens"
            ],
            "seed": generation_config["seed"],
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": (
                platform.processor()
                or platform.machine()
            ),
        },
    }

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    generations_path = (
        output_dir
        / "generations.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    with generations_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            generations,
            file,
            indent=2,
        )

    print(
        f"Saved model to "
        f"{checkpoint_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved generations to "
        f"{generations_path}"
    )


if __name__ == "__main__":
    main()
