"""Evaluate a trained neural LM, generate samples, and benchmark next-token inference.

Example: python src/evaluate_lm.py --model lstm --benchmark_tokens 200
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import tokenize
from neural_data import (
    ROOT,
    NextTokenWindows,
    collate_next_tokens,
    create_model,
    encode_reports,
    load_config,
    load_texts,
    parameter_count,
)


def evaluate_test(model, loader, pad_id, device):
    model.eval()
    negative_log_likelihood = 0.0
    tokens = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            negative_log_likelihood += loss.item()
            tokens += (y != pad_id).sum().item()
    ce = negative_log_likelihood / tokens
    return ce, math.exp(ce), tokens


def synced(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def next_token_workload(reports, size):
    """First `size` target positions in the test split, as in n-gram code."""
    workload = []
    for report in reports:
        for i in range(1, len(report)):
            workload.append((report[:i], report[i]))
            if len(workload) >= size:
                return workload
    return workload


def benchmark_inference(model, workload, context_length, device, repeats):
    """Batch-1 next-token prediction; tokenization and disk I/O excluded."""
    model.eval()
    contexts = [
        (torch.tensor(prefix[-context_length:], device=device, dtype=torch.long)[None, :], target)
        for prefix, target in workload
    ]
    with torch.inference_mode():
        for x, target in contexts[: min(10, len(contexts))]:
            nn.functional.log_softmax(model(x)[:, -1, :], dim=-1)[0, target]
        synced(device)
        times = []
        for _ in range(repeats):
            synced(device)
            start = time.perf_counter()
            for x, target in contexts:
                nn.functional.log_softmax(model(x)[:, -1, :], dim=-1)[0, target]
            synced(device)
            times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {
        "workload_tokens": len(contexts),
        "batch_size": 1,
        "repeats": repeats,
        "times_seconds": times,
        "median_seconds": median,
        "tokens_per_second": len(contexts) / median,
        "timing_includes": "scoring known next-token probability, excluding tokenization and disk I/O; batch=1",
    }


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    return re.sub(r"([(\[{])\s+", r"\1", text)


def generate_samples(model, prompts, vocab, cfg, context_length, device):
    word_to_id = {word: i for i, word in enumerate(vocab)}
    bos_id, eos_id = word_to_id["<bos>"], word_to_id["<eos>"]
    pad_id = word_to_id["<pad>"]
    unk_id = word_to_id["<unk>"]
    temperature = float(cfg.get("temperature", 1.0))
    top_k = int(cfg.get("top_k") or 0)
    top_p = float(cfg.get("top_p", 1.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 40))
    seed = int(cfg.get("seed", 42))
    if temperature <= 0 or not (0 < top_p <= 1):
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    model.eval()
    results = []
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(prompts):
            generator = torch.Generator(device="cpu").manual_seed(seed + prompt_index)
            prefix = [bos_id, *[word_to_id.get(token, unk_id) for token in tokenize(prompt)]]
            continuation = []
            for _ in range(max_new_tokens):
                x = torch.tensor(prefix[-context_length:], dtype=torch.long, device=device)[None, :]
                logits = model(x)[0, -1].float().cpu() / temperature
                logits[pad_id] = float("-inf")
                logits[bos_id] = float("-inf")
                if top_k > 0:
                    cutoff = torch.topk(logits, min(top_k, len(logits))).values[-1]
                    logits[logits < cutoff] = float("-inf")
                if top_p < 1:
                    values, indices = torch.sort(logits, descending=True)
                    probabilities = torch.softmax(values, dim=-1)
                    # Keep the token that first crosses the cumulative threshold.
                    remove = probabilities.cumsum(dim=-1) - probabilities > top_p
                    logits[indices[remove]] = float("-inf")
                probabilities = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, 1, generator=generator).item()
                if next_id == eos_id:
                    break
                prefix.append(next_id)
                continuation.append(vocab[next_id])
            results.append({
                "prompt": prompt,
                "generated_tokens": continuation,
                "continuation": detokenize(continuation),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("rnn", "lstm", "transformer"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--embedding_variant", choices=("random", "cbow", "glove"),
                        default="random")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--benchmark_tokens", type=int, default=200,
                        help="Same first N test targets as the n-gram benchmark; 200 keeps autoregressive scoring manageable")
    parser.add_argument("--benchmark_repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)

    cfg = load_config(args.config)
    if args.embedding_variant != "random" and args.model != "lstm":
        parser.error("Part II embedding variants are implemented for --model lstm only")
    checkpoint_path = ROOT / "checkpoints" / (
        f"{args.model}_{args.embedding_variant}.pt"
        if args.embedding_variant != "random" else f"{args.model}.pt"
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != args.model:
        raise ValueError("Checkpoint architecture does not match --model")
    if ckpt.get("embedding_variant", "random") != args.embedding_variant:
        raise ValueError("Checkpoint embedding variant does not match --embedding_variant")
    vocab = ckpt["vocab"]
    pad_id = ckpt["pad_id"]
    context_length = ckpt["context_length"]
    model = create_model(args.model, len(vocab), pad_id, ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    test_texts = load_texts(cfg, "test")
    test_reports = encode_reports(test_texts, vocab)
    collate = partial(collate_next_tokens, pad_id=pad_id)
    test_loader = DataLoader(
        NextTokenWindows(test_reports, context_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    print(f"Testing {args.model} on {len(test_reports):,} reports using {device}...")
    ce, perplexity, n_tokens = evaluate_test(model, test_loader, pad_id, device)
    print(f"Test cross-entropy: {ce:.4f} nats/token | perplexity: {perplexity:.2f} | tokens: {n_tokens:,}")

    workload = next_token_workload(test_reports, args.benchmark_tokens)
    benchmark = benchmark_inference(model, workload, context_length, device, args.benchmark_repeats)
    print(f"Batch-1 inference: {benchmark['tokens_per_second']:.1f} tokens/s")

    generation_cfg = cfg["evaluation"]["generation"]
    generations = generate_samples(
        model, generation_cfg["prompts"], vocab, generation_cfg, context_length, device
    )
    for sample in generations:
        print(f"\nPROMPT: {sample['prompt']}\nCONTINUATION: {sample['continuation']}")

    output_dir = (ROOT / "results" / "part2" / args.embedding_variant
                  if args.embedding_variant != "random" else ROOT / "results" / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": args.model,
        "embedding_variant": args.embedding_variant,
        "embedding_frozen": ckpt.get("embedding_frozen", False),
        "context_length": context_length,
        "vocabulary_size": len(vocab),
        "test_cross_entropy_nats": ce,
        "test_perplexity": perplexity,
        "test_tokens": n_tokens,
        "parameters": parameter_count(model),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "inference_benchmark": benchmark,
        "generation_settings": generation_cfg,
        "device": str(device),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "generations.json").open("w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2)
    print(f"\nSaved metrics and generations to {output_dir}")


if __name__ == "__main__":
    main()
