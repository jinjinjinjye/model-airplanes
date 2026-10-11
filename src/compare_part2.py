"""Summarize the completed Part II LSTM ablation and plot learning curves.

Run from the project root after evaluating the CBOW and GloVe LSTMs:
    python src/compare_part2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_data import ROOT


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; train and evaluate this variant first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    paths = {
        "random": ROOT / "results" / "lstm",
        "cbow": ROOT / "results" / "part2" / "cbow",
        "glove": ROOT / "results" / "part2" / "glove",
    }
    output_dir = ROOT / "results" / "part2"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "embedding_coverage.json"
    coverage = load_json(coverage_path)["variants"]

    summary = {"variants": {}}
    fig, ax = plt.subplots(figsize=(9, 5))
    print(f"{'Variant':<10} {'Test CE':>12} {'Test PPL':>12} {'Best Val CE':>14} {'LM Train (s)':>15}")
    print("-" * 70)
    for variant, result_dir in paths.items():
        train = load_json(result_dir / "train_metrics.json")
        test = load_json(result_dir / "metrics.json")
        row = {
            "test_cross_entropy_nats": test["test_cross_entropy_nats"],
            "test_perplexity": test["test_perplexity"],
            "best_validation_ce_nats": train["best_validation_ce_nats"],
            "training_seconds": train["training_seconds"],
            "best_epoch": train["best_epoch"],
            "parameters": train["parameters"],
            "trainable_parameters": train.get("trainable_parameters", train["parameters"]),
            "embedding_coverage": coverage.get(variant),
        }
        summary["variants"][variant] = row
        print(f"{variant:<10} {row['test_cross_entropy_nats']:>12.4f} "
              f"{row['test_perplexity']:>12.2f} "
              f"{row['best_validation_ce_nats']:>14.4f} "
              f"{row['training_seconds']:>15.1f}")
        history = train["history"]
        epochs = [point["epoch"] for point in history]
        ax.plot(epochs, [point["train_ce"] for point in history],
                "--", label=f"{variant}: train")
        ax.plot(epochs, [point["validation_ce"] for point in history],
                "-", label=f"{variant}: validation")

    summary["cbow_embedding_training_seconds"] = coverage.get("cbow", {}).get("training_seconds")
    summary["glove_loading_seconds"] = coverage.get("glove", {}).get("loading_seconds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Next-token cross-entropy (nats)")
    ax.set_title("Part II: LSTM input embedding variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves_comparison.png", dpi=160)
    plt.close(fig)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved comparison and overlaid learning curves to {output_dir}")


if __name__ == "__main__":
    main()
