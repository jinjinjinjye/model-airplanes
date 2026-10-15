# model-airplanes

## Model Airplanes: Language Modelling on NASA ASRS Narratives
A Natural Language Processing (NLP) project exploring language models using narratives from NASA's **Aviation Safety Reporting System (ASRS)**. Part I compares the following language models: n-gram, RNN, LSTM, and a Transformer with causal self-attention. Part II compares input-embedding settings within neural architectures. Part III explores the performance of models on the task of incident classification.

## Repository layout

```text
model-airplanes/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/ASRS_merged.csv
│   └── processed/{train,validation,test}.csv
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── neural_data.py
│   ├── models/{ngram,rnn,lstm,transformer}.py
│   ├── evaluate_ngrams.py
│   ├── train_lm.py
│   ├── evaluate_lm.py
│   ├── train_embeddings.py
│   ├── compare_part2.py
│   └── downstream.py
├── checkpoints/
│   ├── ngram.pkl
│   └── embeddings/cbow_100d.model
├── results/
│   ├── ngram/
│   ├── rnn/
│   ├── lstm/
│   ├── transformer/
│   ├── part2/
│   └── part3/
└── notebooks/                 # placeholder
```
PyTorch checkpoints (`*.pt`) and prepared PyTorch embedding matrices are excluded by `.gitignore` and need to be regenerated locally before rerunning neural evaluation, Part II, or Part III. The tracked `results/` directory contains the original metrics, training histories, generated continuations, figures, and downstream error analyses.

## Setup

Run commands below **from the repository root**. Python **3.10+** is needed for the source syntax. A CUDA-capable NVIDIA GPU is recommended for neural training; scripts also support the CPU (with longer training times).

```powershell
git clone https://github.com/jinjinjinjye/model-airplanes.git
cd model-airplanes

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recorded runs used a **NVIDIA GeForce RTX 3060** with CUDA and PyTorch `2.14.1+cu126`. 


## Reproducing the experiments

The committed processed CSVs and results let you inspect the project **without rerunning training**. For a complete rerun, execute the commands in the order below. Retraining will overwrite the corresponding `results/` files; consider preserving the originals first.

### Preprocessing

The processed splits are already included. To regenerate them from the committed raw CSV and check the vocabulary:

```bash
python src/preprocess.py
python src/dataset.py
```

`config.yaml` sets the preprocessing split seed to `27`. **Do not re-split the corpus between experiments**; all parts are designed to use the same original document partitions.

###  Part I: fit and evaluate the four language models

```bash
# Select an n-gram order on the validation split and train the winner.
python src/evaluate_ngrams.py --select-order

# Alternatively, skip order selection and train the selected 3-gram directly:
# python src/evaluate_ngrams.py --order 3

python src/train_lm.py --model rnn
python src/evaluate_lm.py --model rnn

python src/train_lm.py --model lstm
python src/evaluate_lm.py --model lstm

python src/train_lm.py --model transformer
python src/evaluate_lm.py --model transformer
```

Outputs appear in `results/ngram/`, `results/rnn/`, `results/lstm/`, and `results/transformer/`. Neural evaluation reads the corresponding `.pt` checkpoint from `checkpoints/`. Use `--device cpu` or `--device cuda` on neural scripts to override automatic device selection. The neural default inference benchmark scores **200** next-token positions with batch size 1; the n-gram count is taken separately from `config.yaml` (`10,000` in the committed configuration).

### Part II: prepare and compare embedding variants

This step requires the **original Part I LSTM checkpoint** (`checkpoints/lstm.pt`). It reuses the Part I trainable-random LSTM as the baseline.

```bash
# Train ASRS CBOW and download/load GloVe 100D; needs internet for first GloVe use.
python src/train_embeddings.py

# Train two new LSTMs with fully frozen input embeddings.
python src/train_lm.py --model lstm --embedding_variant cbow
python src/evaluate_lm.py --model lstm --embedding_variant cbow

python src/train_lm.py --model lstm --embedding_variant glove
python src/evaluate_lm.py --model lstm --embedding_variant glove

# Collect tables and the overlaid learning curves.
python src/compare_part2.py
```

### Part III: evaluate downstream representations

This step requires `checkpoints/lstm.pt`, `checkpoints/transformer.pt`, the original labeled raw CSV, and the same processed splits used for Part I.

```bash
# Optional: check labels, split integrity, and deduplication without loading checkpoints.
python src/downstream.py --prepare-only

# Extract features and run the common downstream classifier comparison.
python src/downstream.py
```

This saves `results/part3/metrics.json`, `comparison.csv`, confusion matrices, classification plots, and representative prediction errors. Extracted representations are cached in `results/part3/cache/`; use `--no-cache` to recompute them. Results use **frozen original Part I LSTM and Transformer**, not the Part II frozen-CBOW/GloVe variants.

## Reproducibility and evaluation notes

- **Seeds:** `27` for preprocessing; `42` for neural training, generation, CBOW training, downstream classification, and the n-gram order-search subset. Most seeds are script defaults; see `--help` on each script.
- **Evaluation protocol:** model selection uses validation data; reported model quality uses a held-out test set. Neural test cross-entropy excludes padded targets. A common tokenizer and vocabulary are used across the language models.
- **Generation:** the four shared prompts are `the aircraft`, `during approach`, `the captain`, and `after landing`, with up to 40 newly sampled tokens and seed `42`. Neural models use temperature `1.0` and unrestricted top-k/top-p sampling; generation stops early at the end-of-sequence token. The n-gram model uses ancestral sampling from its interpolated distribution.
- **Model sizes and timing:** neural parameter counts and saved checkpoint sizes are tracked separately. Training timings exclude some setup overhead. The n-gram CPU and neural GPU inference benchmark configurations differ and are **not directly comparable**.
- **Downstream leakage safeguards:** the code joins class labels using ASRS accession numbers (`ACN`), preserves the Part I splits, checks for accession-number overlap, removes conflicting labels on identical narratives, and de-duplicates identical normalized text.
- **Caveats:** single recorded neural runs per configuration; differences may reflect training budget or hyperparameters. ASRS jargon, redacted identifiers, truncated long narratives, and class imbalance can affect generalization and generated text.

## Output files

| Path | Contents |
| --- | --- |
| `results/{ngram,rnn,lstm,transformer}/metrics.json` | Test cross-entropy, perplexity, inference metadata |
| `results/{ngram,rnn,lstm,transformer}/generations.json` | Prompt-based continuations |
| `results/{rnn,lstm,transformer}/train_metrics.json` | Hyperparameters, epochs, loss history, time, hardware |
| `results/{rnn,lstm,transformer}/loss_curves.png` | Train/validation loss plots |
| `results/part2/comparison.json` | Embedding ablation summary |
| `results/part2/embedding_coverage.json` | Coverage and embedding-training details |
| `results/part2/loss_curves_comparison.png` | Embedding-variant loss curves |
| `results/part3/metrics.json` | Split/class audit, macro-F1, accuracy, confusion matrices |
| `results/part3/comparison.csv` | Model-level classification results |
| `results/part3/*_representative_errors.csv` | Representative classification errors |

## Data and attribution

The data used comes from NASA's Aviation Safety Reporting System (ASRS) database; specifically, incident reports between January 2022 and June 2025, inclusive. See the [ASRS database](https://asrs.arc.nasa.gov/search/database.html) for source information.

The code uses PyTorch, pandas, scikit-learn, and Gensim; Part II's public pretrained word vectors come from the Gensim `glove-wiki-gigaword-100` dataset. 
