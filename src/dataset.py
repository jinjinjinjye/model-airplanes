# the same tokenizer is used for all four language models
# shared-word level vocabulary

import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\w+(?:[-']\w+)*|[^\w\s]",
        text.lower(),
    )


def build_vocab(texts, min_frequency=2):
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    vocab = SPECIAL_TOKENS.copy()

    vocab.extend(
        token
        for token, count in counts.items()
        if count >= min_frequency
    )

    token_to_id = {
        token: index
        for index, token in enumerate(vocab)
    }

    return vocab, token_to_id

if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv("data/processed/train.csv")

    texts = train_df["text"].dropna()

    # Count all tokens in the training corpus.
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    total_tokens = sum(counts.values())
    unique_tokens = len(counts)

    print(f"Documents:     {len(texts):,}")
    print(f"Total tokens:  {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print()

    for min_freq in [1, 2, 3, 5, 10, 20]:
        kept_types = sum(
            count >= min_freq
            for count in counts.values()
        )

        kept_tokens = sum(
            count
            for count in counts.values()
            if count >= min_freq
        )

        coverage = kept_tokens / total_tokens

        print(
            f"min_freq={min_freq:>2}: "
            f"{kept_types:>7,} vocabulary items | "
            f"{coverage:>7.2%} token coverage"
        )

    vocab, token_to_id = build_vocab(
        texts,
        min_frequency=2,
    )

    print()
    print(f"Vocabulary size with min_freq=2: {len(vocab):,}")

    # min_freq = 2 gives 99.67% token coverage on train.csv