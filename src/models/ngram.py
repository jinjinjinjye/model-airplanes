from __future__ import annotations

import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


class NGramLanguageModel:
    """
    Interpolated Kneser-Ney n-gram language model.

    The implementation stores only observed n-grams and precomputes the
    continuation counts used by Kneser-Ney smoothing. This avoids the much
    slower probability-time continuation scans performed by NLTK's generic
    language-model implementation.
    """

    def __init__(self, order: int, discount: float = 0.75):
        if order < 2:
            raise ValueError("order must be at least 2")

        if not 0 < discount < 1:
            raise ValueError("discount must be between 0 and 1")

        self.order = order
        self.context_length = order - 1
        self.discount = discount

        # Index k contains counts for k-grams represented as:
        #   context tuple -> Counter(next_token -> count)
        # For k == order these are raw counts. For lower orders these are
        # Kneser-Ney continuation counts N1+(* context token).
        self._counts: list[dict[tuple[str, ...], Counter[str]] | None] = [
            None
        ] * (order + 1)
        self._totals: list[dict[tuple[str, ...], int] | None] = [None] * (
            order + 1
        )
        self._vocab: tuple[str, ...] = ()
        self._fitted = False

    @staticmethod
    def _prepare_sequence(
        sequence: Sequence[str],
        order: int,
    ) -> list[str]:
        """
        Add enough <bos> padding for the requested order.

        Input sequences already contain one <bos>. Repeating it internally
        lets higher-order models make proper predictions at the beginning of
        each report without changing the external data format.
        """
        tokens = list(sequence)

        if not tokens:
            return []

        if tokens[0] == "<bos>":
            return ["<bos>"] * (order - 1) + tokens[1:]

        return ["<bos>"] * (order - 1) + tokens

    @staticmethod
    def _count_raw_ngrams(
        sequences: Sequence[Sequence[str]],
        n: int,
        order: int,
    ) -> Counter[tuple[str, ...]]:
        counts: Counter[tuple[str, ...]] = Counter()

        for sequence in sequences:
            tokens = NGramLanguageModel._prepare_sequence(sequence, order)

            if len(tokens) < n:
                continue

            # Do not learn <bos> as a target token. It may still appear in
            # the context, which is exactly what we want for sentence starts.
            for i in range(n - 1, len(tokens)):
                if tokens[i] == "<bos>":
                    continue

                counts[tuple(tokens[i - n + 1 : i + 1])] += 1

        return counts

    @staticmethod
    def _group_raw_counts(
        raw_counts: Counter[tuple[str, ...]],
    ) -> dict[tuple[str, ...], Counter[str]]:
        grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

        for ngram, count in raw_counts.items():
            grouped[ngram[:-1]][ngram[-1]] = count

        return dict(grouped)

    @staticmethod
    def _continuation_counts_from_raw(
        raw_counts: Counter[tuple[str, ...]],
    ) -> dict[tuple[str, ...], Counter[str]]:
        """
        Convert observed (k+1)-gram *types* into Kneser-Ney k-gram
        continuation counts.

        Every unique (left_context, kgram) contributes exactly one to
        N1+(* kgram), regardless of its raw frequency.
        """
        grouped: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

        for higher_ngram in raw_counts.keys():
            lower_ngram = higher_ngram[1:]
            grouped[lower_ngram[:-1]][lower_ngram[-1]] += 1

        return dict(grouped)

    def fit(
        self,
        sequences: Sequence[Sequence[str]],
        verbose: bool = False,
    ) -> None:
        """
        Fit the interpolated Kneser-Ney model.

        Each input sequence should already contain:
            <bos> ...tokens... <eos>

        Training is linear in the number of observed token positions for each
        n-gram order. No V^n vocabulary enumeration is performed.
        """
        if not sequences:
            raise ValueError("sequences must not be empty")

        self._counts = [None] * (self.order + 1)
        self._totals = [None] * (self.order + 1)

        vocab = {
            token
            for sequence in sequences
            for token in sequence
            if token != "<bos>"
        }
        self._vocab = tuple(sorted(vocab))

        # For Kneser-Ney:
        #   level 1 comes from unique bigram types,
        #   level 2 comes from unique trigram types,
        #   ...,
        #   the highest level uses raw n-gram frequencies.
        #
        # We count one raw order at a time so we do not keep all raw n-gram
        # tables in memory simultaneously.
        for n in range(2, self.order + 1):
            if verbose:
                print(f"    counting observed {n}-grams...", flush=True)
                start = time.perf_counter()

            raw_counts = self._count_raw_ngrams(
                sequences=sequences,
                n=n,
                order=self.order,
            )

            # Unique n-gram types define the continuation counts for the
            # level immediately below them.
            self._counts[n - 1] = self._continuation_counts_from_raw(
                raw_counts
            )

            # At the highest order, ordinary observed frequencies are used.
            if n == self.order:
                self._counts[n] = self._group_raw_counts(raw_counts)

            if verbose:
                elapsed = time.perf_counter() - start
                print(
                    f"      {len(raw_counts):,} unique {n}-grams "
                    f"({elapsed:.2f} s)",
                    flush=True,
                )

            del raw_counts

        # Cache denominators once. Probability lookup then becomes only a few
        # dictionary accesses per interpolation level.
        for k in range(1, self.order + 1):
            level = self._counts[k]

            if level is None:
                raise RuntimeError(f"Missing count table for order {k}")

            self._totals[k] = {
                context: sum(followers.values())
                for context, followers in level.items()
            }

        self._fitted = True

    def _probability_at_order(
        self,
        target: str,
        context: tuple[str, ...],
        k: int,
    ) -> float:
        # Base Kneser-Ney unigram distribution: continuation probability.
        if k == 1:
            level = self._counts[1]
            totals = self._totals[1]

            assert level is not None
            assert totals is not None

            followers = level.get(())
            total = totals.get((), 0)

            if not followers or total == 0:
                return 0.0

            return followers.get(target, 0) / total

        level = self._counts[k]
        totals = self._totals[k]

        assert level is not None
        assert totals is not None

        current_context = context[-(k - 1) :]
        followers = level.get(current_context)
        total = totals.get(current_context, 0)

        # Unseen context: back off immediately.
        if not followers or total == 0:
            return self._probability_at_order(
                target=target,
                context=context,
                k=k - 1,
            )

        count = followers.get(target, 0)
        distinct_followers = len(followers)

        discounted = max(count - self.discount, 0.0) / total
        backoff_weight = (
            self.discount * distinct_followers / total
        )

        lower_probability = self._probability_at_order(
            target=target,
            context=context,
            k=k - 1,
        )

        return discounted + backoff_weight * lower_probability

    def probability(
        self,
        target: str,
        context: Sequence[str],
    ) -> float:
        """Return P(target | preceding context)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring")

        context_tuple = tuple(context[-self.context_length :])
        usable_order = min(self.order, len(context_tuple) + 1)

        return self._probability_at_order(
            target=target,
            context=context_tuple,
            k=usable_order,
        )

    @staticmethod
    def _weighted_choice(
        rng: random.Random,
        items: Sequence[tuple[str, float]],
    ) -> str:
        total = sum(weight for _, weight in items)

        if total <= 0:
            raise RuntimeError("Cannot sample from zero-mass distribution")

        threshold = rng.random() * total
        cumulative = 0.0

        for token, weight in items:
            cumulative += weight
            if cumulative >= threshold:
                return token

        return items[-1][0]

    def _sample_at_order(
        self,
        rng: random.Random,
        context: tuple[str, ...],
        k: int,
    ) -> str:
        """
        Sample exactly from the interpolated Kneser-Ney mixture without
        scoring every vocabulary item.
        """
        if k == 1:
            level = self._counts[1]
            assert level is not None
            followers = level.get((), Counter())

            return self._weighted_choice(
                rng,
                list(followers.items()),
            )

        level = self._counts[k]
        totals = self._totals[k]
        assert level is not None
        assert totals is not None

        current_context = context[-(k - 1) :]
        followers = level.get(current_context)
        total = totals.get(current_context, 0)

        if not followers or total == 0:
            return self._sample_at_order(
                rng=rng,
                context=context,
                k=k - 1,
            )

        discounted_items = [
            (token, max(count - self.discount, 0.0))
            for token, count in followers.items()
            if count > self.discount
        ]
        discounted_mass = sum(
            weight for _, weight in discounted_items
        )
        backoff_mass = self.discount * len(followers)
        total_mass = discounted_mass + backoff_mass

        if total_mass <= 0:
            return self._sample_at_order(
                rng=rng,
                context=context,
                k=k - 1,
            )

        if rng.random() * total_mass < discounted_mass:
            return self._weighted_choice(rng, discounted_items)

        return self._sample_at_order(
            rng=rng,
            context=context,
            k=k - 1,
        )

    def generate(
        self,
        prompt_tokens: Sequence[str],
        max_new_tokens: int,
        seed: int,
        eos_token: str = "<eos>",
    ) -> list[str]:
        """Generate by ancestral sampling."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before generation")

        rng = random.Random(seed)
        context = list(prompt_tokens)

        if not context:
            context = ["<bos>"]

        generated: list[str] = []

        for _ in range(max_new_tokens):
            context_tuple = tuple(context[-self.context_length :])
            usable_order = min(
                self.order,
                len(context_tuple) + 1,
            )

            next_token = self._sample_at_order(
                rng=rng,
                context=context_tuple,
                k=usable_order,
            )

            if next_token == eos_token:
                break

            generated.append(next_token)
            context.append(next_token)

        return generated

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NGramLanguageModel":
        with Path(path).open("rb") as file:
            return pickle.load(file)
