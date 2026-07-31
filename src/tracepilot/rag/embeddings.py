from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable


_TOKEN_RE = re.compile(r"[^\W_]+(?:[.+#/-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return stable, language-agnostic tokens used by retrieval and checks."""

    normalized = text.casefold()
    words = [match.group(0) for match in _TOKEN_RE.finditer(normalized)]
    features: list[str] = []
    for word in words:
        features.append(word)
        compact = re.sub(r"\W+", "", word, flags=re.UNICODE)
        if len(compact) >= 5:
            features.extend(f"#{compact[index:index + 3]}" for index in range(len(compact) - 2))
    return tuple(features)


class DeterministicHashEmbedder:
    """A dependency-free feature-hashing embedder for tests and local fallback.

    It is intentionally deterministic across Python processes. It is not a
    replacement for a semantic embedding provider, but gives the RAG pipeline a
    reproducible vector-retrieval path when no provider is configured.
    """

    def __init__(self, dimensions: int = 256, projections: int = 2) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if projections < 1:
            raise ValueError("projections must be positive")
        self.dimensions = dimensions
        self.projections = projections

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return tuple(vector)

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            weight = 1.0 + math.log(float(count))
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for projection in range(self.projections):
                offset = projection * 4
                bucket = int.from_bytes(digest[offset : offset + 2], "big") % self.dimensions
                sign = 1.0 if digest[offset + 2] & 1 else -1.0
                vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def embed_many(self, texts: Iterable[str]) -> list[tuple[float, ...]]:
        return [self.embed(text) for text in texts]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same dimensions")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values)) / (left_norm * right_norm)
