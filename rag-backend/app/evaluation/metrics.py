"""Pure offline retrieval and generation metric helpers."""

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    target = set(expected)
    return len(target.intersection(retrieved[:k])) / len(target) if target else 0.0


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    target = set(expected)
    for index, item in enumerate(retrieved, 1):
        if item in target:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    target = set(expected)
    dcg = sum((1.0 / math.log2(index + 1)) for index, item in enumerate(retrieved[:k], 1) if item in target)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(k, len(target)) + 1))
    return dcg / ideal if ideal else 0.0


def string_relevancy(answer: str, expected_answer: str) -> float:
    expected = set(expected_answer.lower().split())
    actual = set(answer.lower().split())
    return len(expected.intersection(actual)) / len(expected) if expected else 0.0
