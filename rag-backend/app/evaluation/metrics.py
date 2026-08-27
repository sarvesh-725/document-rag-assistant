"""Deterministic retrieval, citation, and answer evaluation metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def document_recall_at_k(
    retrieved: Sequence[str], expected: Sequence[str], k: int
) -> float:
    target = set(expected)
    return len(target.intersection(retrieved[:k])) / len(target) if target else 0.0


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Compatibility alias for existing offline runner callers."""
    return document_recall_at_k(retrieved, expected, k)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    target = set(expected)
    for index, item in enumerate(retrieved, 1):
        if item in target:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    target = set(expected)
    seen: set[str] = set()
    dcg = 0.0
    for index, item in enumerate(retrieved[:k], 1):
        if item in target and item not in seen:
            seen.add(item)
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, min(k, len(target)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(value.lower()))


def answer_token_f1(answer: str, expected_answer: str) -> float:
    actual = _tokens(answer)
    expected = _tokens(expected_answer)
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    overlap = len(actual.intersection(expected))
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def string_relevancy(answer: str, expected_answer: str) -> float:
    """Compatibility alias; new artifacts use ``answer_token_f1``."""
    return answer_token_f1(answer, expected_answer)


def _page_range(source: dict[str, Any]) -> tuple[int, int] | None:
    start = source.get("page_start", source.get("page"))
    if start is None:
        return None
    end = source.get("page_end", start)
    return int(start), int(end)


def _source_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if str(actual.get("document_id")) != str(expected.get("document_id")):
        return False
    actual_pages = _page_range(actual)
    expected_pages = _page_range(expected)
    if expected_pages is None:
        return (
            not expected.get("section")
            or actual.get("section") == expected.get("section")
        )
    if actual_pages is None:
        return False
    return actual_pages[0] <= expected_pages[1] and expected_pages[0] <= actual_pages[1]


def citation_precision(
    citations: Sequence[dict[str, Any]], expected_sources: Sequence[dict[str, Any]]
) -> float:
    if not citations:
        return 1.0 if not expected_sources else 0.0
    matches = sum(
        any(_source_matches(citation, expected) for expected in expected_sources)
        for citation in citations
    )
    return matches / len(citations)


def citation_recall(
    citations: Sequence[dict[str, Any]], expected_sources: Sequence[dict[str, Any]]
) -> float:
    if not expected_sources:
        return 1.0 if not citations else 0.0
    matches = sum(
        any(_source_matches(citation, expected) for citation in citations)
        for expected in expected_sources
    )
    return matches / len(expected_sources)


def evidence_support_rate(answer: str, contexts: Sequence[str]) -> float:
    answer_tokens = _tokens(answer)
    context_tokens = _tokens("\n".join(contexts))
    return len(answer_tokens.intersection(context_tokens)) / len(answer_tokens) if answer_tokens else 1.0


def no_answer_correct(
    answer: str,
    category: str,
    expected_sources: Sequence[dict[str, Any]],
    evidence_available: bool = False,
) -> float | None:
    if category != "no_answer":
        return None
    return 1.0 if not expected_sources and bool(answer.strip()) and not evidence_available else 0.0


def page_recall(
    retrieved_pages: Sequence[int], expected_pages: Sequence[int]
) -> float:
    """Measure whether expected evidence pages were cited by the response."""
    expected = {int(page) for page in expected_pages}
    retrieved = {int(page) for page in retrieved_pages}
    return len(expected.intersection(retrieved)) / len(expected) if expected else 0.0
