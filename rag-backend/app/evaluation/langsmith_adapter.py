"""LangSmith dataset and experiment adapter for offline RAG evaluation.

This module is outside the production request path. The target returns the
actual predictions produced locally by the selected RAG strategy; LangSmith
stores those predictions and applies deterministic code evaluators without
calling Gemini again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.evaluation.metrics import (
    answer_token_f1,
    citation_precision,
    citation_recall,
    document_recall_at_k,
    evidence_support_rate,
    ndcg_at_k,
    no_answer_correct,
    page_recall,
    reciprocal_rank,
)


METRIC_NAMES = (
    "document_recall_at_5",
    "document_recall_at_10",
    "mrr",
    "ndcg_at_10",
    "page_recall",
    "citation_precision",
    "citation_recall",
    "answer_token_f1",
    "evidence_support_rate",
    "no_answer_correct",
)


def _page_numbers(citations: Sequence[dict[str, Any]]) -> list[int]:
    pages: set[int] = set()
    for citation in citations:
        start = citation.get("page_start", citation.get("page"))
        end = citation.get("page_end", start)
        if start is not None:
            pages.update(range(int(start), int(end or start) + 1))
    return sorted(pages)


def _metric_scores(
    outputs: dict[str, Any], example_outputs: dict[str, Any]
) -> dict[str, float | None]:
    retrieved_documents = [
        str(value) for value in outputs.get("retrieved_document_ids", [])
    ]
    expected_documents = [
        str(value) for value in example_outputs.get("expected_document_ids", [])
    ]
    citations = [value for value in outputs.get("citations", []) if isinstance(value, dict)]
    expected_sources = [
        value
        for value in example_outputs.get("expected_sources", [])
        if isinstance(value, dict)
    ]
    expected_pages = [int(value) for value in example_outputs.get("expected_pages", [])]
    answer = str(outputs.get("answer", ""))
    reference = str(example_outputs.get("reference", ""))
    contexts = [str(value) for value in outputs.get("contexts", []) if value]
    category = str(example_outputs.get("category", ""))

    return {
        "document_recall_at_5": (
            None
            if not expected_documents
            else document_recall_at_k(retrieved_documents, expected_documents, 5)
        ),
        "document_recall_at_10": (
            None
            if not expected_documents
            else document_recall_at_k(retrieved_documents, expected_documents, 10)
        ),
        "mrr": None if not expected_documents else reciprocal_rank(retrieved_documents, expected_documents),
        "ndcg_at_10": None if not expected_documents else ndcg_at_k(retrieved_documents, expected_documents, 10),
        "page_recall": None if not expected_pages else page_recall(_page_numbers(citations), expected_pages),
        "citation_precision": citation_precision(citations, expected_sources),
        "citation_recall": citation_recall(citations, expected_sources),
        "answer_token_f1": None if category == "no_answer" else answer_token_f1(answer, reference),
        "evidence_support_rate": (
            None if category == "no_answer" else evidence_support_rate(answer, contexts)
        ),
        "no_answer_correct": no_answer_correct(
            answer,
            category,
            expected_sources,
            bool(outputs.get("evidence_available", False)),
        ),
    }


def _summary(scores: Sequence[dict[str, float | None]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [score[name] for score in scores if isinstance(score.get(name), (int, float))]
        values = [value for value in values if not math.isnan(float(value))]
        if values:
            result[name] = round(sum(values) / len(values), 4)
    return result


def _fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    dataset_rows = [
        {
            field: row.get(field)
            for field in (
                "id",
                "category",
                "question",
                "reference",
                "reference_contexts",
                "expected_document_ids",
                "expected_pages",
                "selected_document_ids",
                "conversation",
            )
        }
        for row in rows
    ]
    payload = json.dumps(dataset_rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_name(rows: Sequence[dict[str, Any]], requested_name: str | None) -> str:
    if requested_name:
        return requested_name
    return f"document-assistant-evaluation-{_fingerprint(rows)[:12]}"


def _example_id(row: dict[str, Any]) -> str:
    value = f"{row.get('id', '')}:{row.get('question', '')}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _ensure_dataset(client: Any, name: str, rows: list[dict[str, Any]], reset: bool) -> Any:
    fingerprint = _fingerprint(rows)
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        metadata = getattr(dataset, "metadata", {}) or {}
        if reset:
            client.delete_dataset(dataset_name=name)
        elif metadata.get("source_fingerprint") == fingerprint:
            return dataset
        else:
            raise RuntimeError(
                f"LangSmith dataset '{name}' contains different cases; "
                "use --reset or choose another --langsmith-dataset name"
            )

    dataset = client.create_dataset(
        name,
        description="Document Assistant evaluation cases",
        metadata={"source_fingerprint": fingerprint, "source": "document-assistant"},
    )
    client.create_examples(
        dataset_name=name,
        inputs=[{"case_id": row["id"], "question": row["question"]} for row in rows],
        outputs=[
            {
                "reference": row.get("reference", ""),
                "reference_contexts": row.get("reference_contexts", []),
                "expected_sources": row.get("expected_sources", []),
                "expected_pages": row.get("expected_pages", []),
                "expected_document_ids": row.get("expected_document_ids", []),
                "selected_document_ids": row.get("selected_document_ids", []),
                "category": row.get("category", "factual"),
            }
            for row in rows
        ],
        metadata=[
            {"case_id": row["id"], "category": row.get("category", "factual")}
            for row in rows
        ],
        ids=[_example_id(row) for row in rows],
    )
    return dataset


def _evaluator(name: str):
    def evaluate_run(run: Any, example: Any = None, **_: Any) -> dict[str, Any]:
        outputs = getattr(run, "outputs", None) or {}
        expected = getattr(example, "outputs", None) or {}
        category = expected.get("category")
        if name in {"document_recall_at_5", "document_recall_at_10", "mrr", "ndcg_at_10"} and not expected.get("expected_document_ids"):
            return {"key": name, "comment": "not_applicable"}
        if name == "page_recall" and not expected.get("expected_pages"):
            return {"key": name, "comment": "not_applicable"}
        if name == "no_answer_correct" and category != "no_answer":
            return {"key": name, "comment": "not_applicable"}
        if name in {"answer_token_f1", "evidence_support_rate"} and category == "no_answer":
            return {"key": name, "comment": "not_applicable"}
        return {"key": name, "score": _metric_scores(outputs, expected)[name]}

    evaluate_run.__name__ = name
    return evaluate_run


async def evaluate_with_langsmith(
    rows: list[dict[str, Any]],
    *,
    strategy_name: str,
    dataset_name: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """Sync a golden dataset and upload one sequential experiment."""
    if not rows:
        return {"cases": [], "summary": {}, "dataset_name": dataset_name}

    settings = get_settings()
    if not settings.langsmith_api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is required for --langsmith; "
            "normal application requests do not require it"
        )

    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Install the LangSmith evaluation dependency before using --langsmith"
        ) from exc

    name = _dataset_name(rows, dataset_name)
    client = Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)
    dataset = _ensure_dataset(client, name, rows, reset)
    case_by_id = {row["id"]: row for row in rows}

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        case = case_by_id.get(inputs.get("case_id"))
        if case is None:
            raise ValueError(f"Unknown LangSmith case: {inputs.get('case_id')}")
        return {
            "answer": case.get("answer", ""),
            "contexts": case.get("contexts", []),
            "retrieved_document_ids": case.get("retrieved_document_ids", []),
            "citations": case.get("citations", []),
            "retrieved_candidates": case.get("retrieved_candidates", []),
            "evidence_available": case.get("evidence_available", False),
            "retrieval_latency_ms": case.get("retrieval_latency_ms"),
            "generation_latency_ms": case.get("generation_latency_ms"),
        }

    def run_experiment() -> Any:
        return evaluate(
            target,
            data=name,
            evaluators=[_evaluator(metric) for metric in METRIC_NAMES],
            client=client,
            experiment_prefix=f"{settings.langsmith_project}-{strategy_name}",
            description="Offline RAG experiment using locally executed strategy outputs",
            metadata={"strategy": strategy_name, "dataset_fingerprint": _fingerprint(rows)},
            max_concurrency=0,
            upload_results=True,
            blocking=True,
        )

    experiment = await asyncio.to_thread(run_experiment)
    scores = [
        _metric_scores(
            row,
            {
                "reference": row.get("reference", ""),
                "reference_contexts": row.get("reference_contexts", []),
                "expected_sources": row.get("expected_sources", []),
                "expected_pages": row.get("expected_pages", []),
                "expected_document_ids": row.get("expected_document_ids", []),
                "category": row.get("category", "factual"),
            },
        )
        for row in rows
    ]
    return {
        "cases": scores,
        "summary": _summary(scores),
        "dataset_name": name,
        "dataset_id": str(getattr(dataset, "id", "")),
        "experiment_name": getattr(experiment, "experiment_name", None),
        "experiment_url": getattr(experiment, "experiment_url", None),
    }
