"""LangSmith dataset and experiment adapter for offline RAG evaluation.

This module is intentionally outside the production request path. The target
used by LangSmith returns already-computed cases, so running an experiment does
not call Gemini again and is safe to repeat against the same dataset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.evaluation.metrics import string_relevancy


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(value.lower()))


def _coverage(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    return len(source_tokens.intersection(target_tokens)) / len(source_tokens) if source_tokens else 0.0


def _reference_texts(example_outputs: dict[str, Any]) -> list[str]:
    references = [str(value) for value in example_outputs.get("reference_contexts", []) if value]
    return references or [str(example_outputs.get("reference", ""))]


def _metric_scores(outputs: dict[str, Any], example_outputs: dict[str, Any]) -> dict[str, float]:
    """Calculate bounded, reproducible evaluators without another model call."""
    contexts = [str(value) for value in outputs.get("contexts", []) if value]
    reference_texts = _reference_texts(example_outputs)
    reference = str(example_outputs.get("reference", ""))
    combined_context = "\n".join(contexts)
    combined_reference = "\n".join(reference_texts)

    if contexts and combined_reference:
        context_precision = _coverage(combined_context, combined_reference)
        context_recall = _coverage(combined_reference, combined_context)
    else:
        context_precision = 0.0
        context_recall = 0.0

    answer = str(outputs.get("answer", ""))
    faithfulness = _coverage(answer, combined_context) if answer and combined_context else 0.0
    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(string_relevancy(answer, reference), 4),
    }


def _summary(scores: Sequence[dict[str, float]]) -> dict[str, float]:
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
    inputs = [
        {"case_id": row["id"], "question": row["question"]}
        for row in rows
    ]
    outputs = [
        {
            "reference": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
            "expected_document_ids": row.get("expected_document_ids", []),
        }
        for row in rows
    ]
    metadata = [
        {"case_id": row["id"], "category": row.get("category", "factual")}
        for row in rows
    ]
    client.create_examples(
        dataset_name=name,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        ids=[_example_id(row) for row in rows],
    )
    return dataset


def _evaluator(name: str):
    def evaluate_run(run: Any, example: Any = None, **_: Any) -> dict[str, Any]:
        outputs = getattr(run, "outputs", None) or {}
        expected = getattr(example, "outputs", None) or {}
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
    """Sync a dataset and upload one sequential LangSmith experiment."""
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
        }

    def run_experiment() -> Any:
        return evaluate(
            target,
            data=name,
            evaluators=[_evaluator(metric) for metric in METRIC_NAMES],
            client=client,
            experiment_prefix=f"{settings.langsmith_project}-{strategy_name}",
            description="Offline RAG evaluation; target cases were generated locally",
            metadata={"strategy": strategy_name, "dataset_fingerprint": _fingerprint(rows)},
            max_concurrency=0,
            upload_results=True,
            blocking=True,
        )

    experiment = await asyncio.to_thread(run_experiment)
    scores = [_metric_scores(
        {"answer": row.get("answer", ""), "contexts": row.get("contexts", [])},
        {"reference": row.get("reference", ""), "reference_contexts": row.get("reference_contexts", [])},
    ) for row in rows]
    return {
        "cases": scores,
        "summary": _summary(scores),
        "dataset_name": name,
        "dataset_id": str(getattr(dataset, "id", "")),
        "experiment_name": getattr(experiment, "experiment_name", None),
        "experiment_url": getattr(experiment, "experiment_url", None),
    }
