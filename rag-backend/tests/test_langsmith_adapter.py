from types import SimpleNamespace
import sys

import pytest

import app.evaluation.langsmith_adapter as adapter
from app.evaluation.langsmith_adapter import (
    _ensure_dataset,
    _evaluator,
    _fingerprint,
    _metric_scores,
)


def test_langsmith_metrics_are_bounded_and_reproducible():
    outputs = {
        "answer": "Revenue grew in 2024",
        "contexts": ["Revenue grew in 2024 according to the annual report."],
        "retrieved_document_ids": ["doc-1"],
        "citations": [{"document_id": "doc-1", "page_start": 1, "page_end": 1}],
    }
    expected = {
        "reference": "Revenue grew in 2024",
        "reference_contexts": ["Revenue grew in 2024 according to the annual report."],
        "expected_document_ids": ["doc-1"],
        "expected_pages": [1],
        "expected_sources": [{"document_id": "doc-1", "page": 1}],
        "category": "direct_lookup",
    }

    scores = _metric_scores(outputs, expected)

    assert scores == _metric_scores(outputs, expected)
    assert all(value is None or 0.0 <= value <= 1.0 for value in scores.values())
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 1.0
    assert _evaluator("evidence_support_rate")(
        SimpleNamespace(outputs=outputs), SimpleNamespace(outputs=expected), langsmith_extra={}
    ) == {"key": "evidence_support_rate", "score": 1.0}


def test_dataset_fingerprint_ignores_strategy_predictions():
    base = {
        "id": "q-001",
        "question": "What grew?",
        "reference": "Revenue grew",
        "reference_contexts": ["Revenue grew"],
    }
    changed_prediction = {**base, "answer": "A different answer", "contexts": ["Other context"]}

    assert _fingerprint([base]) == _fingerprint([changed_prediction])


@pytest.mark.asyncio
async def test_langsmith_adapter_uploads_one_reusable_experiment(monkeypatch):
    rows = [
        {
            "id": "q-001",
            "question": "What grew?",
            "reference": "Revenue grew",
            "reference_contexts": ["Revenue grew"],
            "expected_document_ids": ["doc-1"],
            "expected_pages": [1],
            "expected_sources": [{"document_id": "doc-1", "page": 1}],
            "category": "direct_lookup",
            "answer": "Revenue grew",
            "contexts": ["Revenue grew"],
            "retrieved_document_ids": ["doc-1"],
            "citations": [{"document_id": "doc-1", "page_start": 1, "page_end": 1}],
        }
    ]
    fake_client = FakeLangSmithClient(None, dataset_id="dataset-id")
    fake_experiment = SimpleNamespace(experiment_name="experiment", experiment_url="url")

    class FakeLangSmithModule:
        Client = lambda **_: fake_client

    monkeypatch.setattr(adapter, "get_settings", lambda: SimpleNamespace(
        langsmith_api_key="key",
        langsmith_endpoint="https://example.test",
        langsmith_project="project",
    ))
    monkeypatch.setitem(sys.modules, "langsmith", FakeLangSmithModule)
    monkeypatch.setitem(
        sys.modules,
        "langsmith.evaluation",
        SimpleNamespace(evaluate=lambda *args, **kwargs: fake_experiment),
    )

    result = await adapter.evaluate_with_langsmith(
        rows,
        strategy_name="hybrid",
    )

    assert result["dataset_id"] == "dataset-id"
    assert result["experiment_name"] == "experiment"
    assert result["summary"]["evidence_support_rate"] == 1.0


def test_named_dataset_rejects_changed_cases_without_reset():
    rows = [{"id": "q-001", "question": "What?", "reference": "This"}]
    client = FakeLangSmithClient(
        SimpleNamespace(metadata={"source_fingerprint": "different"})
    )

    with pytest.raises(RuntimeError, match="use --reset"):
        _ensure_dataset(client, "evaluation", rows, reset=False)


class FakeLangSmithClient:
    def __init__(self, dataset, dataset_id="dataset-id"):
        self.dataset = dataset
        self.dataset_id = dataset_id

    def has_dataset(self, *, dataset_name):
        return self.dataset is not None

    def read_dataset(self, *, dataset_name):
        return self.dataset

    def create_dataset(self, *args, **kwargs):
        self.dataset = SimpleNamespace(metadata=kwargs.get("metadata", {}), id=self.dataset_id)
        return self.dataset

    def create_examples(self, **kwargs):
        return None
