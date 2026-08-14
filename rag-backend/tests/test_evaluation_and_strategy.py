import pytest

from app.evaluation.datasets import EvaluationSample
from app.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.evaluation.runners import run_retrieval_evaluation
from app.evaluation.strategies import RAGStrategy, STRATEGIES
from app.services.retrieval import HybridRetriever


def test_retrieval_metrics_have_expected_values():
    retrieved = ["d2", "d1", "d3"]
    expected = ["d1", "d3"]
    assert recall_at_k(retrieved, expected, 2) == 0.5
    assert reciprocal_rank(retrieved, expected) == 0.5
    assert ndcg_at_k(retrieved, expected, 3) > 0


def test_strategy_presets_are_descriptions_not_duplicate_pipelines():
    assert set(STRATEGIES) == {"dense", "dense_rerank", "hybrid", "hybrid_rerank", "hybrid_rerank_parent"}
    assert isinstance(STRATEGIES["hybrid_rerank"], RAGStrategy)
    assert all(hasattr(HybridRetriever, attribute) for attribute in ("retrieve",))


@pytest.mark.asyncio
async def test_evaluation_runner_is_offline_and_calls_injected_retriever():
    sample = EvaluationSample(
        question="What is revenue?",
        expected_answer="Revenue grew",
        expected_document_ids=["d1"],
        expected_pages=[1],
        query_type="factual",
    )
    calls = []

    async def retrieve(received, strategy):
        calls.append((received, strategy))
        return [{"document_id": "d1"}]

    result = await run_retrieval_evaluation([sample], retrieve, STRATEGIES["hybrid"])

    assert result[0]["recall_at_5"] == 1.0
    assert calls[0][0] is sample
