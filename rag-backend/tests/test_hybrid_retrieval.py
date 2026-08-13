import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.rag_engine as rag_engine
from app.services.retrieval import (
    BM25Index,
    ContextBuilder,
    DenseRetriever,
    HybridRetriever,
    InMemoryRetrievalEvaluationHook,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalFusion,
    Reranker,
    serialize_context,
)


def candidate(chunk_id, text, *, parent_id="parent", version_id=None):
    return RetrievalCandidate(
        document_id="document-1",
        version_id=str(version_id or uuid.uuid4()),
        parent_id=parent_id,
        chunk_id=str(chunk_id),
        text=text,
    )


def test_rrf_merges_by_stable_child_identity_and_keeps_ranks():
    version_id = uuid.uuid4()
    dense = [
        candidate("1", "revenue", version_id=version_id),
        candidate("2", "costs", version_id=version_id),
    ]
    dense[0].dense_score = 0.12
    dense[1].dense_score = 0.91
    lexical = [
        candidate("2", "costs", version_id=version_id),
        candidate("3", "margin", version_id=version_id),
    ]
    lexical[0].bm25_score = 4.5
    lexical[1].bm25_score = 1.0

    fused = RetrievalFusion(rrf_k=60).fuse(dense, lexical)

    assert [item.chunk_id for item in fused] == ["2", "1", "3"]
    merged = fused[0]
    assert merged.dense_rank == 2
    assert merged.bm25_rank == 1
    assert merged.dense_score == 0.91
    assert merged.bm25_score == 4.5
    assert merged.fusion_score == pytest.approx(1 / 62 + 1 / 61)


def test_bm25_scores_text_without_embedding_score_thresholds():
    documents = [
        candidate("1", "revenue grew strongly in the quarter"),
        candidate("2", "headcount and hiring plans"),
    ]
    scores = BM25Index(documents).score("revenue")
    assert scores[0] > scores[1]
    assert scores[0] > 0


@pytest.mark.asyncio
async def test_dense_retriever_keeps_low_score_top_k_result(monkeypatch):
    version_id = uuid.uuid4()
    point = SimpleNamespace(
        id="chunk-1",
        score=0.01,
        payload={
            "document_id": "document-1",
            "version_id": str(version_id),
            "parent_id": "parent-1",
            "chunk_id": 1,
            "child_text": "revenue context",
        },
    )

    class FakeClient:
        async def query_points(self, **kwargs):
            assert kwargs["limit"] == 1
            return SimpleNamespace(points=[point])

    class FakeEmbeddings:
        def embed_query(self, query):
            return [0.0, 1.0]

    monkeypatch.setattr(rag_engine, "init_qdrant", AsyncMock())
    retriever = DenseRetriever(
        client=FakeClient(),
        embeddings=FakeEmbeddings(),
        config=RetrievalConfig(dense_top_k=1),
    )

    results = await retriever.retrieve("revenue", uuid.uuid4(), [version_id])

    assert len(results) == 1
    assert results[0].dense_score == 0.01


@pytest.mark.asyncio
async def test_hybrid_pipeline_runs_all_stages_and_records_evaluation():
    first = candidate("1", "revenue", parent_id="parent-1")
    first.dense_score = 0.2
    second = candidate("2", "costs", parent_id="parent-2")
    second.bm25_score = 3.0

    class StubDense:
        async def retrieve(self, *args):
            return [first]

    class StubBm25:
        async def retrieve(self, *args):
            return [second]

    class StubExpander:
        async def expand(self, db, candidates):
            return candidates

    hook = InMemoryRetrievalEvaluationHook()
    retriever = HybridRetriever(
        dense=StubDense(),
        bm25=StubBm25(),
        fusion=RetrievalFusion(),
        parent_expander=StubExpander(),
        reranker=Reranker(),
        context_builder=ContextBuilder(
            RetrievalConfig(final_candidate_count=1, rerank_threshold=None)
        ),
        evaluation_hook=hook,
    )

    result = await retriever.retrieve("revenue", uuid.uuid4(), [uuid.uuid4()])

    assert len(result.dense_candidates) == 1
    assert len(result.bm25_candidates) == 1
    assert len(result.fused_candidates) == 2
    assert len(result.context) == 1
    assert hook.evaluations[0].rerank_threshold is None


def test_context_builder_deduplicates_parent_and_serializes_provenance():
    item_a = candidate("1", "child one", parent_id="parent-1")
    item_b = candidate("2", "child two", parent_id="parent-1")
    item_a.rerank_score = 0.2
    item_b.rerank_score = 0.9
    item_a.parent_text = "full parent text"

    result = type("Result", (), {"context": ContextBuilder().build([item_a, item_b])})
    serialized = serialize_context(result)

    assert len(serialized) == 1
    assert serialized[0]["text"] == "full parent text"
    assert {"document_id", "version_id", "parent_id", "chunk_id", "page", "section", "dense_score", "bm25_score", "fusion_score", "rerank_score"} <= serialized[0].keys()
