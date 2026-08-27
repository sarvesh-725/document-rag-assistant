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
    ParentExpander,
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


@pytest.mark.asyncio
async def test_hybrid_reranks_child_candidates_before_parent_expansion():
    first = candidate("1", "child one", parent_id="parent-1")
    second = candidate("2", "child two", parent_id="parent-2")
    first.fusion_score = 0.9
    second.fusion_score = 0.8
    observed = {}

    class StubDense:
        async def retrieve(self, *args):
            return [first, second]

    class StubBm25:
        async def retrieve(self, *args):
            return []

    class StubReranker:
        async def rerank(self, query, candidates):
            observed["reranker_input"] = list(candidates)
            second.rerank_score = 0.9
            first.rerank_score = 0.1
            return [second, first]

    class StubExpander:
        async def expand(self, db, candidates):
            observed["expander_input"] = list(candidates)
            return candidates

    retriever = HybridRetriever(
        dense=StubDense(),
        bm25=StubBm25(),
        reranker=StubReranker(),
        parent_expander=StubExpander(),
        context_builder=ContextBuilder(RetrievalConfig(final_candidate_count=2)),
    )

    result = await retriever.retrieve("query", uuid.uuid4(), [uuid.uuid4()])

    assert [item.chunk_id for item in observed["reranker_input"]] == ["1", "2"]
    assert [item.chunk_id for item in observed["expander_input"]] == ["2", "1"]
    assert result.context[0].candidate.chunk_id == "2"


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


@pytest.mark.asyncio
async def test_parent_expander_groups_children_and_retains_child_evidence():
    version_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    children = [
        candidate(str(index), f"child {index}", parent_id=str(parent_id), version_id=version_id)
        for index in range(3)
    ]
    for index, child in enumerate(children):
        child.fusion_score = 1.0 / (index + 1)
    other_parent = candidate("4", "other", parent_id=str(uuid.uuid4()), version_id=version_id)
    other_parent.fusion_score = 0.2

    class Result:
        def all(self):
            return [
                (
                    SimpleNamespace(
                        id=parent_id,
                        text="P1 full text",
                        page_start=3,
                        page_end=4,
                        section="Revenue",
                        element_type="NarrativeText",
                        source_position={"elements": [1]},
                    ),
                    "report.pdf",
                )
            ]

    db = SimpleNamespace(execute=AsyncMock(return_value=Result()))
    expanded = await ParentExpander().expand(db, children + [other_parent])

    assert len(expanded) == 2
    p1 = next(item for item in expanded if item.parent_id == str(parent_id))
    assert p1.text == "P1 full text"
    assert p1.display_name == "report.pdf"
    assert len(p1.child_evidence or []) == 3
    assert p1.page_start == 3
    assert p1.section == "Revenue"


@pytest.mark.asyncio
async def test_cohere_failure_falls_back_to_fusion_order_and_increments_metric(monkeypatch):
    import app.services.retrieval as retrieval

    class BrokenCohere:
        async def rerank(self, **kwargs):
            raise RuntimeError("cohere unavailable")

    first = candidate("1", "first")
    second = candidate("2", "second")
    first.fusion_score = 0.2
    second.fusion_score = 0.9
    before = retrieval.reranker_failure_count

    results = await Reranker(client=BrokenCohere()).rerank("query", [first, second])

    assert [item.chunk_id for item in results] == ["2", "1"]
    assert retrieval.reranker_failure_count == before + 1
