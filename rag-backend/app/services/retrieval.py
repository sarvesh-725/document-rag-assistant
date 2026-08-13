"""Hybrid dense/BM25 retrieval pipeline.

Retrieval is deliberately composed of independent stages.  Candidate scores
from different retrieval systems are never compared directly; Reciprocal Rank
Fusion combines their ranks before reranking.
"""

from __future__ import annotations

import math
import os
import re
import uuid
import logging
from asyncio import gather
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document, DocumentParent, DocumentVersion
from app.services import rag_engine
from app.services.vector_access import owned_vector_filter

logger = logging.getLogger("retrieval")


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else float(raw)


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_k: int = 60
    final_candidate_count: int = 8
    rerank_threshold: Optional[float] = None
    bm25_scan_limit: int = 10000

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            dense_top_k=_positive_int("DENSE_TOP_K", 30),
            bm25_top_k=_positive_int("BM25_TOP_K", 30),
            rrf_k=_positive_int("RRF_K", 60),
            final_candidate_count=_positive_int("FINAL_CANDIDATE_COUNT", 8),
            rerank_threshold=_optional_float("RERANK_THRESHOLD"),
            bm25_scan_limit=_positive_int("BM25_SCAN_LIMIT", 10000),
        )


@dataclass
class RetrievedEvidence:
    document_id: str
    version_id: str
    parent_id: str
    chunk_id: str
    text: str
    display_name: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    parent_text: Optional[str] = None
    element_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_position: Optional[dict] = None
    child_evidence: list["RetrievedEvidence"] | None = None

    @property
    def identity(self) -> str:
        return f"{self.document_id}:{self.version_id}:{self.chunk_id}"


# Internal stage names remain readable while the structured evidence object is
# the public source of truth for context, citations, debugging, and evaluation.
RetrievalCandidate = RetrievedEvidence


def _candidate_from_payload(
    payload: dict[str, Any], *, point_id: Any = None, score: Optional[float] = None
) -> Optional[RetrievalCandidate]:
    required = ("document_id", "version_id", "parent_id")
    if any(payload.get(key) is None for key in required):
        return None
    text = payload.get("child_text") or payload.get("text")
    if not text:
        return None
    chunk_id = str(payload.get("chunk_id", point_id))
    return RetrievalCandidate(
        document_id=str(payload["document_id"]),
        version_id=str(payload["version_id"]),
        parent_id=str(payload["parent_id"]),
        chunk_id=chunk_id,
        text=str(text),
        page=payload.get("page", payload.get("page_start")),
        section=payload.get("section"),
        dense_score=score,
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        element_type=payload.get("element_type"),
        source_position=payload.get("source_position"),
    )


class DenseRetriever:
    """Retrieve top-K dense candidates constrained to selected versions."""

    def __init__(self, client=None, embeddings=None, config: Optional[RetrievalConfig] = None):
        self.client = client or rag_engine.client
        self.embeddings = embeddings
        self.config = config or RetrievalConfig.from_env()
        self._injected_client = client is not None

    async def retrieve(
        self, query: str, user_id: uuid.UUID, version_ids: Sequence[uuid.UUID]
    ) -> list[RetrievalCandidate]:
        if not version_ids or (not self._injected_client and not rag_engine.QDRANT_ENDPOINT):
            return []
        await rag_engine.init_qdrant()
        embedding_service = self.embeddings or rag_engine._get_embeddings()
        vector = embedding_service.embed_query(query)
        vector_filter = owned_vector_filter(user_id, list(version_ids))

        if hasattr(self.client, "query_points"):
            result = await self.client.query_points(
                collection_name=rag_engine.QDRANT_COLLECTION_NAME,
                query=vector,
                query_filter=vector_filter,
                limit=self.config.dense_top_k,
                with_payload=True,
            )
            points = getattr(result, "points", result)
        else:
            points = await self.client.search(
                collection_name=rag_engine.QDRANT_COLLECTION_NAME,
                query_vector=vector,
                query_filter=vector_filter,
                limit=self.config.dense_top_k,
                with_payload=True,
            )

        candidates = []
        for point in points:
            candidate = _candidate_from_payload(
                point.payload or {}, point_id=getattr(point, "id", None), score=point.score
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Small in-process BM25 index over the selected Qdrant text payloads."""

    def __init__(self, documents: Sequence[RetrievalCandidate], *, k1: float = 1.2, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokens = [_tokenize(document.text) for document in self.documents]
        self.avgdl = sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0
        document_frequency: dict[str, int] = {}
        for tokens in self.tokens:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self.document_frequency = document_frequency

    def score(self, query: str) -> list[float]:
        query_terms = _tokenize(query)
        document_count = len(self.documents)
        scores: list[float] = []
        for tokens in self.tokens:
            term_frequency: dict[str, int] = {}
            for token in tokens:
                term_frequency[token] = term_frequency.get(token, 0) + 1
            document_length = len(tokens)
            total = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / (self.avgdl or 1.0)
                )
                total += idf * (frequency * (self.k1 + 1.0)) / denominator
            scores.append(total)
        return scores


class BM25Retriever:
    """Retrieve top-K lexical candidates only from selected versions."""

    def __init__(self, client=None, config: Optional[RetrievalConfig] = None):
        self.client = client or rag_engine.client
        self.config = config or RetrievalConfig.from_env()
        self._injected_client = client is not None

    async def _scroll_candidates(
        self, user_id: uuid.UUID, version_ids: Sequence[uuid.UUID]
    ) -> list[RetrievalCandidate]:
        if not version_ids or (not self._injected_client and not rag_engine.QDRANT_ENDPOINT):
            return []
        await rag_engine.init_qdrant()
        vector_filter = owned_vector_filter(user_id, list(version_ids))
        candidates: list[RetrievalCandidate] = []
        offset = None
        while len(candidates) < self.config.bm25_scan_limit:
            points, next_offset = await self.client.scroll(
                collection_name=rag_engine.QDRANT_COLLECTION_NAME,
                scroll_filter=vector_filter,
                limit=min(1000, self.config.bm25_scan_limit - len(candidates)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                candidate = _candidate_from_payload(
                    point.payload or {}, point_id=getattr(point, "id", None)
                )
                if candidate is not None:
                    candidates.append(candidate)
            if next_offset is None or not points:
                break
            offset = next_offset
        return candidates

    async def retrieve(
        self, query: str, user_id: uuid.UUID, version_ids: Sequence[uuid.UUID]
    ) -> list[RetrievalCandidate]:
        documents = await self._scroll_candidates(user_id, version_ids)
        index = BM25Index(documents)
        scored = [replace(document, bm25_score=score) for document, score in zip(documents, index.score(query))]
        scored.sort(key=lambda candidate: candidate.bm25_score or 0.0, reverse=True)
        return scored[: self.config.bm25_top_k]


class RetrievalFusion:
    """Merge dense and BM25 rankings with Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.rrf_k = rrf_k

    def fuse(
        self,
        dense_candidates: Sequence[RetrievalCandidate],
        bm25_candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        for rank, candidate in enumerate(dense_candidates, start=1):
            current = merged.get(candidate.identity, replace(candidate))
            current.dense_rank = rank
            current.dense_score = candidate.dense_score
            current.fusion_score = (current.fusion_score or 0.0) + 1.0 / (self.rrf_k + rank)
            merged[candidate.identity] = current
        for rank, candidate in enumerate(bm25_candidates, start=1):
            current = merged.get(candidate.identity, replace(candidate))
            current.bm25_rank = rank
            current.bm25_score = candidate.bm25_score
            current.fusion_score = (current.fusion_score or 0.0) + 1.0 / (self.rrf_k + rank)
            merged[candidate.identity] = current
        return sorted(
            merged.values(), key=lambda candidate: candidate.fusion_score or 0.0, reverse=True
        )


class ParentExpander:
    """Group child evidence by parent and attach one PostgreSQL parent record."""

    async def expand(
        self, db: Optional[AsyncSession], candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        grouped: dict[str, list[RetrievalCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(f"{candidate.document_id}:{candidate.version_id}:{candidate.parent_id}", []).append(candidate)

        parents: dict[str, tuple[Any, str]] = {}
        if db is not None:
            parent_ids = []
            version_ids = []
            for candidate in candidates:
                try:
                    parent_ids.append(uuid.UUID(candidate.parent_id))
                    version_ids.append(uuid.UUID(candidate.version_id))
                except ValueError:
                    continue
            if parent_ids:
                result = await db.execute(
                    select(DocumentParent, Document.display_name)
                    .join(DocumentVersion, DocumentVersion.id == DocumentParent.version_id)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(
                        DocumentParent.id.in_(parent_ids),
                        DocumentParent.version_id.in_(version_ids),
                    )
                )
                parents = {
                    str(parent.id): (parent, display_name)
                    for parent, display_name in result.all()
                }

        expanded: list[RetrievalCandidate] = []
        for children in grouped.values():
            children = sorted(
                children,
                key=lambda item: item.fusion_score or 0.0,
                reverse=True,
            )
            representative = children[0]
            parent_record, display_name = parents.get(representative.parent_id, (None, representative.display_name or representative.document_id))
            fusion_scores = [child.fusion_score or 0.0 for child in children]
            dense_scores = [child.dense_score for child in children if child.dense_score is not None]
            bm25_scores = [child.bm25_score for child in children if child.bm25_score is not None]
            combined_fusion = max(fusion_scores) + sum(fusion_scores[1:]) * 0.1
            expanded.append(
                replace(
                    representative,
                    display_name=display_name,
                    text=getattr(parent_record, "text", None) or representative.text,
                    page=getattr(parent_record, "page_start", None) or representative.page,
                    page_start=getattr(parent_record, "page_start", None) or representative.page_start,
                    page_end=getattr(parent_record, "page_end", None) or representative.page_end,
                    section=getattr(parent_record, "section", None) or representative.section,
                    element_type=getattr(parent_record, "element_type", None) or representative.element_type,
                    source_position=getattr(parent_record, "source_position", None) or representative.source_position,
                    dense_score=max(dense_scores) if dense_scores else None,
                    bm25_score=max(bm25_scores) if bm25_scores else None,
                    fusion_score=combined_fusion,
                    child_evidence=children,
                )
            )
        return sorted(expanded, key=lambda item: item.fusion_score or 0.0, reverse=True)


reranker_failure_count = 0


class Reranker:
    """Rerank fused candidates with Cohere when configured, otherwise lexical fallback."""

    def __init__(self, client=None, model: Optional[str] = None):
        self.client = client
        self.model = model or os.getenv("RERANKER_MODEL", "rerank-v3.5")
        if self.client is None and os.getenv("COHERE_API_KEY"):
            try:
                import cohere

                self.client = cohere.AsyncClient(api_key=os.environ["COHERE_API_KEY"])
            except Exception:
                self.client = None

    async def rerank(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        if self.client is None:
            ordered = sorted(
                candidates,
                key=lambda candidate: candidate.fusion_score or 0.0,
                reverse=True,
            )
            return [replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(ordered, 1)]
        try:
            result = await self.client.rerank(
                model=self.model,
                query=query,
                documents=[candidate.text for candidate in candidates],
                top_n=len(candidates),
            )
            reranked = [
                replace(candidates[item.index], rerank_score=float(item.relevance_score))
                for item in result.results
            ]
            ordered = sorted(
                reranked,
                key=lambda candidate: candidate.rerank_score or 0.0,
                reverse=True,
            )
            return [replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(ordered, 1)]
        except Exception:
            global reranker_failure_count
            reranker_failure_count += 1
            logger.exception("Cohere reranking failed; using fusion-ranked candidates")
            ordered = sorted(
                candidates,
                key=lambda candidate: candidate.fusion_score or 0.0,
                reverse=True,
            )
            return [replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(ordered, 1)]


@dataclass(frozen=True)
class ContextBlock:
    candidate: RetrievalCandidate
    text: str


class ContextBuilder:
    """Select final context without an embedding similarity threshold."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig.from_env()

    def build(self, candidates: Sequence[RetrievalCandidate]) -> list[ContextBlock]:
        selected = list(candidates)
        if self.config.rerank_threshold is not None:
            selected = [
                candidate
                for candidate in selected
                if candidate.rerank_score is not None
                and candidate.rerank_score >= self.config.rerank_threshold
            ]

        blocks: list[ContextBlock] = []
        seen_parents: set[str] = set()
        for candidate in selected:
            if candidate.parent_id in seen_parents:
                continue
            seen_parents.add(candidate.parent_id)
            blocks.append(ContextBlock(candidate=candidate, text=candidate.parent_text or candidate.text))
            if len(blocks) >= self.config.final_candidate_count:
                break
        return blocks


@dataclass(frozen=True)
class RetrievalEvaluation:
    query: str
    dense_count: int
    bm25_count: int
    fused_count: int
    reranked_count: int
    selected_count: int
    rerank_threshold: Optional[float]


class RetrievalEvaluationHook(Protocol):
    def on_retrieval_complete(self, evaluation: RetrievalEvaluation) -> None:
        ...


class InMemoryRetrievalEvaluationHook:
    """Test/evaluation hook; production callers can provide telemetry instead."""

    def __init__(self):
        self.evaluations: list[RetrievalEvaluation] = []

    def on_retrieval_complete(self, evaluation: RetrievalEvaluation) -> None:
        self.evaluations.append(evaluation)


@dataclass(frozen=True)
class HybridRetrievalResult:
    dense_candidates: list[RetrievalCandidate]
    bm25_candidates: list[RetrievalCandidate]
    fused_candidates: list[RetrievalCandidate]
    reranked_candidates: list[RetrievalCandidate]
    context: list[ContextBlock]


def has_sufficient_evidence(result: Optional[HybridRetrievalResult]) -> bool:
    """Application grounding rule: no selected context means no answer."""
    return result is not None and bool(result.context)


class HybridRetriever:
    """Orchestrate dense -> BM25 -> fusion -> expansion -> rerank -> context."""

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        bm25: Optional[BM25Retriever] = None,
        fusion: Optional[RetrievalFusion] = None,
        parent_expander: Optional[ParentExpander] = None,
        reranker: Optional[Reranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        evaluation_hook: Optional[RetrievalEvaluationHook] = None,
    ):
        config = RetrievalConfig.from_env()
        self.dense = dense or DenseRetriever(config=config)
        self.bm25 = bm25 or BM25Retriever(config=config)
        self.fusion = fusion or RetrievalFusion(config.rrf_k)
        self.parent_expander = parent_expander or ParentExpander()
        self.reranker = reranker or Reranker()
        self.context_builder = context_builder or ContextBuilder(config)
        self.evaluation_hook = evaluation_hook

    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        version_ids: Sequence[uuid.UUID],
        db: Optional[AsyncSession] = None,
    ) -> HybridRetrievalResult:
        dense_candidates, bm25_candidates = await gather(
            self.dense.retrieve(query, user_id, version_ids),
            self.bm25.retrieve(query, user_id, version_ids),
        )
        fused_candidates = self.fusion.fuse(dense_candidates, bm25_candidates)
        expanded_candidates = await self.parent_expander.expand(db, fused_candidates)
        reranked_candidates = await self.reranker.rerank(query, expanded_candidates)
        context = self.context_builder.build(reranked_candidates)
        if self.evaluation_hook is not None:
            self.evaluation_hook.on_retrieval_complete(
                RetrievalEvaluation(
                    query=query,
                    dense_count=len(dense_candidates),
                    bm25_count=len(bm25_candidates),
                    fused_count=len(fused_candidates),
                    reranked_count=len(reranked_candidates),
                    selected_count=len(context),
                    rerank_threshold=self.context_builder.config.rerank_threshold,
                )
            )
        return HybridRetrievalResult(
            dense_candidates=list(dense_candidates),
            bm25_candidates=list(bm25_candidates),
            fused_candidates=fused_candidates,
            reranked_candidates=reranked_candidates,
            context=context,
        )


def serialize_context(result: HybridRetrievalResult) -> list[dict[str, Any]]:
    """Serialize context while retaining all retrieval provenance fields."""
    return [
        {
            "document_id": block.candidate.document_id,
            "version_id": block.candidate.version_id,
            "display_name": block.candidate.display_name,
            "parent_id": block.candidate.parent_id,
            "chunk_id": block.candidate.chunk_id,
            "text": block.text,
            "page": block.candidate.page,
            "page_start": block.candidate.page_start,
            "page_end": block.candidate.page_end,
            "section": block.candidate.section,
            "element_type": block.candidate.element_type,
            "source_position": block.candidate.source_position,
            "dense_score": block.candidate.dense_score,
            "bm25_score": block.candidate.bm25_score,
            "fusion_score": block.candidate.fusion_score,
            "rerank_score": block.candidate.rerank_score,
            "rerank_rank": block.candidate.rerank_rank,
            "child_evidence": [
                {
                    "document_id": child.document_id,
                    "version_id": child.version_id,
                    "parent_id": child.parent_id,
                    "chunk_id": child.chunk_id,
                    "text": child.text,
                    "page": child.page,
                    "page_start": child.page_start,
                    "page_end": child.page_end,
                    "section": child.section,
                    "element_type": child.element_type,
                    "source_position": child.source_position,
                    "dense_score": child.dense_score,
                    "bm25_score": child.bm25_score,
                    "fusion_score": child.fusion_score,
                }
                for child in (block.candidate.child_evidence or [])
            ],
        }
        for block in result.context
    ]
