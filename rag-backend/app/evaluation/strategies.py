"""Configurable strategy descriptions for offline comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGStrategy:
    name: str
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fusion: str = "rrf"
    reranker: str | None = "cohere"
    rerank_top_k: int = 10
    parent_expansion: bool = True
    history_recent_messages: int = 4


STRATEGIES = {
    "dense": RAGStrategy("dense", bm25_top_k=0, reranker=None),
    "dense_rerank": RAGStrategy("dense_rerank", bm25_top_k=0),
    "hybrid": RAGStrategy("hybrid", reranker=None),
    "hybrid_rerank": RAGStrategy("hybrid_rerank"),
    "hybrid_rerank_parent": RAGStrategy("hybrid_rerank_parent"),
}
