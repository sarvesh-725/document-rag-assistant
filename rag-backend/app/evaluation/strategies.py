"""Thin, named strategy profiles for offline RAG comparison."""

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
    # Keep the comparison set small: each profile changes one retrieval stage.
    "dense": RAGStrategy(
        "dense", bm25_top_k=0, reranker=None, parent_expansion=False
    ),
    "dense_rerank": RAGStrategy(
        "dense_rerank", bm25_top_k=0, parent_expansion=False
    ),
    "hybrid": RAGStrategy(
        "hybrid", reranker=None, parent_expansion=False
    ),
    "hybrid_rerank": RAGStrategy(
        "hybrid_rerank", parent_expansion=False
    ),
    "hybrid_rerank_parent": RAGStrategy("hybrid_rerank_parent"),
}

# Friendly names used in the project plan without duplicating pipelines.
STRATEGY_ALIASES = {
    "dense_only": RAGStrategy(
        "dense_only", bm25_top_k=0, reranker=None, parent_expansion=False
    ),
    "hybrid_parent": RAGStrategy(
        "hybrid_parent", reranker=None, parent_expansion=True
    ),
    "hybrid_parent_rerank": RAGStrategy("hybrid_parent_rerank"),
}

PROFILE_NAMES = (
    "dense_only",
    "hybrid",
    "hybrid_parent",
    "hybrid_parent_rerank",
)


def get_strategy(name: str) -> RAGStrategy:
    """Resolve one profile name without creating a strategy framework."""
    if name in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[name]
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = sorted({*STRATEGIES, *STRATEGY_ALIASES})
        raise ValueError(
            f"Unknown strategy '{name}'. Choose one of: {', '.join(available)}"
        ) from exc
