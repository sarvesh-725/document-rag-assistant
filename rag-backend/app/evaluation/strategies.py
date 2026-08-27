"""Thin, named strategy profiles for offline RAG comparison."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RAGStrategy:
    name: str
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fusion: str = "rrf"
    reranker: str | None = "cohere"
    rerank_top_k: int = 10
    parent_expansion: bool = True
    final_context_k: int | None = None
    history_recent_messages: int = 4


PROFILE_NAMES = (
    "dense_only",
    "hybrid",
    "hybrid_parent",
    "hybrid_parent_rerank",
)

_DEFAULT_PROFILES = {
    "dense_only": {
        "dense_top_k": 30,
        "bm25_top_k": 0,
        "reranker": None,
        "parent_expansion": False,
    },
    "hybrid": {
        "dense_top_k": 30,
        "bm25_top_k": 30,
        "reranker": None,
        "parent_expansion": False,
    },
    "hybrid_parent": {
        "dense_top_k": 30,
        "bm25_top_k": 30,
        "reranker": None,
        "parent_expansion": True,
    },
    "hybrid_parent_rerank": {
        "dense_top_k": 30,
        "bm25_top_k": 30,
        "reranker": "cohere",
        "parent_expansion": True,
    },
}


def _load_profiles() -> dict[str, RAGStrategy]:
    values = {name: dict(config) for name, config in _DEFAULT_PROFILES.items()}
    profile_path = Path(__file__).resolve().parents[2] / "evaluation" / "strategies.json"
    if profile_path.exists():
        configured = json.loads(profile_path.read_text(encoding="utf-8"))
        for name in PROFILE_NAMES:
            if isinstance(configured.get(name), dict):
                values[name].update(configured[name])
    return {
        name: RAGStrategy(name=name, **values[name]) for name in PROFILE_NAMES
    }


_PROFILES = _load_profiles()

# Legacy names remain lightweight aliases for existing offline callers.
STRATEGIES = {
    "dense": _PROFILES["dense_only"],
    "dense_rerank": RAGStrategy(
        name="dense_rerank",
        dense_top_k=_PROFILES["dense_only"].dense_top_k,
        bm25_top_k=0,
        reranker="cohere",
        parent_expansion=False,
    ),
    "hybrid": _PROFILES["hybrid"],
    "hybrid_rerank": RAGStrategy(
        name="hybrid_rerank",
        dense_top_k=_PROFILES["hybrid"].dense_top_k,
        bm25_top_k=_PROFILES["hybrid"].bm25_top_k,
        reranker="cohere",
        parent_expansion=False,
    ),
    "hybrid_rerank_parent": _PROFILES["hybrid_parent_rerank"],
}

STRATEGY_ALIASES = _PROFILES


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
