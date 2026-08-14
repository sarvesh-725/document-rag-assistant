"""Privacy-safe structured logging and in-process metrics."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_request_context(*, request_id: str | None = None, user_id: str | None = None) -> None:
    if request_id is not None:
        _request_id.set(request_id)
    if user_id is not None:
        _user_id.set(user_id)


def request_id() -> str:
    return _request_id.get() or str(uuid.uuid4())


class MetricsRegistry:
    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        self.increment(f"{name}_count")
        self.increment(f"{name}_total", value)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)


metrics = MetricsRegistry()
for _name in (
    "upload_success_rate", "ingestion_failure_rate", "ingestion_duration",
    "retrieval_no_hit_rate", "reranker_failure_rate", "LLM_failure_rate",
    "answer_latency_p50", "answer_latency_p95", "queue_depth", "stuck_jobs",
    "embedding_latency", "rerank_latency", "LLM_latency",
):
    metrics.increment(_name, 0)


def structured_log(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit JSON fields without accepting sensitive payload fields."""
    safe = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": fields.pop("request_id", request_id()),
        "user_id": fields.pop("user_id", _user_id.get()),
        **{key: value for key, value in fields.items() if key not in {
            "password", "token", "jwt", "content", "document_text", "document_contents"
        }},
    }
    logger.info(json.dumps(safe, default=str, separators=(",", ":")))


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
