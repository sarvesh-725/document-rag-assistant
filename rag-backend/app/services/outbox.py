"""Taskiq-backed durable outbox dispatch."""

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import OutboxEvent

logger = logging.getLogger("outbox")

EnqueueTask = Callable[..., Awaitable[Any]]


async def enqueue_outbox_event(event: OutboxEvent) -> None:
    """Enqueue one domain event using the existing Taskiq task interface."""
    # Imported lazily to avoid a broker/task-module import cycle. The worker
    # tasks remain the only owners of ingestion and cleanup execution.
    from app.tasks.taskiq_worker import (
        async_process_document_task,
        cleanup_deleted_document_task,
    )

    payload = event.payload or {}
    if event.event_type == "DOCUMENT_INGESTION_REQUESTED":
        required = ("document_id", "version_id", "storage_key", "ingestion_job_id")
        if any(not payload.get(key) for key in required):
            raise ValueError("Ingestion outbox payload is missing a required field")
        if not str(payload["storage_key"]).startswith("documents/"):
            raise ValueError("Ingestion payload must contain a storage key, not a local path")
        await async_process_document_task.kiq(
            document_id=str(payload["document_id"]),
            version_id=str(payload["version_id"]),
            storage_key=str(payload["storage_key"]),
            ingestion_job_id=str(payload["ingestion_job_id"]),
        )
        return

    if event.event_type == "DOCUMENT_CLEANUP_REQUESTED":
        if not payload.get("document_id"):
            raise ValueError("Cleanup outbox payload is missing document_id")
        await cleanup_deleted_document_task.kiq(
            document_id=str(payload["document_id"])
        )
        return

    raise ValueError(f"Unsupported outbox event type: {event.event_type}")


async def publish_pending_outbox_events(
    db: AsyncSession,
    *,
    limit: int = 100,
    enqueue: Optional[Callable[[OutboxEvent], Awaitable[Any]]] = None,
) -> int:
    """Dispatch one batch and acknowledge only after Taskiq accepts it."""
    enqueue = enqueue or enqueue_outbox_event
    result = await db.execute(
        select(OutboxEvent)
        .where(OutboxEvent.published_at.is_(None))
        .order_by(OutboxEvent.created_at.asc())
        .limit(limit)
    )
    published = 0
    for event in result.scalars().all():
        event.attempt_count = (event.attempt_count or 0) + 1
        try:
            await enqueue(event)
        except Exception:
            logger.exception("Outbox dispatch failed for event %s", event.id)
            # Keep published_at NULL so this exact event remains retryable.
            await db.commit()
            continue

        event.published_at = datetime.utcnow()
        await db.commit()
        published += 1
    return published
