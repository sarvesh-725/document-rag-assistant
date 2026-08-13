"""Durable Redis outbox publisher."""

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import OutboxEvent


async def publish_pending_outbox_events(db: AsyncSession, redis_client=None, *, limit: int = 100) -> int:
    """Publish unpublished rows; acknowledge only after Redis accepts them."""
    if redis_client is None:
        from redis import asyncio as redis_asyncio
        redis_client = redis_asyncio.from_url(get_settings().require_redis_url(), decode_responses=True)
    result = await db.execute(
        select(OutboxEvent).where(OutboxEvent.published_at.is_(None))
        .order_by(OutboxEvent.created_at.asc()).limit(limit)
    )
    published = 0
    for event in result.scalars().all():
        stream = (
            "rag:document-cleanup"
            if event.event_type == "DOCUMENT_CLEANUP_REQUESTED"
            else "rag:document-ingestion"
        )
        try:
            await redis_client.xadd(stream, {
                "event_id": str(event.id), "event_type": event.event_type,
                "aggregate_id": str(event.aggregate_id),
                "payload": json.dumps(event.payload or {}, separators=(",", ":")),
            })
        except Exception:
            event.attempt_count += 1
            await db.commit()
            continue
        event.published_at = datetime.utcnow()
        event.attempt_count += 1
        await db.commit()
        published += 1
    return published
