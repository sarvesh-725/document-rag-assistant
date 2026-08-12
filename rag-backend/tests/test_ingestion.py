import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.routers.documents as documents
from app.services.outbox import publish_pending_outbox_events


def user():
    return SimpleNamespace(id=uuid.uuid4())


@pytest.mark.asyncio
async def test_upload_creates_durable_ids_and_outbox_payload(monkeypatch):
    current_user, db = user(), AsyncMock()
    events = []
    monkeypatch.setattr(documents, "create_document", AsyncMock())
    monkeypatch.setattr(documents, "create_document_version", AsyncMock())
    monkeypatch.setattr(documents, "create_ingestion_job", AsyncMock())
    monkeypatch.setattr(documents, "create_outbox_event", AsyncMock(side_effect=lambda *args: events.append(args)))
    monkeypatch.setattr(documents.storage_service, "save", AsyncMock())
    file = SimpleNamespace(filename="a.txt", content_type="text/plain", read=AsyncMock(return_value=b"hello"))

    response = await documents.upload_document(file=file, current_user=current_user, db=db)

    assert response["status"] == "PENDING"
    payload = events[0][2]
    assert {"ingestion_job_id", "document_id", "version_id", "storage_key"} <= payload.keys()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_outbox_acknowledged_only_after_successful_publish():
    event = SimpleNamespace(id=uuid.uuid4(), event_type="DOCUMENT_INGESTION_REQUESTED", aggregate_id=uuid.uuid4(), payload={}, published_at=None, attempt_count=0)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [event])))
    redis = AsyncMock()
    assert await publish_pending_outbox_events(db, redis) == 1
    assert event.published_at is not None


@pytest.mark.asyncio
async def test_failed_outbox_publish_remains_unpublished():
    event = SimpleNamespace(id=uuid.uuid4(), event_type="DOCUMENT_INGESTION_REQUESTED", aggregate_id=uuid.uuid4(), payload={}, published_at=None, attempt_count=0)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [event])))
    redis = AsyncMock()
    redis.xadd.side_effect = RuntimeError("redis unavailable")
    assert await publish_pending_outbox_events(db, redis) == 0
    assert event.published_at is None
    assert event.attempt_count == 1
