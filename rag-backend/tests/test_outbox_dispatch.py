import asyncio
import inspect
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.outbox_publisher as publisher
import app.services.outbox as outbox
import app.tasks.taskiq_worker as worker


class Result:
    def __init__(self, events):
        self.events = events

    def scalars(self):
        return self

    def all(self):
        return self.events


class FakeDb:
    def __init__(self, events):
        self.events = events
        self.commits = 0

    async def execute(self, statement):
        return Result(self.events)

    async def commit(self):
        self.commits += 1


def event(event_type, payload):
    return SimpleNamespace(
        id=uuid.uuid4(),
        event_type=event_type,
        payload=payload,
        published_at=None,
        attempt_count=0,
        created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_ingestion_event_dispatches_exact_task_payload_and_acknowledges_after_enqueue():
    item = event(
        "DOCUMENT_INGESTION_REQUESTED",
        {
            "document_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "storage_key": "documents/user/document/version/original",
            "ingestion_job_id": str(uuid.uuid4()),
            "filename": "must-not-be-forwarded",
        },
    )
    enqueue = AsyncMock()

    assert await outbox.publish_pending_outbox_events(
        FakeDb([item]), enqueue=enqueue
    ) == 1

    assert item.published_at is not None
    assert item.attempt_count == 1
    enqueue.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_enqueue_failure_keeps_event_retryable_and_increments_attempt():
    item = event(
        "DOCUMENT_INGESTION_REQUESTED",
        {
            "document_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
            "storage_key": "documents/user/document/version/original",
            "ingestion_job_id": str(uuid.uuid4()),
        },
    )
    enqueue = AsyncMock(side_effect=RuntimeError("broker unavailable"))
    db = FakeDb([item])

    assert await outbox.publish_pending_outbox_events(db, enqueue=enqueue) == 0
    assert item.published_at is None
    assert item.attempt_count == 1
    assert db.commits == 1


@pytest.mark.asyncio
async def test_failed_event_is_retried_on_later_iteration():
    item = event(
        "DOCUMENT_CLEANUP_REQUESTED",
        {"document_id": str(uuid.uuid4()), "user_id": str(uuid.uuid4())},
    )
    enqueue = AsyncMock(side_effect=[RuntimeError("temporary"), None])
    db = FakeDb([item])

    assert await outbox.publish_pending_outbox_events(db, enqueue=enqueue) == 0
    assert item.published_at is None
    assert await outbox.publish_pending_outbox_events(db, enqueue=enqueue) == 1
    assert item.published_at is not None
    assert item.attempt_count == 2


@pytest.mark.asyncio
async def test_cleanup_event_dispatches_existing_cleanup_payload():
    document_id = str(uuid.uuid4())
    item = event("DOCUMENT_CLEANUP_REQUESTED", {"document_id": document_id})
    enqueue = AsyncMock()

    await outbox.publish_pending_outbox_events(FakeDb([item]), enqueue=enqueue)

    enqueue.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_taskiq_enqueue_uses_existing_tasks_and_required_arguments(monkeypatch):
    ingestion_kiq = AsyncMock()
    cleanup_kiq = AsyncMock()
    monkeypatch.setattr(
        worker,
        "async_process_document_task",
        SimpleNamespace(kiq=ingestion_kiq),
    )
    monkeypatch.setattr(
        worker,
        "cleanup_deleted_document_task",
        SimpleNamespace(kiq=cleanup_kiq),
    )
    ingestion = event(
        "DOCUMENT_INGESTION_REQUESTED",
        {
            "document_id": "d",
            "version_id": "v",
            "storage_key": "documents/u/d/v/original",
            "ingestion_job_id": "j",
        },
    )
    cleanup = event("DOCUMENT_CLEANUP_REQUESTED", {"document_id": "d", "user_id": "u"})

    await outbox.enqueue_outbox_event(ingestion)
    await outbox.enqueue_outbox_event(cleanup)

    ingestion_kiq.assert_awaited_once_with(
        document_id="d",
        version_id="v",
        storage_key="documents/u/d/v/original",
        ingestion_job_id="j",
    )
    cleanup_kiq.assert_awaited_once_with(document_id="d")


@pytest.mark.asyncio
async def test_local_paths_are_rejected_before_task_enqueue():
    item = event(
        "DOCUMENT_INGESTION_REQUESTED",
        {
            "document_id": "d",
            "version_id": "v",
            "storage_key": r"C:\uploads\source.pdf",
            "ingestion_job_id": "j",
        },
    )
    with pytest.raises(ValueError, match="storage key"):
        await outbox.enqueue_outbox_event(item)


def test_publisher_is_standalone_and_not_a_taskiq_task():
    source = inspect.getsource(publisher.run_outbox_publisher)
    assert "asyncio" in source
    assert "AsyncSessionLocal" in source
    assert "publish_pending_outbox_events" in source
    assert "@broker.task" not in source


@pytest.mark.asyncio
async def test_publisher_stops_and_closes_resources(monkeypatch):
    stop_event = asyncio.Event()
    startup = AsyncMock()
    shutdown = AsyncMock()
    dispose = AsyncMock()
    monkeypatch.setattr(publisher.broker, "startup", startup)
    monkeypatch.setattr(publisher.broker, "shutdown", shutdown)
    monkeypatch.setattr(publisher, "engine", SimpleNamespace(dispose=dispose))
    monkeypatch.setattr(publisher, "AsyncSessionLocal", lambda: FakeSessionContext())

    async def publish_once(*args, **kwargs):
        stop_event.set()
        return 0

    monkeypatch.setattr(publisher, "publish_pending_outbox_events", publish_once)
    await publisher.run_outbox_publisher(stop_event=stop_event, polling_interval=0.01)

    startup.assert_awaited_once()
    shutdown.assert_awaited_once()
    dispose.assert_awaited_once()


class FakeSessionContext:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *args):
        return False
