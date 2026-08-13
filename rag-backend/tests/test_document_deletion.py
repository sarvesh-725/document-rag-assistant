"""Phase 14 logical-deletion and cleanup contracts."""

import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.routers.documents as documents
from app.database.enums import DocumentStatus, IngestionStatus, QueryRunStatus, VersionStatus
from app.database.repositories import soft_delete_document
from app.services import document_cleanup
from app.services.document_selection import resolve_selected_documents


def test_delete_route_schedules_durable_cleanup():
    source = inspect.getsource(documents.delete_document)
    assert "soft_delete_document" in source
    assert "DOCUMENT_CLEANUP_REQUESTED" in source
    assert "await db.commit()" in source


def test_cleanup_checks_both_active_reference_types():
    source = inspect.getsource(document_cleanup.cleanup_deleted_document)
    assert "_has_active_query_reference" in source
    assert "_has_active_ingestion" in source
    assert "delete_vectors" in source
    assert "DocumentStatus.DELETED.value" in source


@pytest.mark.asyncio
async def test_query_started_after_delete_cannot_resolve_document():
    user_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        deleted_at=object(),
        status=DocumentStatus.DELETING.value,
        current_version_id=version_id,
    )

    class Db:
        async def get(self, model, row_id, **kwargs):
            return document

    with pytest.raises(ValueError, match="deleted"):
        await resolve_selected_documents(
            Db(),
            authenticated_user_id=user_id,
            selected_document_ids=[str(document_id)],
        )


@pytest.mark.asyncio
async def test_delete_endpoint_is_idempotent_and_schedules_cleanup(monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4())
    document_id = uuid.uuid4()
    soft_delete = AsyncMock(return_value=True)
    create_event = AsyncMock()
    db = AsyncMock()
    monkeypatch.setattr(documents, "soft_delete_document", soft_delete)
    monkeypatch.setattr(documents, "create_outbox_event", create_event)

    response = await documents.delete_document(str(document_id), user, db)

    assert response["status"] == "success"
    soft_delete.assert_awaited_once_with(db, document_id, user.id)
    create_event.assert_awaited_once()
    assert create_event.await_args.args[1] == "DOCUMENT_CLEANUP_REQUESTED"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_endpoint_rejects_non_owned_document(monkeypatch):
    soft_delete = AsyncMock(return_value=False)
    monkeypatch.setattr(documents, "soft_delete_document", soft_delete)
    with pytest.raises(HTTPException) as error:
        await documents.delete_document(str(uuid.uuid4()), SimpleNamespace(id=uuid.uuid4()), AsyncMock())
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_repository_logical_delete_is_idempotent():
    document = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=DocumentStatus.READY.value,
        deleted_at=None,
        updated_at=None,
    )
    db = AsyncMock()
    db.execute.side_effect = [
        _Result(scalar=document),
        _Result(),
        _Result(scalar=document),
        _Result(),
    ]

    assert await soft_delete_document(db, document.id, document.user_id) is True
    deleted_at = document.deleted_at
    assert document.status == DocumentStatus.DELETING.value
    assert await soft_delete_document(db, document.id, document.user_id) is True
    assert document.deleted_at == deleted_at


def test_cleanup_treats_query_and_ingestion_running_states_as_active():
    assert QueryRunStatus.PENDING.value in document_cleanup.ACTIVE_QUERY_STATUSES
    assert QueryRunStatus.RUNNING.value in document_cleanup.ACTIVE_QUERY_STATUSES
    assert IngestionStatus.PENDING.value in document_cleanup.ACTIVE_INGESTION_STATUSES
    assert IngestionStatus.RUNNING.value in document_cleanup.ACTIVE_INGESTION_STATUSES
    assert IngestionStatus.RETRYING.value in document_cleanup.ACTIVE_INGESTION_STATUSES
    assert VersionStatus.DELETED.value == "DELETED"


class _Result:
    def __init__(self, *, scalar=None, rows=None):
        self.scalar = scalar
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.scalar

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _CleanupDb:
    def __init__(self, document, versions):
        self.document = document
        self.versions = versions
        self.execute_count = 0
        self.commits = 0

    async def execute(self, statement):
        self.execute_count += 1
        if self.execute_count == 1:
            return _Result(scalar=self.document)
        if self.execute_count == 2:
            return _Result(rows=self.versions)
        return _Result()

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


@pytest.mark.asyncio
async def test_cleanup_deletes_vectors_parents_source_and_marks_terminal(monkeypatch):
    user_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        status=DocumentStatus.DELETING.value,
        current_version_id=version_id,
        updated_at=None,
    )
    version = SimpleNamespace(
        id=version_id,
        document_id=document_id,
        version_number=1,
        storage_key="documents/source/document/version/original",
        status=VersionStatus.DELETING.value,
        updated_at=None,
    )
    db = _CleanupDb(document, [version])
    vectors = AsyncMock()
    storage = AsyncMock()
    monkeypatch.setattr(document_cleanup, "_has_active_query_reference", AsyncMock(return_value=False))
    monkeypatch.setattr(document_cleanup, "_has_active_ingestion", AsyncMock(return_value=False))

    assert await document_cleanup.cleanup_deleted_document(
        db, document_id, storage=storage, delete_vectors=vectors
    ) is True
    vectors.assert_awaited_once_with(user_id, document_id, version_id)
    storage.delete.assert_awaited_once_with(version.storage_key, user_id)
    assert version.status == VersionStatus.DELETED.value
    assert document.status == DocumentStatus.DELETED.value
    assert db.commits == 1


@pytest.mark.asyncio
async def test_cleanup_leaves_physical_data_when_query_is_running(monkeypatch):
    document = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=DocumentStatus.DELETING.value,
        current_version_id=None,
    )
    db = _CleanupDb(document, [])
    vectors = AsyncMock()
    monkeypatch.setattr(document_cleanup, "_has_active_query_reference", AsyncMock(return_value=True))
    monkeypatch.setattr(document_cleanup, "_has_active_ingestion", AsyncMock())

    assert await document_cleanup.cleanup_deleted_document(
        db, document.id, delete_vectors=vectors
    ) is False
    vectors.assert_not_awaited()
    assert document.status == DocumentStatus.DELETING.value


@pytest.mark.asyncio
async def test_cleanup_leaves_physical_data_when_ingestion_is_running(monkeypatch):
    document = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=DocumentStatus.DELETING.value,
        current_version_id=None,
    )
    db = _CleanupDb(document, [])
    vectors = AsyncMock()
    monkeypatch.setattr(document_cleanup, "_has_active_query_reference", AsyncMock(return_value=False))
    monkeypatch.setattr(document_cleanup, "_has_active_ingestion", AsyncMock(return_value=True))

    assert await document_cleanup.cleanup_deleted_document(
        db, document.id, delete_vectors=vectors
    ) is False
    vectors.assert_not_awaited()
    assert document.status == DocumentStatus.DELETING.value
