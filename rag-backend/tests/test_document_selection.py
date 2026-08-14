import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.database.enums import DocumentStatus, VersionStatus
from app.database.models import Document, DocumentVersion
from app.services.document_selection import (
    DocumentSelectionError,
    resolve_selected_documents,
)
from app.services.vector_access import owned_vector_filter


class FakeDb:
    def __init__(self, *, documents=None, versions=None):
        self.documents = documents or {}
        self.versions = versions or {}

    async def get(self, model, row_id):
        if model is Document:
            return self.documents.get(row_id)
        if model is DocumentVersion:
            return self.versions.get(row_id)
        raise AssertionError(f"Unexpected model lookup: {model}")


def ready_document(user_id, document_id=None, version_id=None):
    document_id = document_id or uuid.uuid4()
    version_id = version_id or uuid.uuid4()
    document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        deleted_at=None,
        status=DocumentStatus.READY.value,
        current_version_id=version_id,
    )
    version = SimpleNamespace(
        id=version_id,
        document_id=document_id,
        status=VersionStatus.READY.value,
    )
    return document, version


@pytest.mark.asyncio
async def test_selected_one_document_resolves_current_ready_version():
    user_id = uuid.uuid4()
    document, version = ready_document(user_id)
    db = FakeDb(documents={document.id: document}, versions={version.id: version})

    resolved = await resolve_selected_documents(
        db,
        authenticated_user_id=user_id,
        selected_document_ids=[str(document.id)],
    )

    assert [(item.document_id, item.version_id) for item in resolved] == [
        (document.id, version.id)
    ]


@pytest.mark.asyncio
async def test_selected_multiple_documents_preserves_requested_scope():
    user_id = uuid.uuid4()
    document_a, version_a = ready_document(user_id)
    document_b, version_b = ready_document(user_id)
    db = FakeDb(
        documents={document_a.id: document_a, document_b.id: document_b},
        versions={version_a.id: version_a, version_b.id: version_b},
    )

    resolved = await resolve_selected_documents(
        db,
        authenticated_user_id=user_id,
        selected_document_ids=[str(document_a.id), str(document_b.id)],
    )

    assert [item.version_id for item in resolved] == [version_a.id, version_b.id]


@pytest.mark.asyncio
async def test_invalid_document_id_is_rejected():
    with pytest.raises(DocumentSelectionError, match="Invalid document_id"):
        await resolve_selected_documents(
            FakeDb(),
            authenticated_user_id=uuid.uuid4(),
            selected_document_ids=["not-a-uuid"],
        )


@pytest.mark.asyncio
async def test_deleted_document_is_rejected():
    user_id = uuid.uuid4()
    document, version = ready_document(user_id)
    document.deleted_at = datetime.utcnow()
    db = FakeDb(documents={document.id: document}, versions={version.id: version})

    with pytest.raises(DocumentSelectionError, match="deleted"):
        await resolve_selected_documents(
            db,
            authenticated_user_id=user_id,
            selected_document_ids=[str(document.id)],
        )


@pytest.mark.asyncio
async def test_non_owned_document_is_rejected():
    user_id = uuid.uuid4()
    document, version = ready_document(uuid.uuid4())
    db = FakeDb(documents={document.id: document}, versions={version.id: version})

    with pytest.raises(DocumentSelectionError, match="not available"):
        await resolve_selected_documents(
            db,
            authenticated_user_id=user_id,
            selected_document_ids=[str(document.id)],
        )


@pytest.mark.asyncio
async def test_processing_document_is_rejected():
    user_id = uuid.uuid4()
    document, version = ready_document(user_id)
    document.status = DocumentStatus.PROCESSING.value
    db = FakeDb(documents={document.id: document}, versions={version.id: version})

    with pytest.raises(DocumentSelectionError, match="still being processed"):
        await resolve_selected_documents(
            db,
            authenticated_user_id=user_id,
            selected_document_ids=[str(document.id)],
        )


def test_qdrant_filter_requires_authenticated_user_and_resolved_versions():
    user_id = uuid.uuid4()
    version_ids = [uuid.uuid4(), uuid.uuid4()]

    vector_filter = owned_vector_filter(user_id, version_ids)

    owner_condition = next(condition for condition in vector_filter.must if condition.key == "user_id")
    version_condition = next(condition for condition in vector_filter.must if condition.key == "version_id")
    assert owner_condition.match.value == str(user_id)
    assert version_condition.match.any == [str(version_id) for version_id in version_ids]
