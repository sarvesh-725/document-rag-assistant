import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.routers.documents as documents


def user():
    return SimpleNamespace(id=uuid.uuid4())


@pytest.mark.asyncio
async def test_upload_is_global_and_does_not_require_session_ownership(monkeypatch):
    current_user = user()
    db = AsyncMock()
    events = []
    monkeypatch.setattr(
        documents,
        "create_document",
        AsyncMock(return_value=SimpleNamespace(display_name="report.pdf")),
    )
    monkeypatch.setattr(documents, "create_document_version", AsyncMock())
    monkeypatch.setattr(documents, "create_ingestion_job", AsyncMock())
    monkeypatch.setattr(
        documents,
        "create_outbox_event",
        AsyncMock(side_effect=lambda *args: events.append(args)),
    )
    monkeypatch.setattr(documents.storage_service, "save", AsyncMock())
    file = SimpleNamespace(
        filename="../report.pdf",
        content_type="application/pdf",
        read=AsyncMock(return_value=b"%PDF-1.7 source"),
    )

    response = await documents.upload_document(
        file=file,
        session_id=str(uuid.uuid4()),
        current_user=current_user,
        db=db,
    )

    assert response["display_name"] == "report.pdf"
    assert response["status"] == "PROCESSING"
    assert uuid.UUID(response["document_id"])
    assert uuid.UUID(response["version_id"])
    assert uuid.UUID(response["job_id"])
    assert events[0][1] == "DOCUMENT_INGESTION_REQUESTED"
    assert "session" not in str(events[0][3]).lower()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_rejects_mismatched_pdf_magic_bytes(monkeypatch):
    file = SimpleNamespace(
        filename="report.pdf",
        content_type="application/pdf",
        read=AsyncMock(return_value=b"plain text"),
    )

    with pytest.raises(HTTPException) as error:
        await documents.upload_document(file=file, current_user=user(), db=AsyncMock())

    assert error.value.status_code == 415


@pytest.mark.asyncio
async def test_upload_rejects_files_over_configured_size(monkeypatch):
    monkeypatch.setattr(documents, "MAX_UPLOAD_SIZE_BYTES", 3)
    file = SimpleNamespace(
        filename="notes.txt",
        content_type="text/plain",
        read=AsyncMock(return_value=b"four"),
    )

    with pytest.raises(HTTPException) as error:
        await documents.upload_document(file=file, current_user=user(), db=AsyncMock())

    assert error.value.status_code == 413


def test_upload_endpoint_is_collection_scoped_and_does_not_parse_inline():
    paths = {route.path for route in documents.router.routes}
    assert "/api/v1/documents" in paths
    assert "/api/v1/documents/upload" not in paths
    source = inspect.getsource(documents.upload_document)
    assert "parse_parent_child_chunks" not in source
    assert "process_and_store_document" not in source
