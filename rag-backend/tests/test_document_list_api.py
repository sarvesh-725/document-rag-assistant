import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.routers.documents as documents


def make_document(status, version_status, *, current=True):
    document_id = uuid.uuid4()
    version = SimpleNamespace(
        id=uuid.uuid4(),
        version_number=1,
        status=version_status,
    )
    return SimpleNamespace(
        id=document_id,
        display_name="report (2).pdf",
        original_filename="report.pdf",
        created_at=datetime(2026, 8, 13, 12, 0),
        status=status,
        duplicate_index=2,
        current_version_id=version.id if current else None,
        versions=[version],
        deleted_at=None,
    )


@pytest.mark.asyncio
async def test_document_list_returns_ready_processing_and_failed_documents(monkeypatch):
    current_user = SimpleNamespace(id=uuid.uuid4())
    ready = make_document("READY", "READY")
    processing = make_document("PROCESSING", "PROCESSING")
    failed = make_document("PROCESSING", "FAILED", current=False)
    list_documents = AsyncMock(return_value=[ready, processing, failed])
    monkeypatch.setattr(documents, "list_user_documents", list_documents)
    db = AsyncMock()

    response = await documents.get_documents(current_user, db)

    assert [item["status"] for item in response] == [
        "READY",
        "PROCESSING",
        "FAILED",
    ]
    assert response[0] == {
        "document_id": str(ready.id),
        "display_name": "report (2).pdf",
        "original_filename": "report.pdf",
        "created_at": "2026-08-13T12:00:00",
        "status": "READY",
        "duplicate_index": 2,
    }
    list_documents.assert_awaited_once_with(db, current_user.id)


@pytest.mark.asyncio
async def test_compatibility_global_list_uses_same_non_deleted_contract(monkeypatch):
    current_user = SimpleNamespace(id=uuid.uuid4())
    failed = make_document("PROCESSING", "FAILED", current=False)
    monkeypatch.setattr(documents, "list_user_documents", AsyncMock(return_value=[failed]))

    response = await documents.get_global_documents(current_user, AsyncMock())

    assert len(response) == 1
    assert response[0]["status"] == "FAILED"


def test_canonical_document_list_route_exists():
    assert any(route.path == "/api/v1/documents" for route in documents.router.routes)
