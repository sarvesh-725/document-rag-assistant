import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, ANY

import pytest

import app.routers.chat as chat
from app.services.document_selection import ResolvedQueryDocument


def user():
    return SimpleNamespace(id=uuid.uuid4())


@pytest.mark.asyncio
async def test_query_creates_query_run_document_rows_from_resolved_versions(monkeypatch):
    current_user = user()
    session_id = uuid.uuid4()
    message_id = uuid.uuid4()
    query_run_id = uuid.uuid4()
    document_a = ResolvedQueryDocument(document_id=uuid.uuid4(), version_id=uuid.uuid4())
    document_b = ResolvedQueryDocument(document_id=uuid.uuid4(), version_id=uuid.uuid4())

    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "resolve_selected_documents", AsyncMock(return_value=[document_a, document_b]))
    monkeypatch.setattr(chat, "create_message", AsyncMock(return_value=SimpleNamespace(id=message_id)))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock(return_value=SimpleNamespace(id=query_run_id)))
    add_documents = AsyncMock(return_value=[])
    monkeypatch.setattr(chat, "add_query_run_documents", add_documents)
    monkeypatch.setattr(
        chat,
        "hybrid_retriever",
        SimpleNamespace(retrieve=AsyncMock(return_value=SimpleNamespace(context=[]))),
    )
    db = AsyncMock()

    response = await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="request-1",
            question="What matters here?",
            selected_document_ids=[str(document_a.document_id), str(document_b.document_id)],
        ),
        db,
        None,
        current_user.id,
    )

    chat.resolve_selected_documents.assert_awaited_once_with(
        db,
        authenticated_user_id=current_user.id,
        selected_document_ids=[str(document_a.document_id), str(document_b.document_id)],
    )
    chat.create_query_run.assert_awaited_once_with(
        db,
        session_id,
        current_user.id,
        message_id=message_id,
    )
    add_documents.assert_awaited_once_with(
        db,
        query_run_id,
        [(document_a.document_id, document_a.version_id), (document_b.document_id, document_b.version_id)],
    )
    assert response["retrieval_scope"]["version_ids"] == [
        str(document_a.version_id),
        str(document_b.version_id),
    ]
    assert "document_id" not in str(response["qdrant_filter"])
    assert db.commit.await_count == 2  # RUNNING snapshot, then COMPLETED
    assert response["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_query_rejects_invalid_selection_without_creating_query_run(monkeypatch):
    current_user = user()
    session_id = uuid.uuid4()
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "create_message", AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock())
    db = AsyncMock()

    with pytest.raises(chat.HTTPException) as error:
        await chat.query_chat_stream(
            chat.ChatQueryRequest(
                session_id=str(session_id),
                client_request_id="request-1",
                question="What matters here?",
                selected_document_ids=["not-a-uuid"],
            ),
            db,
            None,
            current_user.id,
        )

    assert error.value.status_code == 400
    chat.create_query_run.assert_not_awaited()
    db.commit.assert_not_awaited()
