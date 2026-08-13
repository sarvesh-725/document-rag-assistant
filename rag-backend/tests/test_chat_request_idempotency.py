import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.routers.chat as chat
from app.database.enums import QueryRunStatus
from app.database.repositories import create_message


class Result:
    def __init__(self, value=None):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


def user():
    return SimpleNamespace(id=uuid.uuid4())


@pytest.mark.asyncio
async def test_duplicate_request_reconnects_to_existing_query_run(monkeypatch):
    current_user = user()
    session_id = uuid.uuid4()
    message = SimpleNamespace(id=uuid.uuid4(), client_request_id="request-1")
    version_id = uuid.uuid4()
    query_run = SimpleNamespace(
        id=uuid.uuid4(),
        status=QueryRunStatus.COMPLETED.value,
        query_run_documents=[SimpleNamespace(version_id=version_id)],
    )
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=message))
    monkeypatch.setattr(chat, "get_query_run_for_message", AsyncMock(return_value=query_run))
    create_run = AsyncMock()
    monkeypatch.setattr(chat, "create_query_run", create_run)

    response = await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="request-1",
            question="A retry with the same ID",
            selected_document_ids=[str(uuid.uuid4())],
        ),
        current_user,
        AsyncMock(),
    )

    assert response["status"] == QueryRunStatus.COMPLETED.value
    assert response["message_id"] == str(message.id)
    assert response["query_run_id"] == str(query_run.id)
    assert response["retrieval_scope"]["version_ids"] == [str(version_id)]
    create_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_running_request_returns_running_state(monkeypatch):
    current_user = user()
    session_id = uuid.uuid4()
    message = SimpleNamespace(id=uuid.uuid4(), client_request_id="request-2")
    query_run = SimpleNamespace(
        id=uuid.uuid4(),
        status=QueryRunStatus.RUNNING.value,
        query_run_documents=[],
    )
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=message))
    monkeypatch.setattr(chat, "get_query_run_for_message", AsyncMock(return_value=query_run))

    response = await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="request-2",
            question="retry",
            selected_document_ids=[str(uuid.uuid4())],
        ),
        current_user,
        AsyncMock(),
    )

    assert response["status"] == QueryRunStatus.RUNNING.value
    assert response["query_run_id"] == str(query_run.id)


@pytest.mark.asyncio
async def test_create_message_reuses_duplicate_request_under_session_lock():
    existing = SimpleNamespace(id=uuid.uuid4(), client_request_id="request-3")
    db = AsyncMock()
    db.execute.side_effect = [Result(), Result(existing)]

    result = await create_message(
        db,
        uuid.uuid4(),
        "user",
        content="retry",
        client_request_id="request-3",
    )

    assert result is existing
    assert result._existing_client_request is True
    db.flush.assert_not_awaited()
