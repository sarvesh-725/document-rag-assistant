import asyncio
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.routers.chat as chat
from app.database.enums import QueryRunStatus
from app.database.repositories import transition_query_run_status
from app.services.query_run_lifecycle import reconcile_stale_query_runs


def user():
    return SimpleNamespace(id=uuid.uuid4())


def request_for(*document_ids):
    session_id = uuid.uuid4()
    return (
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="request-1",
            question="What matters here?",
            selected_document_ids=[str(document_id) for document_id in document_ids],
        ),
        session_id,
    )


def configure_query(monkeypatch, current_user, session_id, query_run):
    document = SimpleNamespace(document_id=uuid.uuid4(), version_id=uuid.uuid4())
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "resolve_selected_documents", AsyncMock(return_value=[document]))
    monkeypatch.setattr(chat, "create_message", AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock(return_value=query_run))
    monkeypatch.setattr(chat, "add_query_run_documents", AsyncMock())
    monkeypatch.setattr(
        chat,
        "hybrid_retriever",
        SimpleNamespace(retrieve=AsyncMock(return_value=SimpleNamespace(context=[]))),
    )
    return document


@pytest.mark.asyncio
async def test_successful_query_transitions_running_to_completed(monkeypatch):
    current_user = user()
    request, session_id = request_for(uuid.uuid4())
    query_run = SimpleNamespace(id=uuid.uuid4(), status=QueryRunStatus.RUNNING.value, completed_at=None)
    configure_query(monkeypatch, current_user, session_id, query_run)
    db = AsyncMock()

    response = await chat.query_chat_stream(request, current_user, db)

    assert response["status"] == QueryRunStatus.COMPLETED.value
    assert query_run.status == QueryRunStatus.COMPLETED.value
    assert query_run.completed_at is not None
    assert db.commit.await_count == 2


@pytest.mark.asyncio
async def test_disconnected_client_cancels_running_query(monkeypatch):
    current_user = user()
    request, session_id = request_for(uuid.uuid4())
    query_run = SimpleNamespace(id=uuid.uuid4(), status=QueryRunStatus.RUNNING.value, completed_at=None)
    configure_query(monkeypatch, current_user, session_id, query_run)
    disconnected = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    db = AsyncMock()

    response = await chat.query_chat_stream(request, current_user, db, disconnected)

    assert response["status"] == QueryRunStatus.CANCELLED.value
    assert query_run.status == QueryRunStatus.CANCELLED.value
    assert query_run.completed_at is not None
    assert db.commit.await_count == 2


@pytest.mark.asyncio
async def test_application_failure_transitions_query_to_failed(monkeypatch):
    current_user = user()
    request, session_id = request_for(uuid.uuid4())
    query_run = SimpleNamespace(id=uuid.uuid4(), status=QueryRunStatus.RUNNING.value, completed_at=None)
    configure_query(monkeypatch, current_user, session_id, query_run)
    monkeypatch.setattr(chat, "add_query_run_documents", AsyncMock(side_effect=RuntimeError("retrieval failed")))

    with pytest.raises(RuntimeError, match="retrieval failed"):
        await chat.query_chat_stream(request, current_user, AsyncMock())

    assert query_run.status == QueryRunStatus.FAILED.value
    assert query_run.completed_at is not None


def test_terminal_query_transitions_are_idempotent_and_not_reversible():
    query_run = SimpleNamespace(
        id=uuid.uuid4(), status=QueryRunStatus.COMPLETED.value, completed_at=datetime.utcnow()
    )
    assert transition_query_run_status(query_run, QueryRunStatus.COMPLETED.value) is query_run
    with pytest.raises(ValueError, match="terminal"):
        transition_query_run_status(query_run, QueryRunStatus.FAILED.value)


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


@pytest.mark.asyncio
async def test_reconciliation_cancels_stale_running_queries():
    query_run = SimpleNamespace(
        id=uuid.uuid4(),
        status=QueryRunStatus.RUNNING.value,
        created_at=datetime.utcnow() - timedelta(hours=1),
        completed_at=None,
    )
    db = AsyncMock()
    db.execute.return_value = _Result([query_run])

    count = await reconcile_stale_query_runs(db, stale_after=timedelta(minutes=15))

    assert count == 1
    assert query_run.status == QueryRunStatus.CANCELLED.value
    assert query_run.completed_at is not None
    db.commit.assert_awaited_once()


def test_query_run_status_values_match_required_lifecycle():
    assert {status.value for status in QueryRunStatus} == {
        "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    }
