"""Session lifecycle and tenant-isolation regression tests."""

import uuid
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from fastapi import HTTPException

import app.main as main
import app.database.repositories as repositories
from app.database.models import ChatSession, ConversationSummary, Document, Message, QueryRun


def user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), username=f"user-{uuid.uuid4()}")


@pytest.mark.asyncio
async def test_create_session_uses_a_backend_generated_uuid(monkeypatch):
    current_user = user()
    created = SimpleNamespace(id=uuid.uuid4(), title="Planning")
    create = AsyncMock(return_value=created)
    database = AsyncMock()
    monkeypatch.setattr(main, "create_session", create)

    response = await main.create_new_session(main.NewSessionRequest(title="Planning"), current_user, database)

    assert response["session_id"] == str(created.id)
    create.assert_awaited_once_with(database, current_user.id, "Planning")
    database.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_sessions_is_scoped_to_authenticated_user(monkeypatch):
    current_user, other_user = user(), user()
    session = SimpleNamespace(id=uuid.uuid4(), title="Mine", created_at=None)
    list_sessions = AsyncMock(return_value=[session])
    monkeypatch.setattr(main, "list_user_sessions", list_sessions)

    response = await main.get_user_sessions(current_user, AsyncMock())

    assert response == [{"id": str(session.id), "title": "Mine", "created_at": None}]
    list_sessions.assert_awaited_once_with(ANY, current_user.id)
    assert list_sessions.await_args.args[1] != other_user.id


@pytest.mark.asyncio
async def test_duplicate_create_requests_do_not_replace_an_existing_session(monkeypatch):
    current_user = user()
    first, second = (SimpleNamespace(id=uuid.uuid4(), title="New Conversation") for _ in range(2))
    create = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(main, "create_session", create)
    database = AsyncMock()

    first_response = await main.create_new_session(main.NewSessionRequest(), current_user, database)
    second_response = await main.create_new_session(main.NewSessionRequest(), current_user, database)

    assert first_response["session_id"] != second_response["session_id"]
    assert create.await_count == 2
    assert not hasattr(database, "delete") or database.delete.await_count == 0


@pytest.mark.asyncio
async def test_user_cannot_delete_another_users_session(monkeypatch):
    current_user, other_user = user(), user()
    delete = AsyncMock(return_value=False)
    monkeypatch.setattr(main, "delete_session", delete)

    with pytest.raises(HTTPException) as error:
        await main.delete_session(str(uuid.uuid4()), current_user, AsyncMock())

    assert error.value.status_code == 404
    delete.assert_awaited_once()
    assert delete.await_args.args[2] == current_user.id
    assert delete.await_args.args[2] != other_user.id


@pytest.mark.asyncio
async def test_deleting_active_session_removes_only_that_session(monkeypatch):
    current_user = user()
    active_session_id = uuid.uuid4()
    delete = AsyncMock(return_value=True)
    database = AsyncMock()
    monkeypatch.setattr(main, "delete_session", delete)

    response = await main.delete_session(str(active_session_id), current_user, database)

    assert response["status"] == "success"
    delete.assert_awaited_once_with(database, active_session_id, current_user.id)
    database.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_deletion_removes_only_the_session_aggregate(monkeypatch):
    current_user = user()
    session = SimpleNamespace(id=uuid.uuid4())
    database = AsyncMock()
    monkeypatch.setattr(repositories, "get_session_by_id", AsyncMock(return_value=session))

    assert await repositories.delete_session(database, session.id, current_user.id) is True

    database.delete.assert_awaited_once_with(session)
    database.flush.assert_awaited_once()


def test_session_database_cascades_cover_conversation_state_not_documents():
    def ondelete(model, column):
        return next(iter(model.__table__.c[column].foreign_keys)).ondelete

    assert ondelete(Message, "session_id") == "CASCADE"
    assert ondelete(QueryRun, "session_id") == "CASCADE"
    assert ondelete(ConversationSummary, "session_id") == "CASCADE"
    assert all(foreign_key.column.table.name != "chat_sessions" for foreign_key in Document.__table__.foreign_keys)
    assert ChatSession.__tablename__ == "chat_sessions"
