import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.routers.chat as chat
from app.services.intent_classifier import QueryAnalysis


class ExpiredUser:
    """Mimics an expired async ORM identity that must never be touched."""

    @property
    def id(self):
        raise AssertionError("expired ORM user identity was accessed")


@pytest.mark.asyncio
async def test_chat_uses_scalar_authenticated_id_not_expired_user_orm(monkeypatch):
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    message = SimpleNamespace(id=uuid.uuid4(), selected_document_snapshot=None)
    query_run = SimpleNamespace(id=uuid.uuid4(), status="RUNNING")
    analysis = QueryAnalysis("Hi", "chitchat", False, True)
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "create_message", AsyncMock(return_value=message))
    monkeypatch.setattr(chat, "classify_intent", Mock(return_value=analysis))
    monkeypatch.setattr(chat, "resolve_selected_documents", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock(return_value=query_run))
    monkeypatch.setattr(chat, "add_query_run_documents", AsyncMock())
    monkeypatch.setattr(chat, "prompt_context_builder", SimpleNamespace(build=lambda **kwargs: SimpleNamespace(summary="", recent_messages=[], evidence=[], citations=[], estimated_tokens=0)))

    await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="scalar-id",
            question="question",
            selected_document_ids=[],
        ),
        AsyncMock(),
        None,
        user_id,
    )
