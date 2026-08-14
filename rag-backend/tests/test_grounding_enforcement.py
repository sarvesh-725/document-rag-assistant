import uuid
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest

import app.routers.chat as chat
from app.services.intent_classifier import QueryAnalysis


@pytest.mark.asyncio
async def test_zero_retrieval_evidence_creates_deterministic_completed_assistant_message(monkeypatch):
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    message = SimpleNamespace(id=uuid.uuid4())
    query_run = SimpleNamespace(id=uuid.uuid4(), status="RUNNING", completed_at=None)
    assistant = SimpleNamespace(id=uuid.uuid4())
    analysis = QueryAnalysis(
        normalized_query="what was revenue",
        intent="knowledge_specific",
        likely_needs_retrieval=True,
        is_obvious_chitchat=False,
    )
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "classify_intent", Mock(return_value=analysis))
    monkeypatch.setattr(
        chat,
        "resolve_selected_documents",
        AsyncMock(return_value=[SimpleNamespace(document_id=document_id, version_id=version_id)]),
    )
    monkeypatch.setattr(chat, "create_message", AsyncMock(side_effect=[message, assistant]))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock(return_value=query_run))
    monkeypatch.setattr(chat, "add_query_run_documents", AsyncMock())
    update_status = AsyncMock()
    monkeypatch.setattr(chat, "update_message_status", update_status)
    monkeypatch.setattr(
        chat,
        "hybrid_retriever",
        SimpleNamespace(retrieve=AsyncMock(return_value=SimpleNamespace(context=[]))),
    )

    response = await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="grounding-1",
            question="what was revenue",
            selected_document_ids=[str(document_id)],
        ),
        SimpleNamespace(id=user_id),
        AsyncMock(),
    )

    assert response["grounded"] is False
    assert response["answer"] == chat.NO_GROUNDING_RESPONSE
    assert response["retrieval"]["context"] == []
    assert chat.create_message.await_count == 2
    assert chat.create_message.await_args_list[1].kwargs["status"] == "STREAMING"
    assert chat.create_message.await_args_list[1].kwargs["parent_message_id"] == message.id
    update_status.assert_awaited_once_with(
        ANY,
        assistant.id,
        "COMPLETED",
        content=chat.NO_GROUNDING_RESPONSE,
    )
    assert query_run.status == "COMPLETED"
