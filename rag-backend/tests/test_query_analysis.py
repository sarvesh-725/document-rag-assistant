import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import app.routers.chat as chat
from app.services import intent_classifier
from app.services.intent_classifier import QueryAnalysis


def test_pure_chitchat_is_detected_by_tier_one(monkeypatch):
    tier2 = Mock(side_effect=AssertionError("Tier 2 should not run for pure chitchat"))
    monkeypatch.setattr(intent_classifier, "_classify_tier2", tier2)

    for query in ("Hi", "Thanks", "How are you?"):
        analysis = intent_classifier.analyze_query(query)
        assert analysis.intent == "chitchat"
        assert analysis.is_obvious_chitchat is True
        assert analysis.likely_needs_retrieval is False
    tier2.assert_not_called()


@pytest.mark.parametrize(
    "query",
    ["Thanks, what was the revenue?", "Can you explain this paragraph?", "What does this document mean?"],
)
def test_mixed_or_factual_questions_need_retrieval(monkeypatch, query):
    tier2 = Mock(return_value="chitchat")
    monkeypatch.setattr(intent_classifier, "_classify_tier2", tier2)

    analysis = intent_classifier.analyze_query(query)

    assert analysis.is_obvious_chitchat is False
    assert analysis.likely_needs_retrieval is True
    tier2.assert_called_once()


def test_analysis_is_one_structured_result(monkeypatch):
    monkeypatch.setattr(intent_classifier, "_classify_tier2", Mock(return_value="knowledge_broad"))

    analysis = intent_classifier.analyze_query("What was the revenue?")

    assert isinstance(analysis, QueryAnalysis)
    assert analysis.normalized_query == "What was the revenue?"
    assert set(analysis.__dataclass_fields__) == {
        "normalized_query",
        "intent",
        "likely_needs_retrieval",
        "is_obvious_chitchat",
    }


def analysis(*, chitchat: bool, retrieval: bool) -> QueryAnalysis:
    return QueryAnalysis(
        normalized_query="Hi" if chitchat else "What was revenue?",
        intent="chitchat" if chitchat else "knowledge_specific",
        likely_needs_retrieval=retrieval,
        is_obvious_chitchat=chitchat,
    )


@pytest.mark.asyncio
async def test_no_documents_rejects_non_chitchat_in_application_policy(monkeypatch):
    session_id = uuid.uuid4()
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    classify = Mock(return_value=analysis(chitchat=False, retrieval=True))
    monkeypatch.setattr(chat, "classify_intent", classify)
    resolve = AsyncMock()
    monkeypatch.setattr(chat, "resolve_selected_documents", resolve)

    with pytest.raises(HTTPException) as error:
        await chat.query_chat_stream(
            chat.ChatQueryRequest(
                session_id=str(session_id),
                client_request_id="request-1",
                question="What was revenue?",
                selected_document_ids=[],
            ),
            SimpleNamespace(id=uuid.uuid4()),
            AsyncMock(),
        )

    assert error.value.status_code == 400
    assert error.value.detail == "Please select a document first."
    classify.assert_called_once_with("What was revenue?")
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_documents_allows_pure_chitchat_without_retrieval(monkeypatch):
    session_id = uuid.uuid4()
    query_run = SimpleNamespace(id=uuid.uuid4(), status="RUNNING")
    monkeypatch.setattr(chat, "get_session_by_id", AsyncMock(return_value=SimpleNamespace(id=session_id)))
    monkeypatch.setattr(chat, "get_message_by_client_request_id", AsyncMock(return_value=None))
    monkeypatch.setattr(chat, "classify_intent", Mock(return_value=analysis(chitchat=True, retrieval=False)))
    monkeypatch.setattr(chat, "create_message", AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())))
    monkeypatch.setattr(chat, "create_query_run", AsyncMock(return_value=query_run))
    monkeypatch.setattr(chat, "add_query_run_documents", AsyncMock())
    vector_filter = Mock(side_effect=AssertionError("chitchat must not retrieve"))
    monkeypatch.setattr(chat, "owned_vector_filter", vector_filter)

    response = await chat.query_chat_stream(
        chat.ChatQueryRequest(
            session_id=str(session_id),
            client_request_id="request-2",
            question="Hi",
            selected_document_ids=[],
        ),
        SimpleNamespace(id=uuid.uuid4()),
        AsyncMock(),
    )

    assert response["query_analysis"]["is_obvious_chitchat"] is True
    assert "qdrant_filter" not in response
    vector_filter.assert_not_called()
