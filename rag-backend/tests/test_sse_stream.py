import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.routers.chat as chat
from app.services.context_builder import ContextPackage
from app.services.intent_classifier import QueryAnalysis
from app.services.sse import format_sse_event


def package(*, evidence=None):
    return ContextPackage(
        system_prompt="safe",
        summary="",
        recent_messages=[],
        evidence=evidence or [],
        citations=[],
        current_question="question",
        estimated_tokens=2,
        query_analysis=QueryAnalysis(
            normalized_query="question",
            intent="knowledge_specific",
            likely_needs_retrieval=True,
            is_obvious_chitchat=False,
        ),
    )


def parse_frames(frames):
    parsed = []
    for frame in frames:
        lines = frame.strip().splitlines()
        event = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("event:"))
        data = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("data:"))
        parsed.append((event, json.loads(data)))
    return parsed


def test_sse_frames_are_typed_and_blank_line_delimited():
    frame = format_sse_event("token", {"text": "hello"})
    assert frame == 'event: token\ndata: {"text":"hello"}\n\n'


@pytest.mark.asyncio
async def test_stream_emits_required_events_and_completes_state():
    query_run = SimpleNamespace(id=uuid.uuid4(), status="RUNNING", completed_at=None)
    assistant = SimpleNamespace(id=uuid.uuid4())
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    db = AsyncMock()
    update = AsyncMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(chat, "update_message_status", update)
    try:
        frames = [
            frame
            async for frame in chat._stream_query_response(
                http_request=request,
                db=db,
                query_run=query_run,
                assistant_message=assistant,
                message=SimpleNamespace(id=uuid.uuid4()),
                context_package=package(),
                retrieval_result=None,
            )
        ]
    finally:
        monkeypatch.undo()

    events = parse_frames(frames)
    event_names = [event[0] for event in events]
    assert event_names[0:2] == ["message_start", "retrieval_complete"]
    assert "token" in event_names
    assert event_names[-1] == "message_complete"
    assert query_run.status == "COMPLETED"
    update.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_cancellation_persists_partial_content_and_cancelled_status():
    query_run = SimpleNamespace(id=uuid.uuid4(), status="RUNNING", completed_at=None)
    assistant = SimpleNamespace(id=uuid.uuid4())
    request = SimpleNamespace(is_disconnected=AsyncMock(side_effect=[False, True]))
    db = AsyncMock()
    update = AsyncMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(chat, "update_message_status", update)
    try:
        frames = [
            frame
            async for frame in chat._stream_query_response(
                http_request=request,
                db=db,
                query_run=query_run,
                assistant_message=assistant,
                message=SimpleNamespace(id=uuid.uuid4()),
                context_package=package(),
                retrieval_result=None,
            )
        ]
    finally:
        monkeypatch.undo()

    assert parse_frames(frames)[-1][0] == "cancelled"
    assert query_run.status == "CANCELLED"
    assert update.await_args.args[2] == "CANCELLED"
    assert update.await_args.kwargs["content"]
