import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.conversation_context as conversation_context
from app.database.repositories import get_recent_session_messages


class Result:
    def __init__(self, rows=None, value=None):
        self.rows = rows or []
        self.value = value

    def scalars(self):
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


def message(sequence, role="user", content=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        sequence_number=sequence,
        role=role,
        content=content or f"message {sequence}",
        created_at=datetime.utcnow(),
    )


def test_summary_is_derived_without_removing_canonical_messages():
    summarizer = conversation_context.ConversationSummarizer()
    messages = [message(1), message(2, role="assistant")]

    summary = summarizer.summarize(messages, token_budget=100)

    assert "user: message 1" in summary
    assert "assistant: message 2" in summary
    assert messages[0].content == "message 1"


@pytest.mark.asyncio
async def test_summary_update_tracks_last_summarized_message_and_continues_safely(monkeypatch):
    session_id = uuid.uuid4()
    current = message(7)
    older = [message(index) for index in range(1, 7)]
    db = AsyncMock()
    db.get.return_value = current
    db.execute.side_effect = [Result(), Result(rows=older)]
    upsert = AsyncMock()
    monkeypatch.setattr(conversation_context, "get_or_create_conversation_summary", upsert)

    await conversation_context.maybe_update_summary(
        db,
        session_id,
        current.id,
        conversation_context.HistoryConfig(recent_messages=2),
    )

    upsert.assert_awaited_once()
    assert upsert.await_args.args[1] == session_id
    assert upsert.await_args.args[3] == older[-3].id


@pytest.mark.asyncio
async def test_recent_messages_are_returned_oldest_to_newest_and_current_can_be_excluded():
    newest = message(21)
    oldest = message(20)
    db = AsyncMock()
    db.execute.return_value = Result(rows=[newest, oldest])

    result = await get_recent_session_messages(
        db,
        uuid.uuid4(),
        limit=2,
        exclude_message_id=uuid.uuid4(),
    )

    assert [item.sequence_number for item in result] == [20, 21]
