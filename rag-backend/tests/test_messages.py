"""Message-row persistence, ordering, idempotency, and history pagination tests."""

import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
import app.main as main
import app.database.repositories as repositories
from app.database.models import Message


def row(sequence_number: int, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), sequence_number=sequence_number, role=role, content="text",
        status="COMPLETED", selected_document_snapshot={"document_ids": ["doc-1"]},
        sources={"citations": []}, created_at=None,
    )


@pytest.mark.asyncio
async def test_append_assigns_server_sequence_and_user_snapshot():
    session_id = uuid.uuid4()
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[AsyncMock(), SimpleNamespace(scalar_one_or_none=lambda: 4)])

    message = await repositories.create_message(
        db, session_id, "user", content="Question", client_request_id="request-1",
        selected_document_snapshot={"document_ids": ["doc-1"]},
    )

    assert message.id is not None
    assert message.session_id == session_id
    assert message.sequence_number == 5
    assert message.client_request_id == "request-1"
    assert message.selected_document_snapshot == {"document_ids": ["doc-1"]}
    assert db.add.call_args.args[0] is message


def test_message_model_enforces_row_identity_and_constraints():
    constraints = {constraint.name for constraint in Message.__table__.constraints}
    assert "uq_session_sequence" in constraints
    assert "uq_session_client_request" in constraints
    assert Message.__table__.c.id.primary_key
    assert Message.__table__.c.session_id.nullable is False
    assert Message.__table__.c.role.nullable is False
    assert Message.__table__.c.status.nullable is False


@pytest.mark.asyncio
async def test_duplicate_user_request_is_rejected_before_a_second_message_is_created():
    db = AsyncMock()
    with pytest.raises(ValueError, match="client_request_id"):
        await repositories.create_message(db, uuid.uuid4(), "user", content="Question")


@pytest.mark.asyncio
async def test_message_metadata_is_kept_on_its_correct_role():
    db = AsyncMock()
    with pytest.raises(ValueError, match="selected document snapshot"):
        await repositories.create_message(
            db, uuid.uuid4(), "assistant", content="Answer",
            selected_document_snapshot={"document_ids": ["doc-1"]},
        )
    with pytest.raises(ValueError, match="sources"):
        await repositories.create_message(
            db, uuid.uuid4(), "user", content="Question", client_request_id="request-1",
            sources={"citations": []},
        )


@pytest.mark.asyncio
async def test_concurrent_writes_are_serialized_by_session_row_lock():
    source = inspect.getsource(repositories.create_message)
    assert "with_for_update" in source
    assert "uq_session_sequence" in {constraint.name for constraint in Message.__table__.constraints}


@pytest.mark.asyncio
async def test_messages_are_returned_in_ascending_sequence_order_and_paged(monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4())
    session_id = uuid.uuid4()
    messages = [row(3), row(4, "assistant")]
    monkeypatch.setattr(main, "get_session_by_id", AsyncMock(return_value=SimpleNamespace()))
    fetch = AsyncMock(return_value=messages)
    monkeypatch.setattr(main, "get_session_messages", fetch)

    response = await main.get_session_history(str(session_id), user, AsyncMock(), limit=2, offset=2)

    fetch.assert_awaited_once_with(ANY, session_id, limit=2, offset=2)
    assert [item["sequence_number"] for item in response["items"]] == [3, 4]
    assert response["pagination"] == {"limit": 2, "offset": 2, "count": 2}


def test_repository_history_query_has_explicit_ascending_order_and_bounds():
    source = inspect.getsource(repositories.get_session_messages)
    assert "order_by(Message.sequence_number.asc())" in source
    assert ".offset(offset).limit(limit)" in source


def test_assistant_messages_have_citation_metadata_field():
    assistant = Message(
        session_id=uuid.uuid4(), sequence_number=2, role="assistant", status="COMPLETED",
        content="Answer", sources={"citations": [{"document_id": "doc-1"}]},
    )
    assert assistant.sources["citations"][0]["document_id"] == "doc-1"
