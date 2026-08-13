"""Authorization regression tests for user-owned application resources."""

import uuid
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from fastapi import HTTPException

import app.main as main
import app.routers.documents as documents


def user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), username=f"user-{uuid.uuid4()}")


@pytest.mark.asyncio
async def test_user_a_cannot_access_user_bs_session(monkeypatch):
    user_a, user_b = user(), user()
    session_id = uuid.uuid4()
    get_session = AsyncMock(return_value=None)
    monkeypatch.setattr(main, "get_session_by_id", get_session)

    with pytest.raises(HTTPException) as error:
        await main.get_session_history(str(session_id), user_a, AsyncMock())

    assert error.value.status_code == 404
    get_session.assert_awaited_once_with(ANY, session_id, user_a.id)
    assert get_session.await_args.args[2] != user_b.id


@pytest.mark.asyncio
async def test_user_a_cannot_access_user_bs_document(monkeypatch):
    user_a, user_b = user(), user()
    list_documents = AsyncMock(return_value=[])
    monkeypatch.setattr(documents, "list_user_documents", list_documents)

    response = await documents.get_global_documents(user_a, AsyncMock())

    assert response == []
    list_documents.assert_awaited_once_with(ANY, user_a.id)
    assert list_documents.await_args.args[1] != user_b.id


@pytest.mark.asyncio
async def test_user_a_cannot_delete_user_bs_document(monkeypatch):
    user_a, user_b = user(), user()
    document_id = uuid.uuid4()
    delete_document = AsyncMock(return_value=False)
    monkeypatch.setattr(documents, "soft_delete_document", delete_document)

    with pytest.raises(HTTPException) as error:
        await documents.delete_document(str(document_id), user_a, AsyncMock())

    assert error.value.status_code == 404
    delete_document.assert_awaited_once_with(ANY, document_id, user_a.id)
    assert delete_document.await_args.args[2] != user_b.id


def test_user_a_vector_filter_cannot_match_user_bs_vectors():
    """Vector reads must be constrained by the authenticated owner's UUID."""
    from app.services.vector_access import owned_vector_filter

    user_a, user_b = user(), user()
    vector_filter = owned_vector_filter(user_a.id, [uuid.uuid4()])

    owner_condition = next(condition for condition in vector_filter.must if condition.key == "user_id")
    version_condition = next(condition for condition in vector_filter.must if condition.key == "version_id")
    assert owner_condition.match.value == str(user_a.id)
    assert owner_condition.match.value != str(user_b.id)
    assert version_condition.match.any
