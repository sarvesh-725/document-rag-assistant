"""Chat query routes and durable query-run lifecycle handling."""

import asyncio
import logging
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import User
from app.auth.security import get_current_user
from app.database.enums import MessageRole, QueryRunStatus
from app.database.repositories import (
    add_query_run_documents,
    create_message,
    create_query_run,
    get_session_by_id,
    transition_query_run_status,
)
from app.services.document_selection import (
    DocumentSelectionError,
    resolve_selected_documents,
)
from app.services.vector_access import owned_vector_filter

logger = logging.getLogger("chat_router")
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _qdrant_model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", exclude_none=True)
    return model.dict(exclude_none=True)


class ChatQueryRequest(BaseModel):
    session_id: str
    client_request_id: str
    question: str
    selected_document_ids: List[str] = Field(default_factory=list)


@router.post("/query")
async def query_chat_stream(
    request: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    http_request: Request = None,
):
    """
    Resolve selected documents through PostgreSQL before vector retrieval.

    The query run is committed as RUNNING before query work begins.  The
    current endpoint's query work is the durable scope setup; later retrieval
    stages use the same run and terminal transition helpers.
    """
    try:
        session_id = uuid.UUID(request.session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id.",
        ) from exc

    session = await get_session_by_id(db, session_id, current_user.id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized.",
        )

    try:
        resolved_documents = await resolve_selected_documents(
            db,
            authenticated_user_id=current_user.id,
            selected_document_ids=request.selected_document_ids,
        )
    except DocumentSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    resolved_version_ids = [document.version_id for document in resolved_documents]
    vector_filter = owned_vector_filter(current_user.id, resolved_version_ids)
    selected_document_snapshot = {
        "documents": [
            {
                "document_id": str(document.document_id),
                "version_id": str(document.version_id),
            }
            for document in resolved_documents
        ]
    }

    query_run = None
    try:
        message = await create_message(
            db,
            session_id,
            MessageRole.USER.value,
            content=request.question,
            client_request_id=request.client_request_id,
            selected_document_snapshot=selected_document_snapshot,
        )
        query_run = await create_query_run(
            db,
            session_id,
            current_user.id,
            message_id=message.id,
        )
        await add_query_run_documents(
            db,
            query_run.id,
            [(document.document_id, document.version_id) for document in resolved_documents],
        )
        # Make the active reference visible before any retrieval/generation
        # work, including to document deletion cleanup.
        await db.commit()

        if http_request is not None and await http_request.is_disconnected():
            transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
            await db.commit()
            return {"status": QueryRunStatus.CANCELLED.value, "query_run_id": str(query_run.id)}

        transition_query_run_status(query_run, QueryRunStatus.COMPLETED.value)
        await db.commit()
    except asyncio.CancelledError:
        if query_run is not None:
            try:
                transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
                await db.commit()
            except Exception:
                await db.rollback()
        raise
    except Exception:
        if query_run is not None:
            try:
                transition_query_run_status(query_run, QueryRunStatus.FAILED.value)
                await db.commit()
            except Exception:
                await db.rollback()
        raise

    return {
        "status": getattr(query_run, "status", QueryRunStatus.COMPLETED.value),
        "message_id": str(message.id),
        "query_run_id": str(query_run.id),
        "retrieval_scope": {
            "user_id": str(current_user.id),
            "version_ids": [str(version_id) for version_id in resolved_version_ids],
        },
        "qdrant_filter": _qdrant_model_dump(vector_filter),
    }
