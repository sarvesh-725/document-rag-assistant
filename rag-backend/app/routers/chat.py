"""Chat query routes and durable query-run lifecycle handling."""

import asyncio
import logging
import uuid
from dataclasses import asdict
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import User
from app.auth.security import get_current_user
from app.database.enums import MessageRole, MessageStatus, QueryRunStatus
from app.database.repositories import (
    add_query_run_documents,
    create_message,
    create_query_run,
    get_message_by_client_request_id,
    get_query_run_for_message,
    get_session_by_id,
    transition_query_run_status,
    update_message_status,
)
from app.services.document_selection import (
    DocumentSelectionError,
    resolve_selected_documents,
)
from app.services.intent_classifier import QueryAnalysis, classify_intent
from app.services.retrieval import (
    HybridRetriever,
    has_sufficient_evidence,
    serialize_context,
)
from app.services.vector_access import owned_vector_filter

logger = logging.getLogger("chat_router")
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
hybrid_retriever = HybridRetriever()
NO_GROUNDING_RESPONSE = "The selected documents do not contain enough information to answer this question."


def _qdrant_model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", exclude_none=True)
    return model.dict(exclude_none=True)


async def _existing_request_response(
    db: AsyncSession,
    message,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
) -> dict:
    """Reconnect a retry to the state created by the original request."""
    query_run = await get_query_run_for_message(db, message.id, session_id, user_id)
    if query_run is None:
        return {
            "status": "PENDING",
            "message_id": str(message.id),
            "query_run_id": None,
            "client_request_id": message.client_request_id,
        }

    version_ids = [row.version_id for row in query_run.query_run_documents]
    query_analysis = (
        getattr(message, "selected_document_snapshot", None) or {}
    ).get("query_analysis")
    response = {
        "status": query_run.status,
        "message_id": str(message.id),
        "query_run_id": str(query_run.id),
        "client_request_id": message.client_request_id,
        "query_analysis": query_analysis,
        "retrieval_scope": {
            "user_id": str(user_id),
            "version_ids": [str(version_id) for version_id in version_ids],
        },
    }
    if version_ids and (
        query_analysis is None or query_analysis.get("likely_needs_retrieval", True)
    ):
        response["qdrant_filter"] = _qdrant_model_dump(
            owned_vector_filter(user_id, version_ids)
        )
    return response


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

    existing_message = await get_message_by_client_request_id(
        db, session_id, request.client_request_id
    )
    if existing_message is not None:
        return await _existing_request_response(
            db, existing_message, session_id, current_user.id
        )

    query_analysis: QueryAnalysis = classify_intent(request.question)
    if not request.selected_document_ids and not query_analysis.is_obvious_chitchat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please select a document first.",
        )

    resolved_documents = []
    try:
        if request.selected_document_ids:
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
    vector_filter = (
        owned_vector_filter(current_user.id, resolved_version_ids)
        if query_analysis.likely_needs_retrieval and resolved_version_ids
        else None
    )
    selected_document_snapshot = {
        "query_analysis": asdict(query_analysis),
        "documents": [
            {
                "document_id": str(document.document_id),
                "version_id": str(document.version_id),
            }
            for document in resolved_documents
        ]
    }

    query_run = None
    retrieval_result = None
    assistant_message = None
    try:
        message = await create_message(
            db,
            session_id,
            MessageRole.USER.value,
            content=request.question,
            client_request_id=request.client_request_id,
            selected_document_snapshot=selected_document_snapshot,
        )
        if getattr(message, "_existing_client_request", False):
            await db.rollback()
            return await _existing_request_response(
                db, message, session_id, current_user.id
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

        if vector_filter is not None:
            retrieval_result = await hybrid_retriever.retrieve(
                query_analysis.normalized_query,
                current_user.id,
                resolved_version_ids,
                db,
            )

        if http_request is not None and await http_request.is_disconnected():
            transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
            await db.commit()
            return {"status": QueryRunStatus.CANCELLED.value, "query_run_id": str(query_run.id)}

        if retrieval_result is not None and not has_sufficient_evidence(retrieval_result):
            assistant_message = await create_message(
                db,
                session_id,
                MessageRole.ASSISTANT.value,
                content=NO_GROUNDING_RESPONSE,
                sources={"documents": []},
            )
            await update_message_status(
                db,
                assistant_message.id,
                MessageStatus.COMPLETED.value,
                content=NO_GROUNDING_RESPONSE,
            )

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

    response = {
        "status": getattr(query_run, "status", QueryRunStatus.COMPLETED.value),
        "message_id": str(message.id),
        "query_run_id": str(query_run.id),
        "retrieval_scope": {
            "user_id": str(current_user.id),
            "version_ids": [str(version_id) for version_id in resolved_version_ids],
        },
        "query_analysis": asdict(query_analysis),
        "retrieval": {
            "required": query_analysis.likely_needs_retrieval,
            "context": serialize_context(retrieval_result)
            if retrieval_result is not None
            else [],
        },
        "grounded": retrieval_result is None or has_sufficient_evidence(retrieval_result),
    }
    if assistant_message is not None:
        response["answer"] = NO_GROUNDING_RESPONSE
        response["assistant_message_id"] = str(assistant_message.id)
    if vector_filter is not None:
        response["qdrant_filter"] = _qdrant_model_dump(vector_filter)
    return response
