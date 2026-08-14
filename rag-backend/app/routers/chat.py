"""Chat query routes and durable query-run lifecycle handling."""

import asyncio
import logging
import uuid
from dataclasses import asdict
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
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
    get_assistant_message_for_query,
    get_query_run_for_message,
    get_session_by_id,
    transition_query_run_status,
    update_message_status,
)
from app.services.document_selection import (
    DocumentSelectionError,
    resolve_selected_documents,
)
from app.services.context_builder import ContextBuilder as PromptContextBuilder
from app.services.conversation_context import ConversationContext, load_conversation_context
from app.services.gemini_generation import GeminiAnswerStreamer
from app.services.intent_classifier import QueryAnalysis, classify_intent
from app.services.retrieval import (
    HybridRetriever,
    has_sufficient_evidence,
    serialize_context,
)
from app.services.sse import format_sse_event
from app.services.vector_access import owned_vector_filter

logger = logging.getLogger("chat_router")
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
hybrid_retriever = HybridRetriever()
prompt_context_builder = PromptContextBuilder()
gemini_answer_streamer = GeminiAnswerStreamer()
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

    assistant = None
    if isinstance(db, AsyncSession):
        assistant = await get_assistant_message_for_query(db, session_id, message.id)
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
        "assistant_message_id": str(assistant.id) if assistant else None,
        "answer": assistant.content if assistant else None,
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


async def _existing_sse_response(state: dict):
    yield format_sse_event(
        "message_start",
        {"message_id": state.get("message_id"), "query_run_id": state.get("query_run_id")},
    )
    yield format_sse_event("retrieval_complete", state.get("retrieval", {}))
    if state.get("answer"):
        yield format_sse_event("token", {"text": state["answer"]})
    yield format_sse_event(
        "message_complete",
        {"status": state.get("status"), "message_id": state.get("message_id")},
    )


class _ClientDisconnected(Exception):
    pass


async def _stream_query_response(
    *,
    http_request: Request,
    db: AsyncSession,
    query_run,
    assistant_message,
    message,
    context_package,
    retrieval_result,
):
    """Emit typed SSE events and persist terminal/partial state."""
    partial = ""
    grounded = retrieval_result is None or has_sufficient_evidence(retrieval_result)
    retrieval_data = {
        "grounded": grounded,
        "candidate_count": len(retrieval_result.reranked_candidates)
        if retrieval_result is not None
        else 0,
        "context_count": len(context_package.evidence),
    }
    try:
        yield format_sse_event(
            "message_start",
            {"message_id": str(assistant_message.id), "query_run_id": str(query_run.id)},
        )
        yield format_sse_event("retrieval_complete", retrieval_data)
        for index, citation in enumerate(context_package.citations, start=1):
            source = {
                "source_id": citation.get("source_id") or f"S{index}",
                "document_id": citation.get("document_id"),
                "version_id": citation.get("version_id"),
                "display_name": citation.get("display_name"),
                "section": citation.get("section"),
                "chunk_id": citation.get("chunk_id"),
            }
            if citation.get("page_start") is not None:
                source["page"] = citation["page_start"]
            yield format_sse_event("source", source)

        if not grounded:
            generated = [NO_GROUNDING_RESPONSE]
        else:
            generated = gemini_answer_streamer.stream(
                context_package.to_llm_messages()
            )
        if isinstance(generated, list):
            generated = iter(generated)
            for piece in generated:
                if await http_request.is_disconnected():
                    raise _ClientDisconnected()
                partial += piece
                yield format_sse_event("token", {"text": piece})
                await asyncio.sleep(0)
        else:
            async for piece in generated:
                if await http_request.is_disconnected():
                    raise _ClientDisconnected()
                partial += piece
                yield format_sse_event("token", {"text": piece})
                await asyncio.sleep(0)

        await update_message_status(
            db, assistant_message.id, MessageStatus.COMPLETED.value, content=partial
        )
        transition_query_run_status(query_run, QueryRunStatus.COMPLETED.value)
        await db.commit()
        yield format_sse_event(
            "message_complete",
            {"status": MessageStatus.COMPLETED.value, "message_id": str(assistant_message.id)},
        )
    except (_ClientDisconnected, asyncio.CancelledError) as exc:
        try:
            await update_message_status(
                db, assistant_message.id, MessageStatus.CANCELLED.value, content=partial
            )
            transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
            await db.commit()
        except Exception:
            await db.rollback()
        if not isinstance(exc, asyncio.CancelledError):
            yield format_sse_event(
                "cancelled",
                {"status": MessageStatus.CANCELLED.value, "message_id": str(assistant_message.id)},
            )
    except Exception as exc:
        try:
            await update_message_status(
                db, assistant_message.id, MessageStatus.FAILED.value, content=partial
            )
            transition_query_run_status(query_run, QueryRunStatus.FAILED.value)
            await db.commit()
        except Exception:
            await db.rollback()
        yield format_sse_event("error", {"message": str(exc)})


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
        existing_state = await _existing_request_response(
            db, existing_message, session_id, current_user.id
        )
        if http_request is not None:
            return StreamingResponse(
                _existing_sse_response(existing_state),
                media_type="text/event-stream",
            )
        return existing_state

    message = await create_message(
        db,
        session_id,
        MessageRole.USER.value,
        content=request.question,
        client_request_id=request.client_request_id,
        selected_document_snapshot=None,
    )
    if getattr(message, "_existing_client_request", False):
        await db.rollback()
        existing_state = await _existing_request_response(
            db, message, session_id, current_user.id
        )
        if http_request is not None:
            return StreamingResponse(
                _existing_sse_response(existing_state),
                media_type="text/event-stream",
            )
        return existing_state

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
    message.selected_document_snapshot = {
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
    context_package = None
    try:
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

        conversation_context = ConversationContext("", None, [])
        if isinstance(db, AsyncSession):
            conversation_context = await load_conversation_context(
                db,
                session_id,
                message.id,
            )
        context_package = prompt_context_builder.build(
            query_analysis=query_analysis,
            conversation_summary=conversation_context.summary,
            recent_messages=conversation_context.recent_messages,
            retrieved_evidence=(retrieval_result.context if retrieval_result is not None else []),
            current_question=query_analysis.normalized_query,
            current_message_id=message.id,
        )

        if http_request is not None and await http_request.is_disconnected():
            transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
            await db.commit()
            return {"status": QueryRunStatus.CANCELLED.value, "query_run_id": str(query_run.id)}

        if http_request is not None:
            assistant_message = await create_message(
                db,
                session_id,
                MessageRole.ASSISTANT.value,
                content="",
                sources={"documents": context_package.citations},
                status=MessageStatus.STREAMING.value,
            )
            await db.commit()
            return StreamingResponse(
                _stream_query_response(
                    http_request=http_request,
                    db=db,
                    query_run=query_run,
                    assistant_message=assistant_message,
                    message=message,
                    context_package=context_package,
                    retrieval_result=retrieval_result,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if retrieval_result is not None and not has_sufficient_evidence(retrieval_result):
            assistant_message = await create_message(
                db,
                session_id,
                MessageRole.ASSISTANT.value,
                content=NO_GROUNDING_RESPONSE,
                sources={"documents": []},
                status=MessageStatus.COMPLETED.value,
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
        "context_package": {
            "summary": context_package.summary if context_package else "",
            "recent_messages": context_package.recent_messages if context_package else [],
            "evidence": context_package.evidence if context_package else [],
            "citations": context_package.citations if context_package else [],
            "estimated_tokens": context_package.estimated_tokens if context_package else 0,
        },
    }
    if assistant_message is not None:
        response["answer"] = NO_GROUNDING_RESPONSE
        response["assistant_message_id"] = str(assistant_message.id)
    if vector_filter is not None:
        response["qdrant_filter"] = _qdrant_model_dump(vector_filter)
    return response
