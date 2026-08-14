import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.routers.documents import router as documents_router
from app.routers.ingestion import router as ingestion_router
from app.routers.chat import router as chat_router
from app.database.connection import get_db, engine, Base
from app.database.models import User, ChatSession, Message
from app.database.enums import MessageRole, MessageStatus
from app.auth.security import get_current_user, verify_password, get_password_hash, create_access_token
from app.config import get_settings
from app.services.login_rate_limit import LoginRateLimiter
from app.database.repositories import (
    create_user, get_user_by_username, list_user_sessions,
    create_session, delete_session as delete_owned_session, get_session_by_id, get_session_messages
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_server")


from app.broker import broker
import taskiq_fastapi

from app.services.intent_classifier import train_classifier
from app.services.rag_engine import init_qdrant
from app.observability import metrics, request_id, structured_log, set_request_context

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.require_secret_key()
    settings.require_redis_url()
    taskiq_fastapi.init(broker, "app.main:app")
    if not broker.is_worker_process:
        await broker.startup()

    # Train the hybrid intent router in memory
    train_classifier()
    await init_qdrant()

    yield

    if not broker.is_worker_process:
        await broker.shutdown()
    await engine.dispose()


app = FastAPI(
    title="RAG Document Assistant Server",
    description="FastAPI Web Server for RAG (Retrieval-Augmented Generation) document search",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    started = time.perf_counter()
    correlation_id = request.headers.get("X-Request-ID") or request_id()
    set_request_context(request_id=correlation_id)
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        status_code = response.status_code if response is not None else 500
        structured_log(
            logger,
            "http_request",
            request_id=correlation_id,
            method=request.method,
            path=request.url.path,
            latency=round((time.perf_counter() - started) * 1000, 2),
            status=status_code,
            error_code=None if status_code < 400 else f"HTTP_{status_code}",
        )
        if response is not None:
            response.headers["X-Request-ID"] = correlation_id


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().validated_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(ingestion_router)
app.include_router(chat_router)


class UserSignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class NewSessionRequest(BaseModel):
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/api/v1/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: UserSignupRequest, db: AsyncSession = Depends(get_db)):
    """Registers a new user inside the database."""
    username_clean = request.username.strip()
    if not username_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty or whitespace."
        )

    existing_user = await get_user_by_username(db, username_clean)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    if len(request.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot be longer than 72 bytes."
        )

    hashed_password = get_password_hash(request.password)
    new_user = await create_user(db, username_clean, hashed_password)
    await db.commit()
    await db.refresh(new_user)

    return {"status": "success", "message": "User created successfully", "user_id": str(new_user.id)}


@app.post("/api/v1/auth/login")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticates credentials and returns a JWT access token."""
    username_clean = form_data.username.strip()
    client_ip = request.client.host if request and request.client else "unknown"
    limiter = LoginRateLimiter()
    await limiter.check(username_clean, client_ip)
    user = await get_user_by_username(db, username_clean)

    if len(form_data.password.encode('utf-8')) > 72:
        await limiter.record_failure(username_clean, client_ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    if not user or not verify_password(form_data.password, user.hashed_password):
        await limiter.record_failure(username_clean, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    await limiter.record_success(username_clean, client_ip)
    access_token = create_access_token(data={"sub": user.username, "user_id": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns a list of active sessions for the authenticated user.
    Only returns non-deleted sessions.
    """
    sessions = await list_user_sessions(db, current_user.id)
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


@app.post("/api/v1/sessions", status_code=status.HTTP_201_CREATED)
async def create_new_session(
    request: NewSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Creates exactly one fresh chat session with a server-generated UUID.
    A request cannot supply an id or replace an existing session.
    """
    new_session = await create_session(db, current_user.id, request.title or "New Conversation")
    await db.commit()
    await db.refresh(new_session)

    return {
        "status": "success",
        "session_id": str(new_session.id),
        "title": new_session.title,
    }


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes only the authenticated user's session and its conversation state.
    Global documents, their storage, and their vectors are not session-owned.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format. Must be a valid UUID."
        )

    deleted = await delete_owned_session(db, session_uuid, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )
    
    await db.commit()
    return {"status": "success", "message": "Session deleted."}


@app.get("/api/v1/sessions/{session_id}/history", include_in_schema=False)
@app.get("/api/v1/sessions/{session_id}/messages")
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Returns the chat message history for the target session.
    Messages are individual rows ordered by sequence_number.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format."
        )

    chat_session = await get_session_by_id(db, session_uuid, current_user.id)
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    messages = await get_session_messages(db, session_uuid, limit=limit, offset=offset)

    return {
        "items": [
        {
            "id": str(m.id),
            "parent_message_id": str(m.parent_message_id) if m.parent_message_id else None,
            "sequence_number": m.sequence_number,
            "role": m.role,
            "content": m.content,
            "status": m.status,
            "client_request_id": m.client_request_id,
            "selected_document_snapshot": m.selected_document_snapshot,
            "sources": m.sources,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
        ],
        "pagination": {"limit": limit, "offset": offset, "count": len(messages)},
    }


# ---------------------------------------------------------------------------
# Stubbed routes (pending Phase 2 rewrite)
# ---------------------------------------------------------------------------

@app.get("/api/v1/sessions/files")
async def get_session_files_stub(
    current_user: User = Depends(get_current_user),
):
    """
    STUB: SessionFile has been removed. Documents are now global.
    This endpoint will be replaced in Phase 2.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Route pending Phase 2 rewrite. SessionFile model removed. Use /api/v1/documents/global instead."
    )


@app.get("/")
def read_root():
    return {"status": "running", "message": "Welcome to the RAG Document Assistant API", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local development server on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
