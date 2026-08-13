"""
Domain models for the Document RAG Assistant.

Phase 1 — Frozen domain model.

Models:
    User, Document, DocumentVersion, ChatSession, Message,
    IngestionJob, QueryRun, QueryRunDocument, ConversationSummary, OutboxEvent

Key design decisions:
    - All primary keys are UUID (server-generated via uuid4)
    - Documents are GLOBAL within a user account (no session-document binding)
    - Messages are individual rows (not a JSON blob in ChatSession)
    - Soft-delete via deleted_at on Document and ChatSession
    - QueryRunDocument captures the exact document versions used for each query
    - ConversationSummary is one-to-one with ChatSession
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    String, Integer, Float, ForeignKey, DateTime, Text, Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base
from app.database.enums import (
    DocumentStatus, VersionStatus, MessageRole, MessageStatus,
    IngestionStatus, IngestionStage, QueryRunStatus,
)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=DocumentStatus.UPLOADING, nullable=False
    )
    current_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", use_alter=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")
    versions: Mapped[List["DocumentVersion"]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="DocumentVersion.document_id",
    )
    current_version: Mapped[Optional["DocumentVersion"]] = relationship(
        "DocumentVersion",
        foreign_keys=[current_version_id],
        post_update=True,
        uselist=False,
    )
    ingestion_jobs: Mapped[List["IngestionJob"]] = relationship(
        "IngestionJob", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
    )


# ---------------------------------------------------------------------------
# DocumentVersion
# ---------------------------------------------------------------------------
class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    storage_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=VersionStatus.PROCESSING, nullable=False
    )
    parser_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    chunking_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    embedding_profile: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="versions", foreign_keys=[document_id]
    )
    parents: Mapped[List["DocumentParent"]] = relationship(
        "DocumentParent", back_populates="version", cascade="all, delete-orphan",
        order_by="DocumentParent.parent_index",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_doc_version"),
    )


class DocumentParent(Base):
    __tablename__ = "document_parents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    parent_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    element_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source_position: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="parents")

    __table_args__ = (
        UniqueConstraint("version_id", "parent_index", name="uq_document_parent_index"),
        Index("ix_document_parents_version_index", "version_id", "parent_index"),
    )


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    state_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan",
        order_by="Message.sequence_number",
    )
    query_runs: Mapped[List["QueryRun"]] = relationship(
        "QueryRun", back_populates="session", cascade="all, delete-orphan"
    )
    conversation_summary: Mapped[Optional["ConversationSummary"]] = relationship(
        "ConversationSummary", back_populates="session", uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_chat_sessions_user_deleted", "user_id", "deleted_at"),
    )


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=MessageStatus.PENDING, nullable=False
    )
    client_request_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    selected_document_snapshot: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    sources: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="messages"
    )

    __table_args__ = (
        Index("ix_messages_session_seq", "session_id", "sequence_number"),
        UniqueConstraint(
            "session_id", "sequence_number", name="uq_session_sequence"
        ),
        UniqueConstraint(
            "session_id", "client_request_id", name="uq_session_client_request"
        ),
    )


# ---------------------------------------------------------------------------
# IngestionJob
# ---------------------------------------------------------------------------
class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), default=IngestionStatus.PENDING, nullable=False
    )
    stage: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="ingestion_jobs"
    )
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion")


# ---------------------------------------------------------------------------
# QueryRun
# ---------------------------------------------------------------------------
class QueryRun(Base):
    __tablename__ = "query_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default=QueryRunStatus.PENDING, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="query_runs"
    )
    user: Mapped["User"] = relationship("User")
    message: Mapped[Optional["Message"]] = relationship("Message")
    query_run_documents: Mapped[List["QueryRunDocument"]] = relationship(
        "QueryRunDocument", back_populates="query_run", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# QueryRunDocument
# ---------------------------------------------------------------------------
class QueryRunDocument(Base):
    __tablename__ = "query_run_documents"

    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("query_runs.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), primary_key=True
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id"), nullable=False
    )

    # Relationships
    query_run: Mapped["QueryRun"] = relationship(
        "QueryRun", back_populates="query_run_documents"
    )
    document: Mapped["Document"] = relationship("Document")
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion")


# ---------------------------------------------------------------------------
# ConversationSummary
# ---------------------------------------------------------------------------
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_summarized_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="conversation_summary"
    )
    last_summarized_message: Mapped[Optional["Message"]] = relationship("Message")


# ---------------------------------------------------------------------------
# OutboxEvent
# ---------------------------------------------------------------------------
class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    aggregate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("ix_outbox_unpublished", "published_at", "created_at"),
    )
