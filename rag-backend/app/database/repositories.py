"""
Repository layer for database access.

Contains CRUD operations and complex queries for the 10 domain models.
All functions take an AsyncSession to participate in the caller's transaction.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import (
    User, Document, DocumentVersion, DocumentParent, ChatSession, Message,
    IngestionJob, QueryRun, QueryRunDocument, ConversationSummary, OutboxEvent,
)
from app.database.enums import (
    DocumentStatus, VersionStatus, IngestionStatus, IngestionStage, QueryRunStatus,
)


# -----------------------------------------------------------------------------
# User Repository
# -----------------------------------------------------------------------------
async def create_user(db: AsyncSession, username: str, hashed_password: str) -> User:
    user = User(username=username, hashed_password=hashed_password)
    db.add(user)
    await db.flush()
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    return await db.get(User, user_id)


# -----------------------------------------------------------------------------
# Document Repository
# -----------------------------------------------------------------------------
async def create_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    original_filename: str,
    document_id: Optional[uuid.UUID] = None,
) -> Document:
    """Create a new Document, automatically assigning the next duplicate_index and display_name."""
    stmt = select(Document).where(
        Document.user_id == user_id,
        Document.original_filename == original_filename,
    ).order_by(Document.duplicate_index.desc())
    
    result = await db.execute(stmt)
    latest_duplicate = result.scalars().first()
    
    next_index = 0
    if latest_duplicate is not None:
        next_index = latest_duplicate.duplicate_index + 1

    import os
    if next_index == 0:
        display_name = original_filename
    else:
        base, ext = os.path.splitext(original_filename)
        # Handle cases where multiple dots exist like .tar.gz
        if base.endswith('.tar') and ext == '.gz':
            base = base[:-4]
            ext = '.tar.gz'
        display_name = f"{base} ({next_index + 1}){ext}"

    doc = Document(
        id=document_id or uuid.uuid4(),
        user_id=user_id,
        display_name=display_name,
        original_filename=original_filename,
        duplicate_index=next_index,
        status=DocumentStatus.UPLOADING,
    )
    db.add(doc)
    await db.flush()
    return doc


async def get_document_by_id(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[Document]:
    stmt = select(Document).where(
        Document.id == document_id,
        Document.user_id == user_id,
        Document.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_user_documents(db: AsyncSession, user_id: uuid.UUID) -> List[Document]:
    stmt = (
        select(Document)
        .options(selectinload(Document.versions))
        .where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None),
        )
        .order_by(Document.created_at.desc())
    )
    
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def soft_delete_document(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    """Mark an owned document as deleting, safely under a row lock.

    The lock is shared with query snapshot creation.  This makes the commit
    that creates QueryRunDocument either happen before deletion starts or see
    DELETING and fail selection; it cannot be inserted after cleanup observes
    an unlocked document.
    """
    stmt = (
        select(Document)
        .where(Document.id == document_id, Document.user_id == user_id)
        .with_for_update()
    )
    doc = (await db.execute(stmt)).scalar_one_or_none()
    if not doc:
        return False
    if doc.status != DocumentStatus.DELETED.value:
        now = datetime.utcnow()
        if doc.deleted_at is None:
            doc.deleted_at = now
        doc.status = DocumentStatus.DELETING.value
        doc.updated_at = now
        await db.execute(
            update(DocumentVersion)
            .where(
                DocumentVersion.document_id == doc.id,
                DocumentVersion.status != VersionStatus.DELETED.value,
            )
            .values(status=VersionStatus.DELETING.value, updated_at=now)
        )
    await db.flush()
    return True


# -----------------------------------------------------------------------------
# DocumentVersion Repository
# -----------------------------------------------------------------------------
async def create_document_version(
    db: AsyncSession,
    document_id: uuid.UUID,
    content_hash: str,
    storage_key: str,
    parser_version: str,
    chunking_version: str,
    embedding_profile: str,
    version_id: Optional[uuid.UUID] = None,
    status: str = VersionStatus.PROCESSING.value,
) -> DocumentVersion:
    if status not in {item.value for item in VersionStatus}:
        raise ValueError(f"Invalid document version status: {status}")
    stmt = select(DocumentVersion).where(
        DocumentVersion.document_id == document_id
    ).order_by(DocumentVersion.version_number.desc())
    
    result = await db.execute(stmt)
    latest = result.scalars().first()
    
    next_version = 1 if latest is None else latest.version_number + 1

    version = DocumentVersion(
        document_id=document_id,
        version_number=next_version,
        content_hash=content_hash,
        storage_key=storage_key,
        status=status,
        parser_version=parser_version,
        chunking_version=chunking_version,
        embedding_profile=embedding_profile,
    )
    if version_id is not None:
        version.id = version_id
    db.add(version)
    await db.flush()
    return version


async def set_current_version(
    db: AsyncSession, document_id: uuid.UUID, version_id: uuid.UUID
) -> bool:
    """Promote a processing version only when its document is still live."""
    locked = await db.execute(
        select(Document).where(Document.id == document_id).with_for_update()
    )
    doc = locked.scalar_one_or_none()
    version = await db.get(DocumentVersion, version_id)
    if not doc or not version or version.document_id != doc.id:
        return False
    if doc.deleted_at is not None or doc.status in {
        DocumentStatus.DELETING.value, DocumentStatus.DELETED.value,
    }:
        return False
    if version.status != VersionStatus.PROCESSING.value:
        return False
    doc.current_version_id = version_id
    doc.status = DocumentStatus.READY.value
    version.status = VersionStatus.READY.value
    now = datetime.utcnow()
    doc.updated_at = now
    version.updated_at = now
    await db.flush()
    return True


async def replace_version_parents(db: AsyncSession, version_id: uuid.UUID, parents) -> list[DocumentParent]:
    """Persist the canonical parent snapshot for a version, idempotently."""
    await db.execute(delete(DocumentParent).where(DocumentParent.version_id == version_id))
    rows = []
    for parent in parents:
        row = DocumentParent(
            id=uuid.UUID(str(parent.id)), version_id=version_id,
            parent_index=parent.parent_index, text=parent.text,
            page_start=parent.page_start, page_end=parent.page_end,
            section=parent.section, element_type=parent.element_type,
            source_position=parent.source_position,
            parser_version=parent.parser_version,
            chunking_version=parent.chunking_version,
        )
        db.add(row); rows.append(row)
    await db.flush()
    return rows


# -----------------------------------------------------------------------------
# ChatSession Repository
# -----------------------------------------------------------------------------
async def create_session(db: AsyncSession, user_id: uuid.UUID, title: str) -> ChatSession:
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.flush()
    return session


async def get_session_by_id(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[ChatSession]:
    stmt = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
        ChatSession.deleted_at.is_(None),
    )
    
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_user_sessions(db: AsyncSession, user_id: uuid.UUID) -> List[ChatSession]:
    stmt = select(ChatSession).where(
        ChatSession.user_id == user_id,
        ChatSession.deleted_at.is_(None),
    ).order_by(ChatSession.updated_at.desc())
    
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_session(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    """Delete one owned conversation aggregate and its dependent rows.

    Database cascades remove messages, query runs, query-run document snapshots,
    and the conversation summary. Documents are not related to a session and are
    therefore deliberately untouched.
    """
    session = await get_session_by_id(db, session_id, user_id)
    if not session:
        return False
    await db.delete(session)
    await db.flush()
    return True


# -----------------------------------------------------------------------------
# Message Repository
# -----------------------------------------------------------------------------
async def create_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: Optional[str] = None,
    client_request_id: Optional[str] = None,
    selected_document_snapshot: Optional[dict] = None,
    sources: Optional[dict] = None,
) -> Message:
    """Append a row message with a transaction-safe, server-assigned sequence.

    Locking the parent session serializes appends from multiple requests. The
    database unique constraints remain the final guard against duplicate client
    requests and sequence values.
    """
    if role not in {"user", "assistant", "system"}:
        raise ValueError("Message role must be user, assistant, or system")
    if role == "user" and not client_request_id:
        raise ValueError("User messages require client_request_id")
    if role != "user" and client_request_id is not None:
        raise ValueError("Only user messages may have client_request_id")
    if role != "user" and selected_document_snapshot is not None:
        raise ValueError("Only user messages may store a selected document snapshot")
    if role != "assistant" and sources is not None:
        raise ValueError("Only assistant messages may store sources")

    await db.execute(
        select(ChatSession.id).where(ChatSession.id == session_id).with_for_update()
    )
    stmt = select(Message.sequence_number).where(Message.session_id == session_id).order_by(
        Message.sequence_number.desc()
    ).limit(1)
    latest_seq = (await db.execute(stmt)).scalar_one_or_none()
    next_seq = 1 if latest_seq is None else latest_seq + 1

    message = Message(
        id=uuid.uuid4(),
        session_id=session_id,
        sequence_number=next_seq,
        role=role,
        content=content,
        client_request_id=client_request_id,
        selected_document_snapshot=selected_document_snapshot,
        sources=sources,
    )
    db.add(message)
    await db.flush()
    return message


async def get_session_messages(
    db: AsyncSession, session_id: uuid.UUID, *, limit: int = 50, offset: int = 0
) -> List[Message]:
    """Fetch one bounded, deterministically ordered page of message rows."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if offset < 0:
        raise ValueError("offset must not be negative")
    stmt = select(Message).where(
        Message.session_id == session_id
    ).order_by(Message.sequence_number.asc()).offset(offset).limit(limit)
    
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_message_status(
    db: AsyncSession, message_id: uuid.UUID, status: str, content: str = None
) -> None:
    stmt = update(Message).where(Message.id == message_id).values(status=status)
    if content is not None:
        stmt = stmt.values(content=content)
    await db.execute(stmt)
    await db.flush()


# -----------------------------------------------------------------------------
# QueryRun Repository
# -----------------------------------------------------------------------------
async def create_query_run(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    message_id: Optional[uuid.UUID] = None,
    status: str = QueryRunStatus.RUNNING.value,
) -> QueryRun:
    if status not in {item.value for item in QueryRunStatus}:
        raise ValueError(f"Invalid query run status: {status}")
    query_run = QueryRun(
        id=uuid.uuid4(),
        session_id=session_id,
        user_id=user_id,
        message_id=message_id,
        status=status,
    )
    db.add(query_run)
    await db.flush()
    return query_run


def transition_query_run_status(
    query_run: QueryRun, status: str, *, completed_at: Optional[datetime] = None
) -> QueryRun:
    """Apply one legal QueryRun lifecycle transition in memory.

    The caller commits the surrounding transaction.  Terminal transitions are
    idempotent, which lets cancellation and reconciliation safely race with a
    normal completion path.
    """
    valid_statuses = {item.value for item in QueryRunStatus}
    if status not in valid_statuses:
        raise ValueError(f"Invalid query run status: {status}")
    current_status = getattr(query_run, "status", QueryRunStatus.RUNNING.value)
    if current_status in {
        QueryRunStatus.COMPLETED.value,
        QueryRunStatus.FAILED.value,
        QueryRunStatus.CANCELLED.value,
    }:
        if current_status != status:
            raise ValueError(
                f"Cannot transition terminal QueryRun {query_run.id} "
                f"from {current_status} to {status}"
            )
        return query_run
    if current_status not in {
        QueryRunStatus.PENDING.value,
        QueryRunStatus.RUNNING.value,
    }:
        raise ValueError(f"Cannot transition QueryRun from {current_status}")
    if (
        current_status == QueryRunStatus.PENDING.value
        and status != QueryRunStatus.RUNNING.value
    ):
        raise ValueError("A PENDING QueryRun must transition to RUNNING first")
    query_run.status = status
    if status in {
        QueryRunStatus.COMPLETED.value,
        QueryRunStatus.FAILED.value,
        QueryRunStatus.CANCELLED.value,
    }:
        query_run.completed_at = completed_at or datetime.utcnow()
    return query_run


async def add_query_run_documents(
    db: AsyncSession,
    query_run_id: uuid.UUID,
    resolved_documents: list[tuple[uuid.UUID, uuid.UUID]],
) -> list[QueryRunDocument]:
    rows = []
    for document_id, version_id in resolved_documents:
        row = QueryRunDocument(
            query_run_id=query_run_id,
            document_id=document_id,
            version_id=version_id,
        )
        db.add(row)
        rows.append(row)
    await db.flush()
    return rows


# -----------------------------------------------------------------------------
# IngestionJob Repository
# -----------------------------------------------------------------------------
async def create_ingestion_job(
    db: AsyncSession, document_id: uuid.UUID, version_id: uuid.UUID,
    job_id: Optional[uuid.UUID] = None, status: str = "PENDING", stage: str = "UPLOAD"
) -> IngestionJob:
    if status not in {item.value for item in IngestionStatus}:
        raise ValueError(f"Invalid ingestion job status: {status}")
    if stage not in {item.value for item in IngestionStage}:
        raise ValueError(f"Invalid ingestion stage: {stage}")
    job = IngestionJob(id=job_id or uuid.uuid4(), document_id=document_id, version_id=version_id, status=status, stage=stage)
    db.add(job)
    await db.flush()
    return job


async def get_ingestion_job_for_user(
    db: AsyncSession, job_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[IngestionJob]:
    """Fetch an ingestion job only through its owning document."""
    stmt = (
        select(IngestionJob)
        .join(Document, Document.id == IngestionJob.document_id)
        .where(
            IngestionJob.id == job_id,
            Document.user_id == user_id,
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# -----------------------------------------------------------------------------
# ConversationSummary & OutboxEvent
# -----------------------------------------------------------------------------
async def get_or_create_conversation_summary(
    db: AsyncSession, session_id: uuid.UUID, new_summary: str, last_message_id: uuid.UUID
) -> ConversationSummary:
    stmt = select(ConversationSummary).where(ConversationSummary.session_id == session_id)
    result = await db.execute(stmt)
    summary = result.scalar_one_or_none()

    if summary:
        summary.summary = new_summary
        summary.last_summarized_message_id = last_message_id
    else:
        summary = ConversationSummary(
            session_id=session_id,
            summary=new_summary,
            last_summarized_message_id=last_message_id,
        )
        db.add(summary)
    
    await db.flush()
    return summary


async def create_outbox_event(
    db: AsyncSession, event_type: str, aggregate_id: uuid.UUID, payload: dict,
    event_id: Optional[uuid.UUID] = None,
) -> OutboxEvent:
    event = OutboxEvent(
        id=event_id or uuid.uuid4(),
        event_type=event_type,
        aggregate_id=aggregate_id,
        payload=payload,
    )
    db.add(event)
    await db.flush()
    return event
