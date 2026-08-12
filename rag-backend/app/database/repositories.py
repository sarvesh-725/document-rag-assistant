"""
Repository layer for database access.

Contains CRUD operations and complex queries for the 10 domain models.
All functions take an AsyncSession to participate in the caller's transaction.
"""

import uuid
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import (
    User, Document, DocumentVersion, ChatSession, Message,
    IngestionJob, QueryRun, QueryRunDocument, ConversationSummary, OutboxEvent,
)
from app.database.enums import DocumentStatus, VersionStatus


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
    stmt = select(Document).where(
        Document.user_id == user_id,
        Document.deleted_at.is_(None),
    ).order_by(Document.created_at.desc())
    
    result = await db.execute(stmt)
    return list(result.scalars().all())


from datetime import datetime
async def soft_delete_document(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    doc = await get_document_by_id(db, document_id, user_id)
    if not doc:
        return False
    doc.deleted_at = datetime.utcnow()
    doc.status = DocumentStatus.DELETING
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
) -> DocumentVersion:
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
        status=VersionStatus.PENDING,
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
) -> None:
    doc = await db.get(Document, document_id)
    if doc:
        doc.current_version_id = version_id
        doc.status = DocumentStatus.READY
        await db.flush()


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
    ).options(selectinload(ChatSession.messages))
    
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def list_user_sessions(db: AsyncSession, user_id: uuid.UUID) -> List[ChatSession]:
    stmt = select(ChatSession).where(
        ChatSession.user_id == user_id,
        ChatSession.deleted_at.is_(None),
    ).order_by(ChatSession.updated_at.desc())
    
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def soft_delete_session(
    db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    session = await get_session_by_id(db, session_id, user_id)
    if not session:
        return False
    session.deleted_at = datetime.utcnow()
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
    stmt = select(Message.sequence_number).where(
        Message.session_id == session_id
    ).order_by(Message.sequence_number.desc()).limit(1)
    
    result = await db.execute(stmt)
    latest_seq = result.scalar_one_or_none()
    next_seq = 1 if latest_seq is None else latest_seq + 1

    message = Message(
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


async def get_session_messages(db: AsyncSession, session_id: uuid.UUID) -> List[Message]:
    stmt = select(Message).where(
        Message.session_id == session_id
    ).order_by(Message.sequence_number.asc())
    
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
# IngestionJob Repository
# -----------------------------------------------------------------------------
async def create_ingestion_job(
    db: AsyncSession, document_id: uuid.UUID, version_id: uuid.UUID
) -> IngestionJob:
    job = IngestionJob(document_id=document_id, version_id=version_id)
    db.add(job)
    await db.flush()
    return job


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
    db: AsyncSession, event_type: str, aggregate_id: uuid.UUID, payload: dict
) -> OutboxEvent:
    event = OutboxEvent(
        event_type=event_type,
        aggregate_id=aggregate_id,
        payload=payload,
    )
    db.add(event)
    await db.flush()
    return event
