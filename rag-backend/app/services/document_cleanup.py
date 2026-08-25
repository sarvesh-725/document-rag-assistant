"""Safe physical cleanup for logically deleted documents."""

import uuid
from datetime import datetime
from typing import Awaitable, Callable, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.enums import DocumentStatus, IngestionStatus, QueryRunStatus, VersionStatus
from app.database.models import (
    Document,
    DocumentParent,
    DocumentVersion,
    IngestionJob,
    QueryRun,
    QueryRunDocument,
)
from app.services.rag_engine import delete_version_vectors
from app.services.storage import LocalStorageService, StorageService


ACTIVE_QUERY_STATUSES = {
    QueryRunStatus.PENDING.value,
    QueryRunStatus.RUNNING.value,
}
ACTIVE_INGESTION_STATUSES = {
    IngestionStatus.PENDING.value,
    IngestionStatus.RUNNING.value,
    IngestionStatus.RETRYING.value,
}


def _now() -> datetime:
    return datetime.utcnow()


async def _has_active_query_reference(
    db: AsyncSession, document_id: uuid.UUID
) -> bool:
    result = await db.execute(
        select(QueryRunDocument.query_run_id)
        .join(QueryRun, QueryRun.id == QueryRunDocument.query_run_id)
        .where(
            QueryRunDocument.document_id == document_id,
            QueryRun.status.in_(ACTIVE_QUERY_STATUSES),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _has_active_ingestion(
    db: AsyncSession, document_id: uuid.UUID
) -> bool:
    result = await db.execute(
        select(IngestionJob.id)
        .where(
            IngestionJob.document_id == document_id,
            IngestionJob.status.in_(ACTIVE_INGESTION_STATUSES),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def cleanup_deleted_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    storage: Optional[StorageService] = None,
    delete_vectors: Callable[..., Awaitable[None]] = delete_version_vectors,
) -> bool:
    """Delete physical data once no query or ingestion still references it.

    Returns ``True`` when the document is fully deleted or was already in the
    terminal state.  Returns ``False`` when cleanup must be retried later.
    All operations are scoped by UUIDs and are safe to repeat after a worker
    crash: Qdrant deletes and storage deletes are both idempotent.
    """
    locked = await db.execute(
        select(Document).where(Document.id == document_id).with_for_update()
    )
    document = locked.scalar_one_or_none()
    if document is None or document.status == DocumentStatus.DELETED.value:
        await db.commit()
        return True
    if document.status != DocumentStatus.DELETING.value:
        await db.rollback()
        return False

    if await _has_active_query_reference(db, document.id):
        await db.commit()
        return False
    if await _has_active_ingestion(db, document.id):
        await db.commit()
        return False

    if storage is None:
        from app.config import get_settings

        storage = LocalStorageService(get_settings().storage_root)

    versions_result = await db.execute(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document.id)
        .order_by(DocumentVersion.version_number.asc())
    )
    versions = list(versions_result.scalars().all())
    for version in versions:
        # Do not mark the row DELETED unless the external index operation has
        # succeeded. A retry can safely repeat a successful delete.
        await delete_vectors(document.user_id, document.id, version.id)
        await db.execute(
            delete(DocumentParent).where(DocumentParent.version_id == version.id)
        )
        if version.storage_key:
            await storage.delete(version.storage_key, document.user_id)
        version.status = VersionStatus.DELETED.value
        version.updated_at = _now()

    document.current_version_id = None
    document.status = DocumentStatus.DELETED.value
    document.updated_at = _now()
    await db.commit()
    return True
