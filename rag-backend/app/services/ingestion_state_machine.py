"""Durable ingestion state machine and reconciliation operations."""

import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.enums import (
    DocumentStatus,
    IngestionStage,
    IngestionStatus,
    VersionStatus,
)
from app.database.models import Document, DocumentVersion, IngestionJob
from app.database.repositories import create_outbox_event


STAGE_ORDER = tuple(IngestionStage)
_STAGE_INDEX = {stage.value: index for index, stage in enumerate(STAGE_ORDER)}
TERMINAL_JOB_STATUSES = {
    IngestionStatus.COMPLETED.value,
    IngestionStatus.FAILED.value,
    IngestionStatus.CANCELLED.value,
}


class RetryableIngestionError(RuntimeError):
    """A transient error; source data must be retained and the job retried."""

    def __init__(self, message: str, code: str = "RETRYABLE_ERROR") -> None:
        super().__init__(message)
        self.code = code


class NonRetryableIngestionError(RuntimeError):
    """A permanent error; the job/version become FAILED and source is retained."""

    def __init__(self, message: str, code: str = "INGESTION_FAILED") -> None:
        super().__init__(message)
        self.code = code


def _now() -> datetime:
    return datetime.utcnow()


async def get_ingestion_job(db: AsyncSession, job_id: uuid.UUID) -> Optional[IngestionJob]:
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    return result.scalar_one_or_none()


async def transition_stage(
    db: AsyncSession, job_id: uuid.UUID, stage: IngestionStage | str
) -> Optional[IngestionJob]:
    """Persist one forward stage transition and its timestamp."""
    stage_value = stage.value if isinstance(stage, IngestionStage) else stage
    if stage_value not in _STAGE_INDEX:
        raise ValueError(f"Unknown ingestion stage: {stage_value}")
    job = await get_ingestion_job(db, job_id)
    if job is None or job.status in TERMINAL_JOB_STATUSES:
        return None
    current = job.stage or IngestionStage.UPLOAD.value
    if _STAGE_INDEX[stage_value] < _STAGE_INDEX[current]:
        raise ValueError(f"Ingestion stages cannot move backwards: {current} -> {stage_value}")
    if _STAGE_INDEX[stage_value] > _STAGE_INDEX[current] + 1:
        raise ValueError(f"Ingestion stages cannot skip: {current} -> {stage_value}")
    job.stage = stage_value
    job.status = IngestionStatus.RUNNING.value
    job.updated_at = _now()
    await db.commit()
    return job


async def begin_attempt(db: AsyncSession, job_id: uuid.UUID) -> Optional[IngestionJob]:
    locked_result = await db.execute(
        select(IngestionJob).where(IngestionJob.id == job_id).with_for_update()
    )
    job = locked_result.scalar_one_or_none()
    if job is None or job.status in TERMINAL_JOB_STATUSES:
        return None
    # A duplicate delivery must not run concurrently with the active attempt.
    # Reconciliation will requeue it if the active worker becomes stale.
    if job.status == IngestionStatus.RUNNING.value:
        return None
    if job.status == IngestionStatus.RETRYING.value:
        # A retry starts a fresh persisted attempt from the upload boundary.
        job.stage = IngestionStage.UPLOAD.value
    document_result = await db.execute(
        select(Document)
        .where(Document.id == job.document_id)
        .with_for_update()
    )
    document = document_result.scalar_one_or_none()
    version = await db.get(DocumentVersion, job.version_id)
    if document is None or version is None:
        if job.status not in TERMINAL_JOB_STATUSES:
            job.status = IngestionStatus.FAILED.value
            job.error_code = "MISSING_INGESTION_ENTITY"
            job.error_message = "Document or version no longer exists"
            job.completed_at = _now()
            job.updated_at = _now()
            await db.commit()
        return None
    if document.status in {DocumentStatus.DELETING.value, DocumentStatus.DELETED.value}:
        job.status = IngestionStatus.CANCELLED.value
        job.error_code = "DOCUMENT_DELETING"
        job.error_message = "Document is no longer available for ingestion"
        job.completed_at = _now()
        job.updated_at = _now()
        await db.commit()
        return None
    if version.status in {VersionStatus.DELETING.value, VersionStatus.DELETED.value}:
        job.status = IngestionStatus.CANCELLED.value
        job.error_code = "VERSION_DELETING"
        job.error_message = "Document version is no longer available for ingestion"
        job.completed_at = _now()
        job.updated_at = _now()
        await db.commit()
        return None
    document.status = DocumentStatus.PROCESSING.value
    document.updated_at = _now()
    version.status = VersionStatus.PROCESSING.value
    version.updated_at = _now()
    job.status = IngestionStatus.RUNNING.value
    job.attempt_count = (job.attempt_count or 0) + 1
    job.started_at = _now()
    job.error_code = None
    job.error_message = None
    job.updated_at = _now()
    await db.commit()
    return job


async def mark_retryable_failure(
    db: AsyncSession, job_id: uuid.UUID, error_code: str, error_message: str
) -> Optional[IngestionJob]:
    job = await get_ingestion_job(db, job_id)
    if job is None or job.status in TERMINAL_JOB_STATUSES:
        return None
    version = await db.get(DocumentVersion, job.version_id)
    job.status = IngestionStatus.RETRYING.value
    job.error_code = error_code
    job.error_message = error_message
    job.updated_at = _now()
    document_result = await db.execute(
        select(Document)
        .where(Document.id == job.document_id)
        .with_for_update()
    )
    document = document_result.scalar_one_or_none()
    if (
        version is not None
        and version.storage_key
        and document is not None
        and document.status not in {DocumentStatus.DELETING.value, DocumentStatus.DELETED.value}
        and version.status not in {VersionStatus.DELETING.value, VersionStatus.DELETED.value}
    ):
        await create_outbox_event(db, "DOCUMENT_INGESTION_REQUESTED", job.id, {
            "ingestion_job_id": str(job.id),
            "document_id": str(job.document_id),
            "version_id": str(job.version_id),
            "storage_key": version.storage_key,
        })
    await db.commit()
    return job


async def mark_non_retryable_failure(
    db: AsyncSession, job_id: uuid.UUID, error_code: str, error_message: str
) -> Optional[IngestionJob]:
    job = await get_ingestion_job(db, job_id)
    if job is None or job.status in TERMINAL_JOB_STATUSES:
        return None
    version = await db.get(DocumentVersion, job.version_id)
    document_result = await db.execute(
        select(Document)
        .where(Document.id == job.document_id)
        .with_for_update()
    )
    document = document_result.scalar_one_or_none()
    if version is not None and version.status not in {
        VersionStatus.DELETING.value,
        VersionStatus.DELETED.value,
    }:
        version.status = VersionStatus.FAILED.value
        version.updated_at = _now()
    job.status = IngestionStatus.FAILED.value
    job.error_code = error_code
    job.error_message = error_message
    job.completed_at = _now()
    job.updated_at = _now()
    # A deleted/deleting document is never resurrected or rewritten.
    if document is not None and document.status in {
        DocumentStatus.DELETING.value,
        DocumentStatus.DELETED.value,
    }:
        job.status = IngestionStatus.CANCELLED.value
    await db.commit()
    return job


async def finalize_attempt(db: AsyncSession, job_id: uuid.UUID) -> bool:
    """Promote a version only after all finalization eligibility checks pass."""
    job = await get_ingestion_job(db, job_id)
    if job is None or job.status != IngestionStatus.RUNNING.value or job.stage != IngestionStage.FINALIZE.value:
        return False
    document_result = await db.execute(
        select(Document)
        .where(Document.id == job.document_id)
        .with_for_update()
    )
    document = document_result.scalar_one_or_none()
    version = await db.get(DocumentVersion, job.version_id)
    eligible = (
        document is not None
        and version is not None
        and version.document_id == document.id
        and version.status == VersionStatus.PROCESSING.value
        and document.status not in {
            DocumentStatus.DELETING.value,
            DocumentStatus.DELETED.value,
        }
        and document.deleted_at is None
        and (document.current_version_id is None or document.current_version_id == version.id)
    )
    if not eligible:
        await mark_non_retryable_failure(
            db, job_id, "FINALIZE_NOT_ELIGIBLE", "Document or version is no longer eligible for finalization"
        )
        return False
    version.status = VersionStatus.READY.value
    version.updated_at = _now()
    document.current_version_id = version.id
    document.status = DocumentStatus.READY.value
    document.updated_at = _now()
    job.status = IngestionStatus.COMPLETED.value
    job.completed_at = _now()
    job.updated_at = _now()
    await db.commit()
    return True


async def reconcile_stale_jobs(
    db: AsyncSession, *, stale_after: timedelta = timedelta(minutes=15), limit: int = 100
) -> int:
    """Requeue stale PENDING/RUNNING jobs through the durable outbox."""
    cutoff = _now() - stale_after
    result = await db.execute(
        select(IngestionJob).where(
            IngestionJob.status.in_([
            IngestionStatus.PENDING.value,
            IngestionStatus.RUNNING.value,
            IngestionStatus.RETRYING.value,
            ]),
            IngestionJob.updated_at < cutoff,
        ).order_by(IngestionJob.updated_at.asc()).limit(limit)
    )
    reconciled = 0
    for job in result.scalars().all():
        document = await db.get(Document, job.document_id)
        version = await db.get(DocumentVersion, job.version_id)
        if document is None or version is None:
            job.attempt_count = (job.attempt_count or 0) + 1
            job.status = IngestionStatus.FAILED.value
            job.error_code = "RECONCILIATION_MISSING_ENTITY"
            job.error_message = "Document or version no longer exists"
            job.completed_at = _now()
            job.updated_at = _now()
            await db.commit()
            reconciled += 1
            continue
        if document.status in {DocumentStatus.DELETING.value, DocumentStatus.DELETED.value} or version.status in {
            VersionStatus.DELETING.value, VersionStatus.DELETED.value,
        }:
            job.attempt_count = (job.attempt_count or 0) + 1
            job.status = IngestionStatus.CANCELLED.value
            job.error_code = "RECONCILIATION_DELETED_ENTITY"
            job.error_message = "Stale job references a deleting or deleted entity"
            job.completed_at = _now()
            job.updated_at = _now()
            await db.commit()
            reconciled += 1
            continue
        job.status = IngestionStatus.RETRYING.value
        job.attempt_count = (job.attempt_count or 0) + 1
        job.error_code = "STALE_JOB_RECONCILED"
        job.error_message = "Job was stale and requeued by reconciliation"
        job.updated_at = _now()
        await create_outbox_event(db, "DOCUMENT_INGESTION_REQUESTED", job.id, {
            "ingestion_job_id": str(job.id),
            "document_id": str(job.document_id),
            "version_id": str(job.version_id),
            "storage_key": version.storage_key,
        })
        await db.commit()
        reconciled += 1
    return reconciled
