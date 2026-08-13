"""
Taskiq background worker — Phase 1 stub.

The document processing task is stubbed pending Phase 2 rewrite
with Document/DocumentVersion/IngestionJob models.
"""

import logging

import os
import uuid
from app.broker import broker
from app.services.storage import LocalStorageService
from app.database.connection import AsyncSessionLocal
from app.services.outbox import publish_pending_outbox_events
from app.database.models import Document
from app.services.ingestion_state_machine import (
    NonRetryableIngestionError,
    RetryableIngestionError,
    begin_attempt,
    finalize_attempt,
    mark_non_retryable_failure,
    mark_retryable_failure,
    transition_stage,
    reconcile_stale_jobs,
)
from app.database.enums import IngestionStage

logger = logging.getLogger("taskiq_worker")
logging.basicConfig(level=logging.INFO)

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)


@broker.task(task_name="tasks.publish_outbox_events")
async def publish_outbox_events_task() -> int:
    """Retry unpublished domain events; failed rows remain eligible next run."""
    async with AsyncSessionLocal() as db:
        return await publish_pending_outbox_events(db)


@broker.task(task_name="tasks.reconcile_ingestion_jobs")
async def reconcile_ingestion_jobs_task() -> int:
    """Periodic safety net for jobs left in PENDING/RUNNING."""
    async with AsyncSessionLocal() as db:
        return await reconcile_stale_jobs(db)


@broker.task(task_name="tasks.process_document")
async def async_process_document_task(
    document_id: str,
    version_id: str,
    storage_key: str,
    ingestion_job_id: str,
):
    """
    Validate and accept a durable task payload. Processing consumes storage_key,
    never a local upload path.

    Will:
    1. Update IngestionJob status to RUNNING
    2. Read file from StorageService via storage_key
    3. Parse with Unstructured API
    ...
    """
    logger.info("Received durable ingestion job %s", ingestion_job_id)
    if not all((document_id, version_id, ingestion_job_id, storage_key)):
        raise ValueError("Durable ingestion payload is missing a required identifier")
    if not storage_key.startswith("documents/"):
        raise ValueError("storage_key must be a StorageService key, never a local path")

    job_uuid = uuid.UUID(ingestion_job_id)
    document_uuid = uuid.UUID(document_id)
    version_uuid = uuid.UUID(version_id)
    async with AsyncSessionLocal() as db:
        job = await begin_attempt(db, job_uuid)
        if job is None:
            return
        try:
            document = await db.get(Document, document_uuid)
            if document is None or document.id != job.document_id or job.version_id != version_uuid:
                raise NonRetryableIngestionError(
                    "Ingestion payload does not match the persisted job",
                    "INGESTION_PAYLOAD_MISMATCH",
                )
            parts = storage_key.split("/")
            if len(parts) != 5:
                raise NonRetryableIngestionError("Invalid storage key", "INVALID_STORAGE_KEY")
            if parts[2] != document_id or parts[3] != version_id or parts[4] != "original":
                raise NonRetryableIngestionError(
                    "Storage key does not match the persisted ingestion identifiers",
                    "STORAGE_KEY_MISMATCH",
                )
            user_uuid = uuid.UUID(parts[1])
            content = await storage_service.read(storage_key, user_uuid)
            if not content:
                raise NonRetryableIngestionError("Stored source is empty", "EMPTY_SOURCE")
            await transition_stage(db, job_uuid, IngestionStage.PARSE)
            from app.services.chunking import parse_parent_child_chunks, PARSER_VERSION, CHUNKING_VERSION
            parsed = parse_parent_child_chunks(content, document.original_filename, version_uuid, document_uuid)
            await transition_stage(db, job_uuid, IngestionStage.CHUNK)
            from app.database.repositories import replace_version_parents
            await replace_version_parents(db, version_uuid, parsed.parents)
            await db.commit()
            await transition_stage(db, job_uuid, IngestionStage.EMBED)
            await transition_stage(db, job_uuid, IngestionStage.INDEX)
            from app.services.rag_engine import process_and_store_document
            indexed_chunks = await process_and_store_document(
                content,
                user_id=str(user_uuid),
                document_id=document_uuid,
                version_id=version_uuid,
                filename=document.original_filename,
                parsed=parsed,
            )
            if not indexed_chunks:
                raise NonRetryableIngestionError("No indexable content was produced", "EMPTY_INDEX")
            await transition_stage(db, job_uuid, IngestionStage.FINALIZE)
            await finalize_attempt(db, job_uuid)
        except RetryableIngestionError as exc:
            await mark_retryable_failure(db, job_uuid, exc.code, str(exc))
        except (TimeoutError, ConnectionError) as exc:
            await mark_retryable_failure(db, job_uuid, "TRANSIENT_INGESTION_ERROR", str(exc))
        except NonRetryableIngestionError as exc:
            await mark_non_retryable_failure(db, job_uuid, exc.code, str(exc))
        except Exception as exc:
            await mark_non_retryable_failure(db, job_uuid, "INGESTION_FAILED", str(exc))
