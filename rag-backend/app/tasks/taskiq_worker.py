"""
Taskiq background worker — Phase 1 stub.

The document processing task is stubbed pending Phase 2 rewrite
with Document/DocumentVersion/IngestionJob models.
"""

import logging

import os
from app.broker import broker
from app.services.storage import LocalStorageService
from app.database.connection import AsyncSessionLocal
from app.services.outbox import publish_pending_outbox_events

logger = logging.getLogger("taskiq_worker")
logging.basicConfig(level=logging.INFO)

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)


@broker.task(task_name="tasks.publish_outbox_events")
async def publish_outbox_events_task() -> int:
    """Retry unpublished domain events; failed rows remain eligible next run."""
    async with AsyncSessionLocal() as db:
        return await publish_pending_outbox_events(db)


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
