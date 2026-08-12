"""
Taskiq background worker — Phase 1 stub.

The document processing task is stubbed pending Phase 2 rewrite
with Document/DocumentVersion/IngestionJob models.
"""

import logging

from app.broker import broker

logger = logging.getLogger("taskiq_worker")
logging.basicConfig(level=logging.INFO)


@broker.task(task_name="tasks.process_document")
async def async_process_document_task(
    document_id: str,
    version_id: str,
    job_id: str,
):
    """
    STUB: Document processing task pending Phase 2 rewrite.

    Will:
    1. Update IngestionJob status to RUNNING
    2. Read file from storage_key
    3. Parse with Unstructured API
    4. Chunk with parent/child strategy
    5. Embed with Gemini
    6. Index in Qdrant
    7. Update DocumentVersion status to READY
    8. Update Document.current_version_id
    9. Update IngestionJob status to COMPLETED
    """
    logger.warning(
        f"STUB: process_document called with document_id={document_id}, "
        f"version_id={version_id}, job_id={job_id}. "
        "Pending Phase 2 implementation."
    )
    raise NotImplementedError("Document processing task pending Phase 2 rewrite.")
