"""
Taskiq background worker — Phase 1 stub.

The document processing task is stubbed pending Phase 2 rewrite
with Document/DocumentVersion/IngestionJob models.
"""

import logging

import os
from app.broker import broker
from app.services.storage import LocalStorageService

logger = logging.getLogger("taskiq_worker")
logging.basicConfig(level=logging.INFO)

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)


@broker.task(task_name="tasks.process_document")
async def async_process_document_task(
    document_id: str,
    version_id: str,
    storage_key: str,
    ingestion_job_id: str,
):
    """
    STUB: Document processing task pending Phase 2/4 rewrite.

    Will:
    1. Update IngestionJob status to RUNNING
    2. Read file from StorageService via storage_key
    3. Parse with Unstructured API
    ...
    """
    logger.warning(
        f"STUB: process_document called with document_id={document_id}, "
        f"version_id={version_id}, storage_key={storage_key}, "
        f"ingestion_job_id={ingestion_job_id}. "
        "Pending full worker implementation."
    )
    
    # Example logic demonstrating storage usage:
    # 
    # user_id = storage_key.split('/')[1] 
    # content = await storage_service.read(storage_key, uuid.UUID(user_id))
    # 
    # * File deliberately kept intact on both success and failure 
    #   per Phase 4 guidelines until retention policy applies.
    
    raise NotImplementedError("Document processing task pending Phase 5 rewrite.")
