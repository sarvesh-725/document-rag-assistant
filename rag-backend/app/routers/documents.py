"""
Document routes — Phase 1 stub.

All routes are stubbed pending Phase 2 rewrite with Document/DocumentVersion models.
The old SessionFile-based logic has been removed entirely.
"""

import logging
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status

from app.database.connection import get_db
from app.database.models import User, Document, DocumentVersion
from app.auth.security import get_current_user

import uuid
import os
import shutil
import hashlib
from typing import List, Optional

from app.database.repositories import (
    create_document, create_document_version, create_ingestion_job,
    create_outbox_event, get_session_by_id, list_user_documents, soft_delete_document
)
from app.database.enums import IngestionStatus, IngestionStage
from app.services.storage import LocalStorageService

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)


ALLOWED_TYPES = {"application/pdf", "text/plain"}
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Uploads a new document for the user. Creates a new Document entry, 
    a DocumentVersion, and schedules an IngestionJob.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS or file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF and plain-text files are accepted")
    if session_id is not None:
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid session ID format") from exc
        if not await get_session_by_id(db, session_uuid, current_user.id):
            raise HTTPException(status_code=404, detail="Session not found or unauthorized")
        # Session validation is read-only; begin the upload transaction only
        # after the source has been durably written below.
        await db.rollback()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File content is empty")
    file_hash = hashlib.sha256(content).hexdigest()
    document_id, version_id, ingestion_job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    storage_key = f"documents/{current_user.id}/{document_id}/{version_id}/original"
    await storage_service.save(storage_key, content, current_user.id)
    try:
        await create_document(db, current_user.id, file.filename, document_id=document_id)
        await create_document_version(db, document_id, file_hash, storage_key, "v1", "v1", "default", version_id)
        await create_ingestion_job(db, document_id, version_id, ingestion_job_id, IngestionStatus.PENDING.value, IngestionStage.UPLOAD.value)
        await create_outbox_event(db, "DOCUMENT_INGESTION_REQUESTED", ingestion_job_id, {
            "ingestion_job_id": str(ingestion_job_id), "document_id": str(document_id),
            "version_id": str(version_id), "storage_key": storage_key,
            "user_id": str(current_user.id),
        })
        await db.commit()
    except Exception:
        await db.rollback()
        await storage_service.delete(storage_key, current_user.id)
        raise
    return {"document_id": str(document_id), "version_id": str(version_id), "job_id": str(ingestion_job_id), "status": IngestionStatus.PENDING.value}


@router.get("/global")
async def get_global_documents(
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Returns a list of all documents belonging to the user.
    """
    docs = await list_user_documents(db, current_user.id)
    return [
        {
            "document_id": str(d.id),
            "display_name": d.display_name,
            "original_filename": d.original_filename,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "status": d.status,
            "duplicate_index": d.duplicate_index,
        }
        for d in docs
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Soft-deletes the document by setting its status and deleted_at.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")
        
    success = await soft_delete_document(db, doc_uuid, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    await db.commit()
    return {"status": "success", "message": "Document deleted."}
