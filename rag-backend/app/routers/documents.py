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
from typing import List
from datetime import datetime

from app.database.repositories import (
    create_document, create_document_version, create_ingestion_job,
    list_user_documents, soft_delete_document
)
from app.services.storage import LocalStorageService

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Uploads a new document for the user. Creates a new Document entry, 
    a DocumentVersion, and schedules an IngestionJob.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # 1. Create Document (duplicate naming handled inside repository)
    doc = await create_document(db, current_user.id, file.filename)

    # 2. Pre-generate Version ID and construct strictly formatted storage_key
    version_id = uuid.uuid4()
    storage_key = f"documents/{current_user.id}/{doc.id}/{version_id}/original"

    # 3. Save file using the StorageService
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    
    await storage_service.save(storage_key, content, current_user.id)

    # 4. Create DocumentVersion explicitly linking the storage_key
    version = await create_document_version(
        db=db,
        document_id=doc.id,
        content_hash=file_hash,
        storage_key=storage_key,
        parser_version="v1",
        chunking_version="v1",
        embedding_profile="default",
        version_id=version_id,
    )

    # 4. Create IngestionJob
    job = await create_ingestion_job(db, doc.id, version.id)
    await db.commit()
    
    return {
        "status": "success",
        "document_id": str(doc.id),
        "display_name": doc.display_name,
        "original_filename": doc.original_filename
    }


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
