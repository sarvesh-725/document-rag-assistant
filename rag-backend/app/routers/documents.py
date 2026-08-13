"""Global document-library routes."""

import hashlib
import logging
import os
import unicodedata
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.auth.security import get_current_user
from app.database.connection import get_db
from app.database.enums import IngestionStage, IngestionStatus, VersionStatus
from app.database.models import User
from app.database.repositories import (
    create_document,
    create_document_version,
    create_ingestion_job,
    create_outbox_event,
    list_user_documents,
    soft_delete_document,
)
from app.services.storage import LocalStorageService

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UPLOADS"))
storage_service = LocalStorageService(STORAGE_ROOT)

MIME_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))


def normalize_filename(filename: str) -> str:
    """Normalize display metadata without allowing a client path to persist."""
    normalized = unicodedata.normalize("NFC", filename).strip()
    normalized = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if not normalized or normalized in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")
    if any(ord(character) < 32 for character in normalized):
        raise HTTPException(status_code=400, detail="Filename contains invalid control characters")
    return normalized


def validate_upload_content(
    filename: str, content_type: Optional[str], content: bytes
) -> str:
    extension = os.path.splitext(filename)[1].lower()
    expected_mime = MIME_BY_EXTENSION.get(extension)
    if expected_mime is None or content_type != expected_mime:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and plain-text files with matching MIME types are accepted",
        )
    if content and extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file is not a valid PDF",
        )
    if content and extension == ".txt" and b"\x00" in content:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file is not valid plain text",
        )
    return extension


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Create a global document and enqueue ingestion without doing RAG work."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Kept temporarily for frontend compatibility. It never participates in
    # authorization and no session-document association is created.
    del session_id
    normalized_filename = normalize_filename(file.filename)
    validate_upload_content(normalized_filename, file.content_type, b"")

    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="File content is empty")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File is too large",
        )
    validate_upload_content(normalized_filename, file.content_type, content)

    file_hash = hashlib.sha256(content).hexdigest()
    document_id, version_id, ingestion_job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    storage_key = f"documents/{current_user.id}/{document_id}/{version_id}/original"
    await storage_service.save(storage_key, content, current_user.id)
    try:
        document = await create_document(
            db, current_user.id, normalized_filename, document_id=document_id
        )
        await create_document_version(
            db,
            document_id,
            file_hash,
            storage_key,
            "v1",
            "v1",
            "default",
            version_id,
        )
        await create_ingestion_job(
            db,
            document_id,
            version_id,
            ingestion_job_id,
            IngestionStatus.PENDING.value,
            IngestionStage.UPLOAD.value,
        )
        await create_outbox_event(
            db,
            "DOCUMENT_INGESTION_REQUESTED",
            ingestion_job_id,
            {
                "ingestion_job_id": str(ingestion_job_id),
                "document_id": str(document_id),
                "version_id": str(version_id),
                "storage_key": storage_key,
                "user_id": str(current_user.id),
            },
        )
        await db.commit()
    except Exception:
        await db.rollback()
        await storage_service.delete(storage_key, current_user.id)
        raise

    return {
        "document_id": str(document_id),
        "version_id": str(version_id),
        "job_id": str(ingestion_job_id),
        "display_name": document.display_name,
        "status": VersionStatus.PROCESSING.value,
    }


@router.get("/global")
async def get_global_documents(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Return all live documents belonging to the authenticated user."""
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
    db=Depends(get_db),
):
    """Logically delete an owned document and schedule physical cleanup."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID format.") from exc

    success = await soft_delete_document(db, doc_uuid, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    await create_outbox_event(
        db,
        "DOCUMENT_CLEANUP_REQUESTED",
        doc_uuid,
        {"document_id": str(doc_uuid), "user_id": str(current_user.id)},
    )
    await db.commit()
    return {"status": "success", "message": "Document deleted."}
