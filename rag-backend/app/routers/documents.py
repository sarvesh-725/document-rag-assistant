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

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    STUB: Upload endpoint pending Phase 2 rewrite.
    Will create Document + DocumentVersion + IngestionJob.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document upload pending Phase 2 rewrite with Document/DocumentVersion/IngestionJob models."
    )


@router.get("/global")
async def get_global_documents(
    current_user: User = Depends(get_current_user),
):
    """
    STUB: Global document list pending Phase 2 rewrite.
    Will query Document table filtered by user_id and status != DELETED.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Global documents endpoint pending Phase 2 rewrite."
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    STUB: Document deletion pending Phase 2 rewrite.
    Will set Document.status = DELETING, then async cleanup.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document deletion pending Phase 2 rewrite."
    )
