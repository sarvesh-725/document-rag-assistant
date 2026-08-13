"""Authenticated ingestion-job status API."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.security import get_current_user
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import get_ingestion_job_for_user

router = APIRouter(prefix="/api/v1/ingestion-jobs", tags=["ingestion"])


@router.get("/{job_id}")
async def get_ingestion_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    try:
        ingestion_job_id = uuid.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ingestion job ID format.",
        ) from exc

    job = await get_ingestion_job_for_user(db, ingestion_job_id, current_user.id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found.",
        )

    return {
        "job_id": str(job.id),
        "document_id": str(job.document_id),
        "version_id": str(job.version_id),
        "status": job.status,
        "stage": job.stage,
        "attempt_count": job.attempt_count,
        "error_code": job.error_code,
        "error_message": job.error_message,
    }
