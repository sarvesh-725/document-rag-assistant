"""Authenticated ingestion-job status API."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.security import get_current_user
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import get_ingestion_job_for_user
from app.errors import ErrorCode, api_error

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
        raise api_error(ErrorCode.INGESTION_FAILED, "Invalid ingestion job ID format.", 400) from exc

    job = await get_ingestion_job_for_user(db, ingestion_job_id, current_user.id)
    if job is None:
        raise api_error(ErrorCode.INGESTION_FAILED, "Ingestion job not found.", 404)

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
