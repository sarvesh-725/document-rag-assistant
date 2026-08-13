import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.routers.ingestion as ingestion


def user():
    return SimpleNamespace(id=uuid.uuid4())


@pytest.mark.asyncio
async def test_ingestion_status_returns_all_progress_fields(monkeypatch):
    current_user = user()
    job = SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        status="RUNNING",
        stage="EMBED",
        attempt_count=1,
        error_code=None,
        error_message=None,
    )
    lookup = AsyncMock(return_value=job)
    monkeypatch.setattr(ingestion, "get_ingestion_job_for_user", lookup)
    db = AsyncMock()

    response = await ingestion.get_ingestion_job_status(str(job.id), current_user, db)

    assert response == {
        "job_id": str(job.id),
        "document_id": str(job.document_id),
        "version_id": str(job.version_id),
        "status": "RUNNING",
        "stage": "EMBED",
        "attempt_count": 1,
        "error_code": None,
        "error_message": None,
    }
    lookup.assert_awaited_once_with(db, job.id, current_user.id)


@pytest.mark.asyncio
async def test_ingestion_status_hides_non_owned_jobs(monkeypatch):
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(ingestion, "get_ingestion_job_for_user", lookup)

    with pytest.raises(HTTPException) as error:
        await ingestion.get_ingestion_job_status(
            str(uuid.uuid4()), user(), AsyncMock()
        )

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_ingestion_status_rejects_invalid_job_id():
    with pytest.raises(HTTPException) as error:
        await ingestion.get_ingestion_job_status("not-a-uuid", user(), AsyncMock())

    assert error.value.status_code == 400
