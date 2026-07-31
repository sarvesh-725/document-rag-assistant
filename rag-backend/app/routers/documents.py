import os
import re
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.database.models import User, ChatSession, SessionFile
from app.auth.security import get_current_user
from app.tasks.celery_worker import async_process_document_task

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

CELERY_UPLOAD_DIR = os.getenv("CELERY_UPLOAD_DIR", "UPLOADS")


class BindFileRequest(BaseModel):
    session_id: str
    filename: str


class UnbindFileRequest(BaseModel):
    session_id: str
    filename: str


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts a document file, saves it to a temporary path,
    inserts a 'processing' state record into SessionFile table,
    and dispatches ingestion worker task.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No upload file sent"
        )
    filename = file.filename
    content_type = file.content_type

    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID string cannot be empty or whitespace."
        )

    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"
    is_txt = filename.lower().endswith((".txt", ".text")) or content_type == "text/plain"

    if not (is_pdf or is_txt):
        logger.warning(
            f"Rejected upload: file '{filename}' with content-type '{content_type}' is not a PDF or text file."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF and TXT (text) files are accepted.",
        )

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = session_result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    os.makedirs(CELERY_UPLOAD_DIR, exist_ok=True)

    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    temp_filename = f"temp_{current_user.id}_{session_id}_{safe_filename}"
    temp_file_path = os.path.join(CELERY_UPLOAD_DIR, temp_filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_query = await db.execute(
            select(SessionFile).where(
                SessionFile.session_id == session_id,
                SessionFile.filename == filename
            )
        )
        session_file = file_query.scalars().first()

        if not session_file:
            session_file = SessionFile(
                session_id=session_id,
                filename=filename,
                file_hash="processing",
                is_committed=False,
                just_uploaded=False,  # Uncommitted and not yet ready for searches
                status="processing"
            )
            db.add(session_file)
        else:
            session_file.file_hash = "processing"
            session_file.is_committed = False
            session_file.just_uploaded = False
            session_file.status = "processing"
            db.add(session_file)

        await db.commit()

        task = async_process_document_task.delay(
            temp_file_path,
            filename,
            current_user.id,
            session_id
        )

        return {
            "status": "processing",
            "message": "Ingestion started in background",
            "task_id": task.id
        }

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logger.exception(f"Unexpected error initializing upload of '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during ingestion startup: {str(e)}",
        )


@router.get("/global")
async def get_global_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns an array of all unique file rows ever indexed under the authenticated user's ID
    that have a status of "completed".
    """
    stmt = select(SessionFile).join(ChatSession).where(
        ChatSession.user_id == current_user.id,
        SessionFile.status == "completed"
    )
    res = await db.execute(stmt)
    files = res.scalars().all()

    unique_files = {}
    for f in files:
        if f.filename not in unique_files:
            unique_files[f.filename] = {
                "filename": f.filename,
                "file_hash": f.file_hash,
                "status": f.status,
                "is_committed": f.is_committed,
                "just_uploaded": f.just_uploaded
            }

    return list(unique_files.values())


@router.post("/bind")
async def bind_document(
    request: BindFileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Binds a completed historical document to the active chat session.
    """
    session_id = request.session_id.strip()
    filename = request.filename.strip()

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = session_result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    stmt = select(SessionFile).join(ChatSession).where(
        ChatSession.user_id == current_user.id,
        SessionFile.filename == filename,
        SessionFile.status == "completed"
    )
    file_result = await db.execute(stmt)
    existing_file = file_result.scalars().first()
    if not existing_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completed file not found in user history"
        )

    stmt = select(SessionFile).where(
        SessionFile.session_id == session_id,
        SessionFile.filename == filename
    )
    assoc_result = await db.execute(stmt)
    assoc = assoc_result.scalars().first()

    if not assoc:
        assoc = SessionFile(
            session_id=session_id,
            filename=filename,
            file_hash=existing_file.file_hash,
            is_committed=True,
            just_uploaded=False,
            status="completed"
        )
        db.add(assoc)
    else:
        assoc.is_committed = True
        assoc.just_uploaded = False
        assoc.status = "completed"
        db.add(assoc)

    await db.commit()

    return {
        "status": "success",
        "message": f"Successfully bound file '{filename}' to session '{session_id}'",
        "file": {
            "filename": assoc.filename,
            "file_hash": assoc.file_hash,
            "is_committed": assoc.is_committed,
            "just_uploaded": assoc.just_uploaded,
            "status": assoc.status
        }
    }


@router.post("/unbind")
async def unbind_document(
    request: UnbindFileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Unbinds/unselects a document from the active chat session.
    """
    session_id = request.session_id.strip()
    filename = request.filename.strip()

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = session_result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    stmt = select(SessionFile).where(
        SessionFile.session_id == session_id,
        SessionFile.filename == filename
    )
    assoc_result = await db.execute(stmt)
    assoc = assoc_result.scalars().first()
    if assoc:
        assoc.is_committed = False
        assoc.just_uploaded = False
        db.add(assoc)
        await db.commit()

    return {
        "status": "success",
        "message": f"Successfully unbound file '{filename}' from session '{session_id}'"
    }


@router.delete("/sessions/{session_id}/files/{filename}")
async def delete_document(
    session_id: str,
    filename: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes the document from all sessions, Qdrant vectors, and local storage.
    """
    session_id = session_id.strip()
    filename = filename.strip()

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = session_result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    stmt = select(SessionFile).join(ChatSession).where(
        ChatSession.user_id == current_user.id,
        SessionFile.filename == filename
    )
    files_res = await db.execute(stmt)
    files_to_delete = files_res.scalars().all()

    for f in files_to_delete:
        await db.delete(f)
    await db.commit()

    from app.services.rag_engine import delete_file_from_db
    try:
        await delete_file_from_db(str(current_user.id), filename)
    except Exception as q_err:
        logger.error(f"Error purging Qdrant vectors for '{filename}': {q_err}")

    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    temp_filename = f"temp_{current_user.id}_{session_id}_{safe_filename}"
    temp_file_path = os.path.join(CELERY_UPLOAD_DIR, temp_filename)
    if os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
            logger.info(f"Deleted local cached file {temp_file_path}")
        except Exception as file_err:
            logger.error(f"Failed to delete local file {temp_file_path}: {file_err}")

    return {
        "status": "success",
        "message": f"Successfully deleted and purged file '{filename}' from system indexes."
    }
