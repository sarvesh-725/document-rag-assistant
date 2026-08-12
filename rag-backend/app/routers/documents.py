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
from app.tasks.taskiq_worker import async_process_document_task

logger = logging.getLogger("documents_router")
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

CELERY_UPLOAD_DIR = os.getenv("CELERY_UPLOAD_DIR", "UPLOADS")



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
    # Validates that a file was actually provided in the request
    # Needed because the endpoint expects a file for upload, and proceeding without one would cause errors later.
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No upload file sent"
        )
    filename = file.filename
    content_type = file.content_type

    # Validates that the session ID is present and not just empty spaces
    # Needed because files must be associated with a specific, valid chat session.
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID string cannot be empty or whitespace."
        )

    # Checks if the uploaded file's extension or content type matches accepted formats (PDF or Text)
    # Needed to ensure only supported document types are processed by the RAG system, preventing ingestion errors.
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

    # Verifies that the chat session exists and belongs to the currently authenticated user
    # Needed for security and access control, ensuring users can only upload files to their own active sessions.
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

    # Ensures the directory for storing uploaded files temporarily before Celery processing exists
    os.makedirs(CELERY_UPLOAD_DIR, exist_ok=True)

    # Generates a safe and unique temporary filename using user ID and session ID
    # Needed to avoid filename collisions if multiple users upload files with the same name, and prevents path traversal attacks.
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    temp_filename = f"temp_{current_user.id}_{session_id}_{safe_filename}"
    temp_file_path = os.path.join(CELERY_UPLOAD_DIR, temp_filename)

    try:
        # Saves the uploaded file content from the API request to the local temporary file path
        # Needed so the background Celery worker can access the physical file for ingestion.
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Checks if this exact file is already associated with the current session
        file_query = await db.execute(
            select(SessionFile).where(
                SessionFile.session_id == session_id,
                SessionFile.filename == filename
            )
        )
        session_file = file_query.scalars().first()

        # Updates or creates the database record for this file in the current session, setting its status to 'processing'
        # Needed to track the file's ingestion state in the UI before the background worker finishes processing it.
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

        # Dispatches a background Taskiq task to process and ingest the document asynchronously
        # Needed to prevent the API request from blocking while a potentially large document is being vectorized.
        task = await async_process_document_task.kiq(
            temp_file_path,
            filename,
            current_user.id,
            session_id
        )

        return {
            "status": "processing",
            "message": "Ingestion started in background",
            "task_id": task.task_id
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
    # Queries the database for all files associated with any of the user's sessions that have finished processing
    # Needed to fetch the user's entire document history so they can re-use previously uploaded files.
    stmt = select(SessionFile).join(ChatSession).where(
        ChatSession.user_id == current_user.id,
        SessionFile.status == "completed"
    )
    res = await db.execute(stmt)
    files = res.scalars().all()

    # Deduplicates the file list based on filename, keeping the latest status
    # Needed because the same file might be bound to multiple sessions, but we only want to show it once in the global list.
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

    # Verifies that the target chat session exists and belongs to the user
    # Needed for security, confirming the user has rights to delete from this session context.
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

    # Finds all database records of this file across all of the user's sessions and deletes them
    # Needed to completely remove the file's metadata from the user's history and all chats.
    stmt = select(SessionFile).join(ChatSession).where(
        ChatSession.user_id == current_user.id,
        SessionFile.filename == filename
    )
    files_res = await db.execute(stmt)
    files_to_delete = files_res.scalars().all()

    for f in files_to_delete:
        await db.delete(f)
    await db.commit()

    # Calls an external service function to remove the document's embeddings from the vector database (Qdrant)
    # Needed because deleting the SQL records doesn't remove the actual vector data used for search/RAG.
    from app.services.rag_engine import delete_file_from_db
    try:
        await delete_file_from_db(str(current_user.id), filename)
    except Exception as q_err:
        logger.error(f"Error purging Qdrant vectors for '{filename}': {q_err}")

    # Attempts to delete the temporary physical file from the local file system if it still exists
    # Needed for cleanup to prevent disk space exhaustion from orphaned temporary files.
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
