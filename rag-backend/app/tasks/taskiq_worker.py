import os
import logging
import hashlib
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from taskiq import TaskiqDepends as Depends

from app.broker import broker
from app.database.connection import get_db
from app.database.models import SessionFile
from app.services import rag_engine

logger = logging.getLogger("taskiq_worker")
logging.basicConfig(level=logging.INFO)

@broker.task(task_name="tasks.process_document")
async def async_process_document_task(
    file_path: str,
    filename: str,
    user_id: int,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Taskiq background worker task for processing uploaded documents.
    """
    logger.info(f"Task received: processing '{filename}'")

    if not os.path.exists(file_path):
        logger.error(f"Target ingestion file not found at path: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    try:
        logger.info(f"Starting background RAG ingestion for '{filename}' (User: {user_id}, Session: {session_id})")
        num_chunks = await rag_engine.process_and_store_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=str(user_id)
        )
        logger.info(f"Ingested {num_chunks} chunks successfully into Qdrant Cloud.")

        result = await db.execute(
            select(SessionFile).where(
                SessionFile.session_id == session_id,
                SessionFile.filename == filename
            )
        )
        session_file = result.scalars().first()

        if not session_file:
            session_file = SessionFile(
                session_id=session_id,
                filename=filename,
                file_hash=file_hash,
                is_committed=False,
                just_uploaded=True,
                status="completed"
            )
            db.add(session_file)
        else:
            session_file.file_hash = file_hash
            session_file.is_committed = False
            session_file.just_uploaded = True
            session_file.status = "completed"
            db.add(session_file)

        await db.commit()
        logger.info(f"Updated SessionFile database record to 'completed' for '{filename}'.")

        return {"status": "completed", "chunks": num_chunks, "filename": filename}

    except Exception as e:
        logger.exception(f"Failed background RAG ingestion for '{filename}': {e}")
        try:
            result = await db.execute(
                select(SessionFile).where(
                    SessionFile.session_id == session_id,
                    SessionFile.filename == filename
                )
            )
            session_file = result.scalars().first()
            if session_file:
                session_file.status = "failed"
                db.add(session_file)
                await db.commit()
                logger.info(f"Updated SessionFile database record to 'failed' for '{filename}'.")
        except Exception as db_err:
            logger.error(f"Failed to record execution error inside database: {db_err}")
        raise e

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Successfully deleted temporary file at: {file_path}")
            except Exception as cleanup_err:
                logger.error(f"Failed to delete temporary file at '{file_path}': {cleanup_err}")
