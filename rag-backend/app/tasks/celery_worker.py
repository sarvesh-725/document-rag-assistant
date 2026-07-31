import os
import asyncio
import logging
from celery import Celery
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger("celery_worker")
logging.basicConfig(level=logging.INFO)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_tasks",
    broker=REDIS_URL
)

celery_app.conf.update(
    task_ignore_result=True,          # Disable storing task results in Redis
    task_track_started=False,         # Disable tracking started state in backend
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC"
)
celery_app.conf.broker_transport_options = {
    'polling_interval': 5.0  # Poll every 5 seconds when idle instead of multiple times a second
}
celery_app.conf.worker_send_task_events = False


async def process_document_internal(file_path: str, filename: str, user_id: int, session_id: str):
    """
    Internal asynchronous processor that performs document ingestion,
    updates database states, and cleans up the temporary file.
    """
    from app.database.connection import AsyncSessionLocal
    from app.database.models import SessionFile
    from app.services import rag_engine
    from sqlalchemy import select
    import hashlib

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

        async with AsyncSessionLocal() as db:
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
            async with AsyncSessionLocal() as db:
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

        try:
            from app.database.connection import engine
            await engine.dispose()
            logger.info("Successfully disposed database engine connection pool for event loop cleanup.")
        except Exception as dispose_err:
            logger.error(f"Failed to dispose database engine pool: {dispose_err}")



@celery_app.task(name="tasks.process_document", bind=True)
def async_process_document_task(self, file_path: str, filename: str, user_id: int, session_id: str):
    """
    Celery background worker task entrypoint. Runs process_document_internal in an
    event loop since Celery operates synchronously.
    """
    logger.info(f"Task {self.request.id} received: processing '{filename}'")
    return asyncio.run(process_document_internal(file_path, filename, user_id, session_id))
