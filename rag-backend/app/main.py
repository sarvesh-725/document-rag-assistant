import logging
import hashlib
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services import rag_engine
from app.routers.documents import router as documents_router
from app.routers.chat import router as chat_router
from app.database.connection import get_db, engine, Base
from app.database.models import User, ChatSession, SessionFile
from app.auth.security import get_current_user, verify_password, get_password_hash, create_access_token

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="RAG Document Assistant Server",
    description="FastAPI Web Server for RAG (Retrieval-Augmented Generation) document search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


class UserSignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class QueryRequest(BaseModel):
    question: str
    session_id: str
    include_prev_files: bool = True
    explicit_files: List[str] = Field(default_factory=list)


class NewSessionRequest(BaseModel):
    session_id: str



@app.post("/api/v1/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: UserSignupRequest, db: AsyncSession = Depends(get_db)):
    """Registers a new user inside the database."""
    username_clean = request.username.strip()
    if not username_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty or whitespace."
        )

    result = await db.execute(select(User).where(User.username == username_clean))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    if len(request.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot be longer than 72 bytes."
        )

    hashed_password = get_password_hash(request.password)
    new_user = User(username=username_clean, hashed_password=hashed_password)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {"status": "success", "message": "User created successfully", "user_id": new_user.id}


@app.post("/api/v1/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticates credentials and returns a JWT access token."""
    username_clean = form_data.username.strip()
    result = await db.execute(select(User).where(User.username == username_clean))
    user = result.scalars().first()

    if len(form_data.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot be longer than 72 bytes."
        )

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}



@app.get("/api/v1/sessions", response_model=List[str])
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns a list of active session IDs for the authenticated user.
    Enforces strict tenant isolation.
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == current_user.id)
    )
    sessions = result.scalars().all()
    return [session.id for session in sessions]


@app.post("/api/v1/sessions/new")
async def create_new_session(
    request: NewSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Force-initializes a fresh session canvas for the authenticated user.
    Deletes any existing session with the same session_id to maintain clean state.
    """
    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID string cannot be empty or whitespace."
        )

    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    existing_session = result.scalars().first()
    if existing_session:
        await db.delete(existing_session)
        await db.commit()

    new_session = ChatSession(
        id=session_id,
        user_id=current_user.id,
        session_name=session_id,
        messages=[]
    )
    db.add(new_session)
    await db.commit()

    return {"status": "success", "user_id": current_user.id, "session_id": session_id}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes the session and cascaded files. Purges vector data if no other session uses it.
    """
    import os
    import re
    from app.services.rag_engine import delete_file_from_db

    session_id = session_id.strip()

    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )

    file_result = await db.execute(
        select(SessionFile).where(SessionFile.session_id == session_id)
    )
    session_files = file_result.scalars().all()
    filenames_to_purge = [f.filename for f in session_files]

    await db.delete(chat_session)
    await db.commit()

    celery_upload_dir = os.getenv("CELERY_UPLOAD_DIR", "UPLOADS")
    for filename in filenames_to_purge:
        try:
            stmt = select(SessionFile).join(ChatSession).where(
                ChatSession.user_id == current_user.id,
                SessionFile.filename == filename
            )
            other_res = await db.execute(stmt)
            if not other_res.scalars().first():
                await delete_file_from_db(str(current_user.id), filename)
                logger.info(f"Purged vector chunks from Qdrant for '{filename}'")
        except Exception as q_err:
            logger.error(f"Error purging Qdrant vectors for '{filename}' on session delete: {q_err}")

        try:
            safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            temp_filename = f"temp_{current_user.id}_{session_id}_{safe_filename}"
            temp_file_path = os.path.join(celery_upload_dir, temp_filename)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up local file: {temp_file_path}")
        except Exception as file_err:
            logger.error(f"Error removing temp file for '{filename}': {file_err}")

    return {"status": "success", "message": f"Successfully deleted session '{session_id}' and clean up resources."}




@app.get("/api/v1/sessions/files")
async def get_session_files(
    session_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns a list of all document files uploaded inside the target session.
    """
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

    result = await db.execute(
        select(SessionFile).where(SessionFile.session_id == session_id)
    )
    files = result.scalars().all()
    return [
        {
            "filename": f.filename,
            "file_hash": f.file_hash,
            "is_committed": f.is_committed,
            "just_uploaded": f.just_uploaded,
            "status": f.status
        }
        for f in files
    ]


@app.get("/api/v1/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns the chat message history for the target session.
    """
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = result.scalars().first()
    if not chat_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )
    return chat_session.messages





@app.get("/")
def read_root():
    return {"status": "running", "message": "Welcome to the RAG Document Assistant API"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local development server on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
