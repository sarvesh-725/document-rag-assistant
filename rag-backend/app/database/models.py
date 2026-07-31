from datetime import datetime
from typing import List
from sqlalchemy import String, Integer, ForeignKey, DateTime, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.connection import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)

    sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    session_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    messages: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="sessions")
    files: Mapped[List["SessionFile"]] = relationship(
        "SessionFile", back_populates="session", cascade="all, delete-orphan"
    )


class SessionFile(Base):
    __tablename__ = "session_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("chat_sessions.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_committed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    just_uploaded: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[str] = mapped_column(String, default="completed", nullable=True)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="files")
