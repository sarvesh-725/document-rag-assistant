import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.connection import get_db
from app.database.models import User
from app.errors import ErrorCode, api_error

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if len(plain_password.encode("utf-8")) > 72:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password cannot be longer than 72 bytes")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    settings = get_settings()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.require_secret_key(), algorithm=settings.algorithm)


def _credentials_error() -> HTTPException:
    error = api_error(ErrorCode.AUTH_REQUIRED, "Could not validate credentials", 401)
    error.headers = {"WWW-Authenticate": "Bearer"}
    return error


async def _user_id_from_token(token: str, db: AsyncSession) -> uuid.UUID:
    credentials_exception = _credentials_error()
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.require_secret_key(), algorithms=[settings.algorithm])
        username = payload.get("sub")
        user_id = uuid.UUID(payload.get("user_id"))
    except (JWTError, ValueError, TypeError):
        raise credentials_exception
    result = await db.execute(select(User.id).where(User.username == username, User.id == user_id))
    if result.scalar_one_or_none() is None:
        raise credentials_exception
    return user_id


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    user_id = await _user_id_from_token(token, db)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if user is None:
        raise _credentials_error()
    return user


async def get_current_user_id(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> uuid.UUID:
    return await _user_id_from_token(token, db)
