import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.database.models import User
from app.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a hashed one, enforcing the 72 bytes limit."""
    if len(plain_password.encode('utf-8')) > 72:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash for a plain text password, enforcing the 72 bytes limit."""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        raise ValueError("Password cannot be longer than 72 bytes")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token containing data claims."""
    to_encode = data.copy()
    settings = get_settings()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.require_secret_key(), algorithm=settings.algorithm)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    FastAPI dependency that decodes JWT from auth header,
    looks up user by 'sub' claim in database, and returns the User.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.require_secret_key(), algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        user_id_claim: str = payload.get("user_id")
        if username is None or user_id_claim is None:
            raise credentials_exception
        user_id = uuid.UUID(user_id_claim)
    except (JWTError, ValueError, TypeError):
        raise credentials_exception

    result = await db.execute(select(User).where(User.username == username, User.id == user_id))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user
