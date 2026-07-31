import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from dotenv import load_dotenv

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

connect_args = {}
if "?" in DATABASE_URL:
    parsed_url = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed_url.query)
    if "sslmode" in query_params:
        query_params.pop("sslmode")
        connect_args["ssl"] = True
    if "channel_binding" in query_params:
        query_params.pop("channel_binding")
    new_query = urlencode(query_params, doseq=True)
    parsed_url = parsed_url._replace(query=new_query)
    DATABASE_URL = urlunparse(parsed_url)

engine = create_async_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
