"""
Database Reset Tool (LOCAL DEVELOPMENT ONLY)

This script:
1. Drops all tables via CASCADE
2. Runs Alembic migrations to upgrade to head
3. Seeds initial test data if requested

WARNING: Destructive operation. All data will be lost.
"""

import asyncio
import os
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.database.connection import engine, AsyncSessionLocal
from app.auth.security import get_password_hash
from app.database.repositories import (
    create_user, create_document, create_document_version,
    create_session, create_message
)


async def drop_all_tables():
    print("Dropping all existing tables...")
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        tables = [row[0] for row in result.fetchall()]
        if tables:
            for t in tables:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                print(f"  Dropped: {t}")
        else:
            print("  No tables found.")


def run_alembic_upgrade():
    print("\nRunning Alembic migrations...")
    alembic_cfg = Config("alembic.ini")
    # Setting the URL is handled internally by env.py reading from .env
    command.upgrade(alembic_cfg, "head")
    print("  Migrations applied successfully.")


async def seed_test_data():
    print("\nSeeding test data...")
    async with AsyncSessionLocal() as db:
        try:
            # 1. Create a test user
            user = await create_user(db, "testuser", get_password_hash("password123"))
            
            # 2. Create a test document
            doc = await create_document(db, user.id, "welcome.pdf")
            version = await create_document_version(
                db, doc.id, "hash123", "uploads/welcome.pdf", "v1", "v1", "gemini-2"
            )
            from app.database.repositories import set_current_version
            await set_current_version(db, doc.id, version.id)
            
            # 3. Create a test session
            session = await create_session(db, user.id, "Initial Chat")
            await create_message(
                db, session.id, "USER", "Hello, what is this system?"
            )
            await create_message(
                db, session.id, "ASSISTANT", "I am your RAG Document Assistant."
            )
            
            await db.commit()
            print("  Test user 'testuser' (pw: password123) and sample data seeded.")
        except Exception as e:
            await db.rollback()
            print(f"  Failed to seed data: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Database Reset Tool")
    parser.add_argument("--seed", action="store_true", help="Seed test data after reset")
    args = parser.parse_args()

    print("=" * 60)
    print("Database Reset Tool (LOCAL DEVELOPMENT ONLY)")
    print("=" * 60)

    # 1. Drop existing tables
    await drop_all_tables()
    
    # Alembic relies on a synchronous DBAPI or SQLAlchemy sync engine normally,
    # but our env.py handles the async engine internally.
    # We must run it outside the async event loop to avoid nest_asyncio issues.
    
    # 2. Dispose current engine before running alembic to free connections
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
    
    # Run alembic synchronously (env.py handles its own async event loop)
    run_alembic_upgrade()
    
    # Run seed data if requested
    import sys
    if "--seed" in sys.argv:
        asyncio.run(seed_test_data())