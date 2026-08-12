"""Drop all tables from the database. Used before initial migration."""
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from app.database.connection import engine
from sqlalchemy import text

async def drop_all():
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        tables = [row[0] for row in result.fetchall()]
        if tables:
            for t in tables:
                await conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                print(f"Dropped: {t}")
        else:
            print("No tables to drop - database is clean")
    await engine.dispose()

asyncio.run(drop_all())
