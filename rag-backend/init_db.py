import asyncio
from app.database.connection import engine
from app.database.models import Base

async def init_models():
    print("Connecting to Neon PostgreSQL and dropping existing tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("Generating fresh tables from models...")
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialization successful!")

if __name__ == "__main__":
    asyncio.run(init_models())