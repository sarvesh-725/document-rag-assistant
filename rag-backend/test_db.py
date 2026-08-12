import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_async_engine(os.getenv('DATABASE_URL'), connect_args={"timeout": 60})

async def test():
    try:
        async with engine.begin() as conn:
            print('Connected successfully!')
    except Exception as e:
        print(f"Failed to connect: {e}")

asyncio.run(test())
