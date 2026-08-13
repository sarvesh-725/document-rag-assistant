"""Standalone asyncio process for polling and dispatching outbox events."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Optional

from app.broker import broker
from app.config import get_settings
from app.database.connection import AsyncSessionLocal, engine
from app.services.outbox import publish_pending_outbox_events

logger = logging.getLogger("outbox_publisher")


async def run_outbox_publisher(
    *,
    stop_event: Optional[asyncio.Event] = None,
    polling_interval: Optional[float] = None,
    batch_size: int = 100,
) -> None:
    """Poll until stopped, using one short-lived SQLAlchemy session per pass."""
    settings = get_settings()
    interval = polling_interval or settings.outbox_polling_interval_seconds
    stop_event = stop_event or asyncio.Event()

    await broker.startup()
    try:
        while not stop_event.is_set():
            async with AsyncSessionLocal() as db:
                await publish_pending_outbox_events(db, limit=batch_size)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        await broker.shutdown()
        await engine.dispose()


async def main() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop_event.set)
        except (NotImplementedError, RuntimeError):
            # Windows and embedded event loops may not support add_signal_handler.
            signal.signal(signal_name, lambda *_args: stop_event.set())
    await run_outbox_publisher(stop_event=stop_event)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
