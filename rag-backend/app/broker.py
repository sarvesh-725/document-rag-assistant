from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
from taskiq import SimpleRetryMiddleware
from app.config import get_settings

REDIS_URL = get_settings().require_redis_url()

result_backend = RedisAsyncResultBackend(
    redis_url=REDIS_URL,
    socket_timeout=None,
)

broker = ListQueueBroker(
    url=REDIS_URL,
    socket_timeout=None,
).with_result_backend(
    result_backend
).with_middlewares(
    SimpleRetryMiddleware(default_retry_count=3)
)

@broker.on_event("shutdown")
async def shutdown_event() -> None:
    if broker.is_worker_process:
        from app.database.connection import engine
        await engine.dispose()

# Import task modules here so the broker discovers them on startup
import app.tasks.taskiq_worker  # noqa
