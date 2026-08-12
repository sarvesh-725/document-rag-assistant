"""Redis-backed login rate limiting and progressive failure backoff."""

import hashlib
from fastapi import HTTPException, status
from app.config import Settings, get_settings


class LoginRateLimiter:
    def __init__(self, settings: Settings | None = None, redis_client=None) -> None:
        self.settings = settings or get_settings()
        if redis_client is None:
            try:
                from redis import asyncio as redis_asyncio
                redis_client = redis_asyncio.from_url(self.settings.require_redis_url(), decode_responses=True)
            except Exception as exc:
                raise RuntimeError("Redis login throttling could not be initialized") from exc
        self.redis = redis_client

    @staticmethod
    def _key_component(username: str, client_ip: str) -> str:
        return hashlib.sha256(f"{username.casefold()}:{client_ip}".encode()).hexdigest()

    def _keys(self, username: str, client_ip: str) -> tuple[str, str, str]:
        component = self._key_component(username, client_ip)
        return (
            f"auth:login:rate:{component}",
            f"auth:login:failure:{component}",
            f"auth:login:backoff:{component}",
        )

    async def check(self, username: str, client_ip: str) -> None:
        rate_key, _, backoff_key = self._keys(username, client_ip)
        try:
            backoff_ttl = await self.redis.ttl(backoff_key)
            if backoff_ttl > 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed login attempts. Try again later.",
                    headers={"Retry-After": str(backoff_ttl)},
                )

            attempts = await self.redis.incr(rate_key)
            if attempts == 1:
                await self.redis.expire(rate_key, self.settings.login_rate_window_seconds)
            if attempts > self.settings.login_rate_limit:
                retry_after = max(await self.redis.ttl(rate_key), 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Try again later.",
                    headers={"Retry-After": str(retry_after)},
                )
        except HTTPException:
            raise
        except Exception as exc:
            # Fail closed: disabling Redis must not silently disable brute-force protection.
            raise HTTPException(status_code=503, detail="Login protection is temporarily unavailable") from exc

    async def record_failure(self, username: str, client_ip: str) -> None:
        _, failure_key, backoff_key = self._keys(username, client_ip)
        try:
            failures = await self.redis.incr(failure_key)
            delay = min(
                self.settings.login_backoff_base_seconds * (2 ** (int(failures) - 1)),
                self.settings.login_backoff_max_seconds,
            )
            await self.redis.expire(failure_key, self.settings.login_failure_window_seconds)
            await self.redis.set(backoff_key, "1", ex=delay)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Login protection is temporarily unavailable") from exc

    async def record_success(self, username: str, client_ip: str) -> None:
        _, failure_key, backoff_key = self._keys(username, client_ip)
        try:
            await self.redis.delete(failure_key, backoff_key)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Login protection is temporarily unavailable") from exc
