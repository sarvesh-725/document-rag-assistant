"""Runtime configuration for security-sensitive application settings."""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    if not raw:
        raise RuntimeError("CORS_ORIGINS must list the permitted browser origins")
    try:
        origins = json.loads(raw) if raw.lstrip().startswith("[") else raw.split(",")
    except json.JSONDecodeError as exc:
        raise RuntimeError("CORS_ORIGINS must be comma-separated origins or a JSON list") from exc
    origins = [origin.strip() for origin in origins if origin.strip()]
    if not origins or "*" in origins:
        raise RuntimeError("CORS_ORIGINS must contain explicit origins; '*' is not permitted")
    return origins


class Settings:
    """Environment-only settings. There is intentionally no JWT secret fallback."""

    def __init__(self) -> None:
        self.secret_key = os.getenv("SECRET_KEY")
        self.algorithm = os.getenv("JWT_ALGORITHM", os.getenv("ALGORITHM", "HS256"))
        self.access_token_expire_minutes = _positive_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        self.redis_url = os.getenv("REDIS_URL")
        self.login_rate_limit = _positive_int("LOGIN_RATE_LIMIT", 10)
        self.login_rate_window_seconds = _positive_int("LOGIN_RATE_WINDOW_SECONDS", 60)
        self.login_failure_window_seconds = _positive_int("LOGIN_FAILURE_WINDOW_SECONDS", 900)
        self.login_backoff_base_seconds = _positive_int("LOGIN_BACKOFF_BASE_SECONDS", 2)
        self.login_backoff_max_seconds = _positive_int("LOGIN_BACKOFF_MAX_SECONDS", 300)
        self.history_recent_messages = self._bounded_int(
            "HISTORY_RECENT_MESSAGES", 4, minimum=0, maximum=20
        )
        self.model_context_limit = _positive_int("MODEL_CONTEXT_LIMIT", 32768)
        self.expected_output_tokens = _positive_int("EXPECTED_OUTPUT_TOKENS", 1024)
        self.summary_token_budget = _positive_int("SUMMARY_TOKEN_BUDGET", 512)
        self.history_token_budget = _positive_int("HISTORY_TOKEN_BUDGET", 2048)
        self.outbox_polling_interval_seconds = self._positive_float(
            "OUTBOX_POLLING_INTERVAL_SECONDS", 1.0
        )

    @staticmethod
    def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
        value = int(os.getenv(name, str(default)))
        if not minimum <= value <= maximum:
            raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _positive_float(name: str, default: float) -> float:
        value = float(os.getenv(name, str(default)))
        if value <= 0:
            raise RuntimeError(f"{name} must be positive")
        return value

    def require_secret_key(self) -> str:
        if not self.secret_key or len(self.secret_key) < 32:
            raise RuntimeError("SECRET_KEY must be set to a random value of at least 32 characters")
        return self.secret_key

    def require_redis_url(self) -> str:
        if not self.redis_url:
            raise RuntimeError("REDIS_URL must be configured for login throttling")
        return self.redis_url

    def validated_cors_origins(self) -> list[str]:
        return _cors_origins()


@lru_cache
def get_settings() -> Settings:
    return Settings()
