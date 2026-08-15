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
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.qdrant_url = os.getenv("QDRANT_URL", os.getenv("QDRANT_ENDPOINT"))
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY"))
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
        self.jwt_secret = os.getenv("JWT_SECRET", os.getenv("SECRET_KEY"))
        self.secret_key = self.jwt_secret
        self.algorithm = os.getenv("JWT_ALGORITHM", os.getenv("ALGORITHM", "HS256"))
        self.jwt_expiration_minutes = _positive_int(
            "JWT_EXPIRATION_MINUTES",
            int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        )
        self.access_token_expire_minutes = self.jwt_expiration_minutes
        self.frontend_origin = os.getenv("FRONTEND_ORIGIN", os.getenv("CORS_ORIGINS", ""))
        self.storage_root = os.getenv("STORAGE_ROOT", "UPLOADS")
        self.embedding_profile = os.getenv("EMBEDDING_PROFILE", os.getenv("EMBEDDING_PROFILE_NAME", "gemini_embedding_v1"))
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", os.getenv("EMBEDDING_COLLECTION_NAME"))
        self.dense_top_k = _positive_int("DENSE_TOP_K", 30)
        self.bm25_top_k = _positive_int("BM25_TOP_K", 30)
        self.rerank_top_k = _positive_int("RERANK_TOP_K", 10)
        self.final_context_k = _positive_int("FINAL_CONTEXT_K", 8)
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
        self.max_file_size = _positive_int("MAX_FILE_SIZE", 10 * 1024 * 1024)
        self.max_documents_per_user = _positive_int("MAX_DOCUMENTS_PER_USER", 100)
        self.max_selected_documents_per_query = _positive_int("MAX_SELECTED_DOCUMENTS_PER_QUERY", 20)
        self.max_pages_per_file = _positive_int("MAX_PAGES_PER_FILE", 500)
        self.max_child_chunks = _positive_int("MAX_CHILD_CHUNKS", 10000)
        self.max_dense_k = _positive_int("MAX_DENSE_K", 100)
        self.max_bm25_k = _positive_int("MAX_BM25_K", 100)
        self.max_rerank_k = _positive_int("MAX_RERANK_K", 100)
        self.max_context_tokens = _positive_int("MAX_CONTEXT_TOKENS", 32768)
        self.max_output_tokens = _positive_int("MAX_OUTPUT_TOKENS", 4096)

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
