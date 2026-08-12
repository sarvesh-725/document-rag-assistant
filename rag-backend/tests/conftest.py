"""Test-only runtime configuration for importing the FastAPI application."""

import os


os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production-0123456789")
os.environ.setdefault("CORS_ORIGINS", "http://testserver")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
