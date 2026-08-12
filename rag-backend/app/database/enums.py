"""
Domain enums for the Document RAG Assistant.

All status, role, and stage enumerations used across the domain models.
These are stored as VARCHAR in PostgreSQL via SQLAlchemy's native Enum support.
"""

import enum


class DocumentStatus(str, enum.Enum):
    """Lifecycle status of a Document."""
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    DELETING = "DELETING"
    DELETED = "DELETED"


class VersionStatus(str, enum.Enum):
    """Status of a DocumentVersion."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class MessageRole(str, enum.Enum):
    """Role of a chat message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, enum.Enum):
    """Lifecycle status of a Message."""
    PENDING = "PENDING"
    STREAMING = "STREAMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class IngestionStatus(str, enum.Enum):
    """Status of an IngestionJob."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class IngestionStage(str, enum.Enum):
    """Current processing stage of an IngestionJob."""
    UPLOAD = "UPLOAD"
    PARSE = "PARSE"
    CHUNK = "CHUNK"
    EMBED = "EMBED"
    INDEX = "INDEX"
    FINALIZE = "FINALIZE"


class QueryRunStatus(str, enum.Enum):
    """Status of a QueryRun."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
