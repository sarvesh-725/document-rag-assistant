"""Authorization and version resolution for query document selection."""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.enums import DocumentStatus, VersionStatus
from app.database.models import Document, DocumentVersion


@dataclass(frozen=True)
class ResolvedQueryDocument:
    document_id: uuid.UUID
    version_id: uuid.UUID


class DocumentSelectionError(ValueError):
    """Raised when a query document selection cannot be used safely."""


def parse_selected_document_ids(selected_document_ids: list[str]) -> list[uuid.UUID]:
    if not selected_document_ids:
        raise DocumentSelectionError("At least one selected_document_id is required.")

    parsed_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw_id in selected_document_ids:
        try:
            document_id = uuid.UUID(str(raw_id))
        except (TypeError, ValueError) as exc:
            raise DocumentSelectionError(f"Invalid document_id: {raw_id}") from exc
        if document_id in seen:
            raise DocumentSelectionError(f"Duplicate document_id: {document_id}")
        seen.add(document_id)
        parsed_ids.append(document_id)
    return parsed_ids


async def resolve_selected_documents(
    db: AsyncSession,
    *,
    authenticated_user_id: uuid.UUID,
    selected_document_ids: list[str],
) -> list[ResolvedQueryDocument]:
    """Resolve a user selection into exact READY document/version pairs."""
    document_ids = parse_selected_document_ids(selected_document_ids)
    resolved: list[ResolvedQueryDocument] = []

    for document_id in document_ids:
        document = await db.get(Document, document_id)
        if document is None:
            raise DocumentSelectionError(f"Document is not available: {document_id}")
        if document.user_id != authenticated_user_id:
            raise DocumentSelectionError(f"Document is not available: {document_id}")
        if document.deleted_at is not None or document.status in {
            DocumentStatus.DELETING.value,
            DocumentStatus.DELETED.value,
        }:
            raise DocumentSelectionError(f"Document is deleted: {document_id}")
        if document.status != DocumentStatus.READY.value:
            raise DocumentSelectionError(f"Document is not ready: {document_id}")
        if document.current_version_id is None:
            raise DocumentSelectionError(f"Document has no ready version: {document_id}")

        version = await db.get(DocumentVersion, document.current_version_id)
        if (
            version is None
            or version.document_id != document.id
            or version.status != VersionStatus.READY.value
        ):
            raise DocumentSelectionError(f"Document is not ready: {document_id}")

        resolved.append(
            ResolvedQueryDocument(document_id=document.id, version_id=version.id)
        )

    return resolved
