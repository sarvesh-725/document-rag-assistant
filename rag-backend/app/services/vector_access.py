"""Shared Qdrant ownership filters for document retrieval.

The caller must supply the authenticated User's id, never an untrusted request user id.
"""

import uuid
from qdrant_client.http import models as qdrant_models


def owned_vector_filter(
    authenticated_user_id: uuid.UUID, version_ids: list[uuid.UUID]
) -> qdrant_models.Filter:
    """Constrain vector retrieval to the authenticated user and resolved versions."""
    conditions = [
        qdrant_models.FieldCondition(
            key="user_id", match=qdrant_models.MatchValue(value=str(authenticated_user_id))
        )
    ]
    if not version_ids:
        raise ValueError("Vector retrieval requires at least one resolved version_id")
    conditions.append(
        qdrant_models.FieldCondition(
            key="version_id",
            match=qdrant_models.MatchAny(any=[str(version_id) for version_id in version_ids]),
        )
    )
    return qdrant_models.Filter(must=conditions)
