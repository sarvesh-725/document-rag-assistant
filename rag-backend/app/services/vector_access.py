"""Shared Qdrant ownership filters for document retrieval.

The caller must supply the authenticated User's id, never an untrusted request user id.
"""

import uuid
from qdrant_client.http import models as qdrant_models


def owned_vector_filter(
    authenticated_user_id: uuid.UUID, document_ids: list[uuid.UUID]
) -> qdrant_models.Filter:
    """Constrain a vector operation to documents owned by the authenticated user."""
    conditions = [
        qdrant_models.FieldCondition(
            key="user_id", match=qdrant_models.MatchValue(value=str(authenticated_user_id))
        )
    ]
    if document_ids:
        conditions.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchAny(any=[str(document_id) for document_id in document_ids]),
            )
        )
    return qdrant_models.Filter(must=conditions)
