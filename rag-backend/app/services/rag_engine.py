"""
RAG Engine — Core Qdrant and embedding operations.

Vector ingestion is version-scoped and idempotent. Qdrant is an index; PostgreSQL
owns document/version state.
"""

import uuid
import logging
import time
import os
from dataclasses import dataclass
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.services.chunking import (
    ParsedDocument, parse_parent_child_chunks, deterministic_point_id,
    deterministic_parent_id,
)
from app.observability import metrics
from app.config import get_settings

settings = get_settings()

logger = logging.getLogger("rag_engine")

QDRANT_ENDPOINT = settings.qdrant_url
QDRANT_API_KEY = settings.qdrant_api_key


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model: str
    dimension: int
    distance: qdrant_models.Distance
    version: str
    collection_name: str


def _embedding_profile_from_env() -> EmbeddingProfile:
    name = settings.embedding_profile
    return EmbeddingProfile(
        name=name,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        distance=qdrant_models.Distance.COSINE,
        version=settings.embedding_profile_version,
        collection_name=settings.qdrant_collection,
    )


EMBEDDING_PROFILE = _embedding_profile_from_env()
QDRANT_COLLECTION_NAME = EMBEDDING_PROFILE.collection_name

client = AsyncQdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

_embeddings = None
_qdrant_initialized = False


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the configured embedding model instance and cache it.

    A Qdrant collection is tied to exactly one embedding profile. Fallbacks are
    deliberately forbidden because mixed dimensions/models inside one collection
    make retrieval correctness undefined.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set in the environment or .env file."
        )

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_PROFILE.model)
    embeddings.embed_query("test connection")
    _embeddings = embeddings
    return _embeddings


def _distance_label(distance) -> str:
    value = getattr(distance, "value", distance)
    return str(value).lower()


def _collection_vector_config(collection_info) -> tuple[int, object]:
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION_NAME}' uses named vectors, "
            "but this service is configured for a single unnamed vector."
        )
    return int(vectors_config.size), vectors_config.distance


async def _verify_collection_matches_profile() -> None:
    collection_info = await client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    vector_size, distance = _collection_vector_config(collection_info)
    expected_distance = _distance_label(EMBEDDING_PROFILE.distance)
    actual_distance = _distance_label(distance)
    if vector_size != EMBEDDING_PROFILE.dimension or actual_distance != expected_distance:
        raise RuntimeError(
            "Qdrant collection is incompatible with the configured embedding profile: "
            f"collection={QDRANT_COLLECTION_NAME}, "
            f"profile={EMBEDDING_PROFILE.name}, "
            f"expected_dimension={EMBEDDING_PROFILE.dimension}, "
            f"actual_dimension={vector_size}, "
            f"expected_distance={expected_distance}, "
            f"actual_distance={actual_distance}. "
            "Create a new profile/collection or migrate explicitly; refusing to rebuild silently."
        )


async def _ensure_payload_indexes() -> None:
    for field_name in ("user_id", "document_id", "version_id", "parent_id"):
        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            logger.debug(f"Index creation for '{field_name}' failed (may already exist): {idx_err}")


async def init_qdrant():
    """Create or verify the configured Qdrant collection and payload indexes."""
    global _qdrant_initialized
    if _qdrant_initialized:
        return

    collections = await client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if QDRANT_COLLECTION_NAME not in collection_names:
        sample_vector = _get_embeddings().embed_query("test connection")
        if len(sample_vector) != EMBEDDING_PROFILE.dimension:
            raise RuntimeError(
                "Configured embedding profile dimension does not match the model output: "
                f"profile={EMBEDDING_PROFILE.name}, "
                f"model={EMBEDDING_PROFILE.model}, "
                f"configured_dimension={EMBEDDING_PROFILE.dimension}, "
                f"actual_dimension={len(sample_vector)}."
            )
        await client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_PROFILE.dimension,
                distance=EMBEDDING_PROFILE.distance,
            ),
        )
        logger.info(
            "Qdrant collection '%s' created for embedding profile '%s'.",
            QDRANT_COLLECTION_NAME,
            EMBEDDING_PROFILE.name,
        )

    await _verify_collection_matches_profile()
    await _ensure_payload_indexes()
    _qdrant_initialized = True


def deterministic_point_id(version_id: uuid.UUID | str, chunk_id: int) -> str:
    """Return the stable point identity for one version/chunk pair."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}:{chunk_id}"))


def deterministic_parent_id(version_id: uuid.UUID | str, parent_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}:parent:{parent_index}"))


async def process_and_store_document(
    file_bytes: bytes,
    *,
    user_id: str,
    document_id: uuid.UUID | str,
    version_id: uuid.UUID | str,
    filename: str,
    parsed: ParsedDocument | None = None,
) -> int:
    """
    Takes raw file bytes from an upload, parses the text using unstructured,
    chunks the text into Parent and Child chunks, generates embeddings for
    Children, and saves them to Qdrant Cloud.

    Re-running this function for the same version produces the same point IDs.
    It only upserts points belonging to the supplied version; it never purges
    filename matches or points from another document/version.
    """
    if not file_bytes:
        raise ValueError("File content is empty.")

    await init_qdrant()

    parsed = parsed or parse_parent_child_chunks(file_bytes, filename, version_id, document_id)
    children = parsed.children
    if not children:
        return 0

    embeddings = _get_embeddings()
    embedding_started = time.perf_counter()
    embeddings_list = embeddings.embed_documents([child.child_text for child in children])
    metrics.observe("embedding_latency", time.perf_counter() - embedding_started)

    points = []
    for child, vector in zip(children, embeddings_list):
        point_id = child.id
        payload = {
            "user_id": str(user_id),
            "document_id": str(document_id),
            "version_id": str(version_id),
            "parent_id": child.parent_id,
            "chunk_id": child.child_index,
            "child_text": child.child_text,
            "page_start": child.page_start,
            "page_end": child.page_end,
            "section": child.section,
            "element_type": child.element_type,
            "source_position": child.source_position,
            "embedding_profile": EMBEDDING_PROFILE.name,
            "parser_version": child.parser_version,
            "chunking_version": child.chunking_version,
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    await client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    return len(children)


async def delete_version_vectors(
    user_id: str, document_id: uuid.UUID | str, version_id: uuid.UUID | str
):
    """
    Delete only one exact document version's vectors. Filename is deliberately
    not accepted because it is not an identity or authorization boundary.
    """
    await init_qdrant()
    try:
        delete_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
                ),
                qdrant_models.FieldCondition(
                    key="document_id", match=qdrant_models.MatchValue(value=str(document_id))
                ),
                qdrant_models.FieldCondition(
                    key="version_id", match=qdrant_models.MatchValue(value=str(version_id))
                ),
            ]
        )
        await client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=delete_filter),
        )
    except Exception:
        logger.exception(
            "Error deleting version %s for document %s",
            version_id,
            document_id,
        )
        raise
