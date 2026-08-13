"""
RAG Engine — Core Qdrant and embedding operations.

Vector ingestion is version-scoped and idempotent. Qdrant is an index; PostgreSQL
owns document/version state.
"""

import os
import io
import hashlib
import uuid
import logging
from typing import List
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.services.chunking import (
    ParsedDocument, parse_parent_child_chunks, deterministic_point_id,
    deterministic_parent_id, PARSER_VERSION, CHUNKING_VERSION,
)

load_dotenv(override=True)

logger = logging.getLogger("rag_engine")

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "document_chunks")

MODEL_NAME = "models/gemini-embedding-2-preview"

client = AsyncQdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

_embeddings = None
_qdrant_initialized = False


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the embedding model instance, fallback to a standard model if
    the requested one fails, and cache it.
    """
    global MODEL_NAME, _embeddings
    if _embeddings is not None:
        return _embeddings

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set in the environment or .env file."
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)
        embeddings.embed_query("test connection")
        _embeddings = embeddings
        return _embeddings
    except Exception as e:
        logger.warning(f"Model initialization failed for {MODEL_NAME} with: {e}")
        fallback_model = "models/text-embedding-004"
        logger.warning(f"Attempting fallback to: {fallback_model}")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=fallback_model)
            embeddings.embed_query("test connection")
            MODEL_NAME = fallback_model
            _embeddings = embeddings
            return _embeddings
        except Exception as fallback_err:
            logger.error(f"Fallback model also failed: {fallback_err}")
            raise e


async def init_qdrant():
    """Checks if collection exists in Qdrant and programmatically creates it if not."""
    global _qdrant_initialized
    if _qdrant_initialized:
        return

    try:
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if QDRANT_COLLECTION_NAME not in collection_names:
            embeddings = _get_embeddings()
            sample_vector = embeddings.embed_query("test connection")
            vector_size = len(sample_vector)

            await client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size, distance=qdrant_models.Distance.COSINE
                ),
            )
            logger.info(
                f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created with size {vector_size}."
            )

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="user_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            logger.debug(f"Index creation for 'user_id' failed (may already exist): {idx_err}")

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="document_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            logger.debug(f"Index creation for 'document_id' failed (may already exist): {idx_err}")

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="version_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            logger.debug(f"Index creation for 'version_id' failed (may already exist): {idx_err}")

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="content_hash",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            logger.debug(f"Index creation for 'content_hash' failed (may already exist): {idx_err}")

        for field_name in ("parent_id", "chunk_id"):
            try:
                await client.create_payload_index(
                    collection_name=QDRANT_COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=(
                        qdrant_models.PayloadSchemaType.INTEGER
                        if field_name == "chunk_id"
                        else qdrant_models.PayloadSchemaType.KEYWORD
                    ),
                )
            except Exception as idx_err:
                logger.debug(f"Index creation for '{field_name}' failed (may already exist): {idx_err}")

        _qdrant_initialized = True
    except Exception as e:
        logger.error(f"Failed to check or initialize Qdrant collection: {e}")


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
    generates an MD5 hash of the bytes, chunks the text into Parent and Child chunks,
    generates embeddings for Children, and saves them to Qdrant Cloud.

    Re-running this function for the same version produces the same point IDs.
    It only upserts points belonging to the supplied version; it never purges
    filename matches or points from another document/version.
    """
    if not file_bytes:
        raise ValueError("File content is empty.")

    await init_qdrant()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    parsed = parsed or parse_parent_child_chunks(file_bytes, filename, version_id, document_id)
    children = parsed.children
    if not children:
        return 0

    embeddings = _get_embeddings()
    embeddings_list = embeddings.embed_documents([child.child_text for child in children])

    points = []
    for child, vector in zip(children, embeddings_list):
        point_id = child.id
        payload = {
            "user_id": str(user_id),
            "document_id": str(document_id),
            "version_id": str(version_id),
            "child_id": child.id,
            "parent_id": child.parent_id,
            "chunk_id": child.child_index,
            "filename": filename,
            "content_hash": file_hash,
            "source": filename,
            "child_text": child.child_text,
            "page": child.page,
            "section": child.section,
            "element_type": child.element_type,
            "source_position": child.source_position,
            "parser_version": PARSER_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "embedding": vector,
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
    except Exception as e:
        logger.error(f"Error deleting version {version_id} for document {document_id}: {e}")
