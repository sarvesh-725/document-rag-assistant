"""
RAG Engine — Core Qdrant and embedding operations.

Phase 1: Dead code removed (query_rag_system, undo_file_upload).
Retained: init_qdrant, _get_embeddings, process_and_store_document, delete_file_from_db.
These functions will be refactored in Phase 2 to use document_id instead of filename.
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
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import TokenTextSplitter

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

        _qdrant_initialized = True
    except Exception as e:
        logger.error(f"Failed to check or initialize Qdrant collection: {e}")


async def process_and_store_document(
    file_bytes: bytes, filename: str, user_id: str
) -> int:
    """
    Takes raw file bytes from an upload, parses the text using unstructured,
    generates an MD5 hash of the bytes, chunks the text into Parent and Child chunks,
    generates embeddings for Children, and saves them to Qdrant Cloud.

    NOTE: This function will be refactored in Phase 2 to use document_id/version_id
    instead of filename for Qdrant payload. Retained as-is for now to avoid
    breaking the ingestion pipeline structure.
    """
    if not file_bytes:
        raise ValueError("File content is empty.")

    await init_qdrant()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    try:
        loader = UnstructuredLoader(
            file=io.BytesIO(file_bytes),
            api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
            url=os.environ.get("UNSTRUCTURED_API_URL"),
            partition_via_api=True,
            strategy="hi_res",
        )
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        if not text.strip():
            raise ValueError("Extracted text is empty or could not be parsed.")
    except Exception as e:
        raise RuntimeError(f"Error parsing document with Unstructured loader: {e}")

    # Parent-Child Chunking Strategy
    parent_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=100)
    child_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)

    parents = parent_splitter.split_text(text)
    
    child_chunks = []
    parent_texts = []
    
    for parent in parents:
        children = child_splitter.split_text(parent)
        for child in children:
            child_chunks.append(child)
            parent_texts.append(parent)

    if not child_chunks:
        return 0

    embeddings = _get_embeddings()
    embeddings_list = embeddings.embed_documents(child_chunks)

    try:
        purge_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
                ),
                qdrant_models.FieldCondition(
                    key="filename", match=qdrant_models.MatchValue(value=filename)
                ),
            ],
            must_not=[
                qdrant_models.FieldCondition(
                    key="content_hash", match=qdrant_models.MatchValue(value=file_hash)
                )
            ],
        )
        await client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=purge_filter),
        )
    except Exception as e:
        logger.warning(f"Qdrant cleanup of outdated chunks failed or was empty: {e}")

    points = []
    for i, (child, parent, vector) in enumerate(zip(child_chunks, parent_texts, embeddings_list)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_hash}_{i}"))

        payload = {
            "user_id": str(user_id),
            "filename": filename,
            "content_hash": file_hash,
            "source": filename,
            "chunk_id": i,
            "text": child,
            "child_text": child,
            "parent_text": parent,
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    await client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    return len(child_chunks)


async def delete_file_from_db(user_id: str, filename: str):
    """
    Deletes all chunks associated with user_id and filename from Qdrant.

    NOTE: Will be refactored in Phase 2 to use document_id instead of filename.
    """
    await init_qdrant()
    try:
        delete_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
                ),
                qdrant_models.FieldCondition(
                    key="filename", match=qdrant_models.MatchValue(value=filename)
                ),
            ]
        )
        await client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=delete_filter),
        )
    except Exception as e:
        logger.error(f"Error deleting file {filename} for user {user_id} from Qdrant: {e}")
