import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.rag_engine as rag_engine
import app.services.chunking as chunking


class FakeEmbeddings:
    def embed_documents(self, chunks):
        return [[float(index)] for index, _ in enumerate(chunks)]


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    async def upsert(self, **kwargs):
        self.upserts.append(kwargs["points"])

    async def delete(self, **kwargs):
        self.deletes.append(kwargs)


class FakeCollections:
    collections = [SimpleNamespace(name=rag_engine.QDRANT_COLLECTION_NAME)]


class FakeCollectionInfo:
    config = SimpleNamespace(
        params=SimpleNamespace(
            vectors=SimpleNamespace(
                size=rag_engine.EMBEDDING_PROFILE.dimension + 1,
                distance=rag_engine.EMBEDDING_PROFILE.distance,
            )
        )
    )


class IncompatibleQdrant(FakeQdrant):
    async def get_collections(self):
        return FakeCollections()

    async def get_collection(self, **kwargs):
        return FakeCollectionInfo()

    async def create_payload_index(self, **kwargs):
        pass


@pytest.mark.asyncio
async def test_same_version_ingestion_twice_reuses_point_ids(monkeypatch):
    document_id, version_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    qdrant = FakeQdrant()

    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def load(self):
            return [SimpleNamespace(page_content="one two three four five six")]

    monkeypatch.setattr(rag_engine, "client", qdrant)
    monkeypatch.setattr(rag_engine, "init_qdrant", AsyncMock())
    monkeypatch.setattr(rag_engine, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(chunking, "UnstructuredLoader", FakeLoader)
    monkeypatch.setattr(chunking, "TokenTextSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace(
        split_text=lambda text: [text] if chunk_size > 128 else [text]
    ))

    kwargs = {
        "user_id": str(user_id),
        "document_id": document_id,
        "version_id": version_id,
        "filename": "same-name.txt",
    }
    assert await rag_engine.process_and_store_document(b"source", **kwargs) == 1
    assert await rag_engine.process_and_store_document(b"source", **kwargs) == 1

    first_ids = [str(point.id) for point in qdrant.upserts[0]]
    second_ids = [str(point.id) for point in qdrant.upserts[1]]
    assert first_ids == second_ids
    assert qdrant.deletes == []
    payload = qdrant.upserts[0][0].payload
    assert payload["user_id"] == str(user_id)
    assert payload["document_id"] == str(document_id)
    assert payload["version_id"] == str(version_id)
    assert payload["parent_id"]
    assert payload["chunk_id"] == 0
    assert payload["embedding_profile"] == rag_engine.EMBEDDING_PROFILE.name
    assert payload["parser_version"] == chunking.PARSER_VERSION
    assert payload["chunking_version"] == chunking.CHUNKING_VERSION
    assert {"page_start", "page_end", "section", "element_type", "source_position"} <= payload.keys()
    assert "filename" not in payload
    assert "source" not in payload
    assert "content_hash" not in payload
    assert "embedding" not in payload


def test_point_id_is_deterministic_only_for_version_and_chunk():
    version_id = uuid.uuid4()
    assert rag_engine.deterministic_point_id(version_id, 4) == rag_engine.deterministic_point_id(version_id, 4)
    assert rag_engine.deterministic_point_id(version_id, 4) != rag_engine.deterministic_point_id(version_id, 5)
    assert rag_engine.deterministic_point_id(version_id, 4) != rag_engine.deterministic_point_id(uuid.uuid4(), 4)


def test_vector_writer_has_no_filename_based_delete_path():
    import inspect

    source = inspect.getsource(rag_engine.process_and_store_document)
    assert ".delete(" not in source
    assert '"filename"' not in source
    assert 'key="filename"' not in source
    assert '"document_id"' in source
    assert '"version_id"' in source


def test_collection_name_comes_from_embedding_profile():
    assert rag_engine.QDRANT_COLLECTION_NAME == rag_engine.EMBEDDING_PROFILE.collection_name


def test_embedding_initialization_does_not_fallback(monkeypatch):
    calls = []

    class BrokenEmbeddings:
        def __init__(self, model):
            calls.append(model)

        def embed_query(self, text):
            raise RuntimeError("model unavailable")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(rag_engine, "_embeddings", None)
    monkeypatch.setattr(rag_engine, "GoogleGenerativeAIEmbeddings", BrokenEmbeddings)

    with pytest.raises(RuntimeError, match="model unavailable"):
        rag_engine._get_embeddings()

    assert calls == [rag_engine.EMBEDDING_PROFILE.model]


@pytest.mark.asyncio
async def test_init_qdrant_fails_on_incompatible_existing_collection(monkeypatch):
    monkeypatch.setattr(rag_engine, "client", IncompatibleQdrant())
    monkeypatch.setattr(rag_engine, "_qdrant_initialized", False)

    with pytest.raises(RuntimeError, match="incompatible"):
        await rag_engine.init_qdrant()
