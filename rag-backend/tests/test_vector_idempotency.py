import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.rag_engine as rag_engine


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
    monkeypatch.setattr(rag_engine, "UnstructuredLoader", FakeLoader)
    monkeypatch.setattr(rag_engine, "TokenTextSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace(
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


def test_point_id_is_deterministic_only_for_version_and_chunk():
    version_id = uuid.uuid4()
    assert rag_engine.deterministic_point_id(version_id, 4) == rag_engine.deterministic_point_id(version_id, 4)
    assert rag_engine.deterministic_point_id(version_id, 4) != rag_engine.deterministic_point_id(version_id, 5)
    assert rag_engine.deterministic_point_id(version_id, 4) != rag_engine.deterministic_point_id(uuid.uuid4(), 4)


def test_vector_writer_has_no_filename_based_delete_path():
    import inspect

    source = inspect.getsource(rag_engine.process_and_store_document)
    assert ".delete(" not in source
    assert 'key="filename"' not in source
    assert '"document_id"' in source
    assert '"version_id"' in source
