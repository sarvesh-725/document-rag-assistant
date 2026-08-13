import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.services.rag_engine as rag_engine
import app.services.chunking as chunking


def _patch_parser(monkeypatch):
    class FakeLoader:
        def __init__(self, **kwargs):
            pass

        def load(self):
            return [
                SimpleNamespace(page_content="Title and first page", metadata={"page_number": 1, "category": "Title", "section": "Intro"}),
                SimpleNamespace(page_content="Second page body", metadata={"page_number": 2, "category": "NarrativeText", "section": "Body"}),
            ]

    monkeypatch.setattr(chunking, "UnstructuredLoader", FakeLoader)


@pytest.mark.asyncio
async def test_child_generation_parent_mapping_and_metadata(monkeypatch):
    _patch_parser(monkeypatch)
    monkeypatch.setattr(chunking, "TokenTextSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace(
        split_text=lambda text: [text] if chunk_size >= 1024 else [text[:8], text[8:]]
    ))
    version_id = uuid.uuid4()
    parsed = rag_engine.parse_parent_child_chunks(b"source", "doc.txt", version_id)

    assert parsed.parents
    assert parsed.children
    assert all(child.parent_id in {parent.id for parent in parsed.parents} for child in parsed.children)
    assert parsed.parents[0].page_start == 1
    assert parsed.parents[-1].page_end == 2
    assert parsed.parents[0].section == "Intro"
    assert parsed.parents[0].element_type == "Title"
    assert parsed.parents[0].source_position["elements"]


def test_parent_and_child_ids_are_deterministic():
    version_id = uuid.uuid4()
    assert rag_engine.deterministic_parent_id(version_id, 0) == rag_engine.deterministic_parent_id(version_id, 0)
    assert rag_engine.deterministic_point_id(version_id, 0) == rag_engine.deterministic_point_id(version_id, 0)
    assert rag_engine.deterministic_parent_id(version_id, 0) != rag_engine.deterministic_parent_id(version_id, 1)


def test_parent_storage_model_has_single_canonical_text_field():
    from app.database.models import DocumentParent

    assert DocumentParent.__table__.c.text.nullable is False
    assert "uq_document_parent_index" in {constraint.name for constraint in DocumentParent.__table__.constraints}


@pytest.mark.asyncio
async def test_no_parent_text_in_child_vector_payload(monkeypatch):
    _patch_parser(monkeypatch)
    monkeypatch.setattr(chunking, "TokenTextSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace(
        split_text=lambda text: [text] if chunk_size >= 1024 else [text[:8], text[8:]]
    ))
    version_id = uuid.uuid4()
    
    # Mock Qdrant client to intercept upsert payload
    mock_upsert = AsyncMock()
    rag_engine.client.upsert = mock_upsert
    rag_engine._get_embeddings = lambda: SimpleNamespace(embed_documents=lambda x: [[0.0]] * len(x))
    rag_engine.init_qdrant = AsyncMock()
    
    await rag_engine.process_and_store_document(
        b"dummy",
        user_id="user1",
        document_id=uuid.uuid4(),
        version_id=version_id,
        filename="test.txt"
    )
    
    mock_upsert.assert_called_once()
    points = mock_upsert.call_args.kwargs["points"]
    assert len(points) > 0
    payload = points[0].payload
    
    assert "parent_id" in payload
    assert "parent_text" not in payload


def test_page_propagation_across_parents(monkeypatch):
    class FakeLoader:
        def __init__(self, **kwargs):
            pass
        def load(self):
            return [
                SimpleNamespace(page_content="A"*100, metadata={"page_number": 1}),
                SimpleNamespace(page_content="B"*100, metadata={"page_number": 2}),
                SimpleNamespace(page_content="C"*100, metadata={"page_number": 3}),
            ]
    
    monkeypatch.setattr(chunking, "UnstructuredLoader", FakeLoader)
    # Force everything into one big chunk to verify min/max page propagation
    monkeypatch.setattr(chunking, "TokenTextSplitter", lambda chunk_size, chunk_overlap: SimpleNamespace(
        split_text=lambda text: [text] if chunk_size >= 1024 else [text]
    ))
    
    version_id = uuid.uuid4()
    parsed = rag_engine.parse_parent_child_chunks(b"source", "doc.txt", version_id)
    assert len(parsed.parents) == 1
    
    parent = parsed.parents[0]
    assert parent.page_start == 1
    assert parent.page_end == 3


def test_parser_and_chunking_versions_stored():
    from app.services.chunking import PARSER_VERSION, CHUNKING_VERSION
    assert PARSER_VERSION
    assert CHUNKING_VERSION
