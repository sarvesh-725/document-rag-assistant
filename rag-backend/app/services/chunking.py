"""
Parent/child document chunking pipeline.

Pure parsing and chunking logic, separated from vector-store I/O.
Preserves page, section, and element-type metadata from Unstructured
elements and produces deterministic parent/child IDs.

Design:
    Parent ≈ 1024 tokens, Child ≈ 128 tokens, with configured overlap.
    Parent text is stored exactly once in PostgreSQL.
    Child text is stored in Qdrant with a reference (parent_id) back to
    the parent — the full parent text is never duplicated in vectors.
"""

import hashlib
import io
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_text_splitters import TokenTextSplitter
from langchain_unstructured import UnstructuredLoader
from app.config import get_settings

# Versioning constants — bump when the algorithm changes.
PARSER_VERSION = "unstructured-hi_res-v1"
CHUNKING_VERSION = "parent1024-child128-v1"

# Chunk size configuration
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 128
CHILD_CHUNK_OVERLAP = 20


@dataclass
class ParsedParent:
    """A parent chunk with metadata derived from its constituent elements."""

    id: str
    version_id: str
    parent_index: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    element_type: Optional[str] = None
    source_position: Optional[dict] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    chunking_version: str = CHUNKING_VERSION


@dataclass
class ParsedChild:
    """A child chunk that references its parent."""

    id: str
    parent_id: str
    document_id: Optional[str] = None
    version_id: Optional[str] = None
    child_text: str = ""
    child_index: int = 0
    page: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    element_type: Optional[str] = None
    source_position: Optional[dict] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    chunking_version: str = CHUNKING_VERSION


@dataclass
class ParsedDocument:
    """Complete parse result for one document version."""

    parents: List[ParsedParent]
    children: List[ParsedChild]
    content_hash: str


def deterministic_point_id(version_id: uuid.UUID | str, chunk_id: int) -> str:
    """Return the stable point identity for one version/chunk pair."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}:{chunk_id}"))


def deterministic_parent_id(version_id: uuid.UUID | str, parent_index: int) -> str:
    """Return the stable parent identity for one version/parent-index pair."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}:parent:{parent_index}"))


@dataclass
class _ElementInfo:
    """Metadata extracted from a single Unstructured element."""

    text: str
    page_number: Optional[int] = None
    category: Optional[str] = None
    section: Optional[str] = None


def _extract_elements(file_bytes: bytes, filename: str) -> List[_ElementInfo]:
    """Parse a document via Unstructured, preserving per-element metadata."""
    loader = UnstructuredLoader(
        file=io.BytesIO(file_bytes),
        api_key=get_settings().unstructured_api_key,
        url=get_settings().unstructured_api_url,
        partition_via_api=True,
        strategy="hi_res",
    )
    docs = loader.load()

    elements: List[_ElementInfo] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {})
        if isinstance(meta, dict):
            page = meta.get("page_number")
            category = meta.get("category")
            section = meta.get("section")
        else:
            page = getattr(meta, "page_number", None)
            category = getattr(meta, "category", None)
            section = getattr(meta, "section", None)

        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        elements.append(_ElementInfo(
            text=doc.page_content,
            page_number=page,
            category=category,
            section=section,
        ))

    return elements


def _build_parents(
    elements: List[_ElementInfo],
    version_id: uuid.UUID | str,
) -> List[ParsedParent]:
    """
    Create parent chunks from parsed elements.

    Joins all element texts, splits into parent-sized chunks, then maps
    each parent chunk back to the elements it spans so we can derive
    page_start, page_end, section, element_type, and source_position.
    """
    if not elements:
        return []

    parent_splitter = TokenTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )

    full_text = "\n\n".join(el.text for el in elements)
    parent_texts = parent_splitter.split_text(full_text)

    if not parent_texts:
        return []

    # Build a character-offset index for each element in the joined text.
    element_offsets: List[tuple[int, int]] = []  # (start, end) for each element
    cursor = 0
    for i, el in enumerate(elements):
        start = full_text.find(el.text, cursor)
        if start == -1:
            start = cursor
        end = start + len(el.text)
        element_offsets.append((start, end))
        cursor = end

    parents: List[ParsedParent] = []
    search_from = 0
    for parent_index, parent_text in enumerate(parent_texts):
        parent_start = full_text.find(parent_text, search_from)
        if parent_start == -1:
            parent_start = search_from
        parent_end = parent_start + len(parent_text)
        # Don't advance search_from past parent_start for overlapping chunks
        search_from = parent_start + 1

        # Find which elements overlap with this parent chunk.
        contributing_elements: List[dict] = []
        pages: List[int] = []
        first_section: Optional[str] = None
        first_element_type: Optional[str] = None

        for el_idx, (el_start, el_end) in enumerate(element_offsets):
            # Check overlap: element intersects parent range
            if el_start < parent_end and el_end > parent_start:
                el = elements[el_idx]
                contributing_elements.append({
                    "index": el_idx,
                    "category": el.category,
                    "section": el.section,
                    "page_number": el.page_number,
                })
                if el.page_number is not None:
                    pages.append(el.page_number)
                if first_section is None and el.section:
                    first_section = el.section
                if first_element_type is None and el.category:
                    first_element_type = el.category

        parent_id = deterministic_parent_id(version_id, parent_index)

        parents.append(ParsedParent(
            id=parent_id,
            version_id=str(version_id),
            parent_index=parent_index,
            text=parent_text,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section=first_section,
            element_type=first_element_type,
            source_position={"elements": contributing_elements},
        ))

    return parents


def _build_children(
    parents: List[ParsedParent],
    version_id: uuid.UUID | str,
    document_id: Optional[uuid.UUID | str] = None,
) -> List[ParsedChild]:
    """Split each parent into child chunks, preserving lineage."""
    child_splitter = TokenTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    children: List[ParsedChild] = []
    global_child_index = 0

    for parent in parents:
        child_texts = child_splitter.split_text(parent.text)
        for child_text in child_texts:
            child_id = deterministic_point_id(version_id, global_child_index)
            children.append(ParsedChild(
                id=child_id,
                parent_id=parent.id,
                document_id=str(document_id) if document_id else None,
                version_id=str(version_id),
                child_text=child_text,
                child_index=global_child_index,
                page=parent.page_start,
                page_start=parent.page_start,
                page_end=parent.page_end,
                section=parent.section,
                element_type=parent.element_type,
                source_position={"parent_index": parent.parent_index, "child_index": global_child_index},
            ))
            global_child_index += 1

    return children


def parse_parent_child_chunks(
    file_bytes: bytes,
    filename: str,
    version_id: uuid.UUID | str,
    document_id: Optional[uuid.UUID | str] = None,
) -> ParsedDocument:
    """
    Full document parsing pipeline.

    1. Parse with Unstructured (preserving page/section/element metadata).
    2. Build parent chunks (~1024 tokens).
    3. Build child chunks (~128 tokens) inside each parent.
    4. Compute content hash.
    5. Return structured ParsedDocument.
    """
    if not file_bytes:
        raise ValueError("File content is empty.")

    content_hash = hashlib.md5(file_bytes).hexdigest()
    elements = _extract_elements(file_bytes, filename)

    if not elements or not any(el.text.strip() for el in elements):
        raise ValueError("Extracted text is empty or could not be parsed.")

    parents = _build_parents(elements, version_id)
    children = _build_children(parents, version_id, document_id)

    return ParsedDocument(
        parents=parents,
        children=children,
        content_hash=content_hash,
    )
