"""Small, page-aware evaluation dataset generation from READY documents."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.enums import DocumentStatus, VersionStatus
from app.database.models import Document, DocumentParent, DocumentVersion


@dataclass(frozen=True)
class SourceContext:
    document_id: str
    display_name: str
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None

    @property
    def pages(self) -> set[int]:
        if self.page_start is None:
            return set()
        return set(range(self.page_start, (self.page_end or self.page_start) + 1))


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start = cleaned.find("[")
    if start < 0:
        raise ValueError("Dataset generator did not return a JSON array")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, list):
        raise ValueError("Dataset generator returned a non-array JSON value")
    return [item for item in value if isinstance(item, dict)]


async def _load_source_contexts(
    db: AsyncSession,
    user_id: uuid.UUID,
    document_ids: Sequence[uuid.UUID] | None,
) -> list[SourceContext]:
    stmt = (
        select(DocumentParent, Document.id, Document.display_name)
        .join(DocumentVersion, DocumentVersion.id == DocumentParent.version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            Document.user_id == user_id,
            Document.status == DocumentStatus.READY.value,
            Document.current_version_id == DocumentVersion.id,
            DocumentVersion.status == VersionStatus.READY.value,
        )
        .order_by(Document.id, DocumentParent.parent_index)
    )
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    result = await db.execute(stmt)
    return [
        SourceContext(
            document_id=str(document_id),
            display_name=display_name,
            text=parent.text,
            page_start=parent.page_start,
            page_end=parent.page_end,
            section=parent.section,
        )
        for parent, document_id, display_name in result.all()
        if parent.text and parent.text.strip()
    ]


def _bounded_context(contexts: list[SourceContext], max_chars: int) -> list[SourceContext]:
    """Keep a deterministic, evenly distributed corpus for one generation call."""
    by_document: dict[str, list[SourceContext]] = {}
    for context in contexts:
        by_document.setdefault(context.document_id, []).append(context)

    selected: list[SourceContext] = []
    while by_document and sum(len(item.text) for item in selected) < max_chars:
        for document_id in list(by_document):
            context = by_document[document_id].pop(0)
            selected.append(context)
            if not by_document[document_id] or sum(len(item.text) for item in selected) >= max_chars:
                by_document.pop(document_id, None)
            if sum(len(item.text) for item in selected) >= max_chars:
                break
    return selected


def _prompt(contexts: list[SourceContext], count: int, offset: int) -> str:
    evidence = []
    for index, context in enumerate(contexts, start=1):
        page = (
            f"{context.page_start}-{context.page_end}"
            if context.page_start is not None and context.page_end not in {None, context.page_start}
            else str(context.page_start) if context.page_start is not None else "unpaginated"
        )
        evidence.append(
            f"[E{index}] document_id={context.document_id} page={page} "
            f"section={context.section or 'none'}\n{context.text[:8000]}"
        )
    categories = (
        "direct_lookup, multi_hop, cross_section, table_or_list, "
        "ambiguous, no_answer, multi_turn"
    )
    return f"""Create {count} evaluation cases from the supplied document evidence.
This is batch {offset // max(count, 1) + 1}. Use only the evidence below.
Generate answerable questions with concise ground-truth answers. Include a mix
of these categories when the evidence supports them: {categories}.
For every answerable case, return the exact document_id and page or page range
that supports the answer. Never invent a page number. For a page-less source,
return page as null and use its section when available. A no_answer case must
have an empty answer only when the supplied evidence cannot answer it.

Return JSON only as an array with this shape:
[{{
  "category": "direct_lookup",
  "question": "...",
  "ground_truth": "...",
  "sources": [{{"document_id": "...", "page": 3, "page_end": 3}}]
}}]

DOCUMENT EVIDENCE:
{chr(10).join(evidence)}"""


async def _generate_batch(
    contexts: list[SourceContext], count: int, batch_number: int
) -> list[dict[str, Any]]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for dataset generation")
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )
    prompt = _prompt(contexts, count, (batch_number - 1) * count)
    for attempt in range(3):
        try:
            return _parse_json_array(_response_text(await llm.ainvoke(prompt)))
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(settings.evaluation_request_delay_seconds * (attempt + 1))
    return []


def _validated_case(
    item: dict[str, Any], contexts: list[SourceContext], index: int
) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("ground_truth", "")).strip()
    if not question or not answer:
        return None

    context_by_document: dict[str, list[SourceContext]] = {}
    for context in contexts:
        context_by_document.setdefault(context.document_id, []).append(context)

    sources: list[dict[str, Any]] = []
    reference_contexts: list[str] = []
    for raw_source in item.get("sources", item.get("expected_sources", [])):
        if not isinstance(raw_source, dict):
            continue
        document_id = str(raw_source.get("document_id", ""))
        matching = context_by_document.get(document_id, [])
        if not matching:
            continue
        page = raw_source.get("page", raw_source.get("page_start"))
        page_end = raw_source.get("page_end", page)
        if page is not None:
            try:
                page = int(page)
                page_end = int(page_end) if page_end is not None else page
            except (TypeError, ValueError):
                continue
            if not any(
                page in context.pages and page_end in context.pages
                for context in matching
            ):
                continue
            source = {"document_id": document_id, "page": page}
            if page_end != page:
                source["page_end"] = page_end
            sources.append(source)
            reference_contexts.extend(
                context.text[:8000] for context in matching if page in context.pages
            )
        else:
            context = matching[0]
            source = {"document_id": document_id, "page": None}
            if context.section:
                source["section"] = context.section
            sources.append(source)
            reference_contexts.append(context.text[:8000])

    if not sources:
        return None
    return {
        "id": f"q-{index:03d}",
        "category": item.get("category", "direct_lookup"),
        "question": question,
        "selected_documents": sorted({source["document_id"] for source in sources}),
        "ground_truth": answer,
        "expected_sources": sources,
        "reference_contexts": list(dict.fromkeys(reference_contexts)),
        "conversation": [],
    }


async def generate_dataset(
    db: AsyncSession,
    user_id: uuid.UUID,
    document_ids: Sequence[uuid.UUID] | None = None,
    *,
    sample_count: int | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Generate a small validated dataset from the user's current READY docs."""
    settings = get_settings()
    requested_count = sample_count or settings.evaluation_sample_count
    requested_batch_size = batch_size or settings.evaluation_batch_size
    if not 1 <= requested_count <= 20:
        raise ValueError("sample_count must be between 1 and 20")
    if not 1 <= requested_batch_size <= 5:
        raise ValueError("batch_size must be between 1 and 5")
    contexts = await _load_source_contexts(db, user_id, document_ids)
    if not contexts:
        raise ValueError("No READY document parent content is available for evaluation")
    bounded = _bounded_context(contexts, settings.evaluation_context_chars)

    generated: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    batch_number = 1
    max_batches = ((requested_count + requested_batch_size - 1) // requested_batch_size) * 2
    while len(generated) < requested_count and batch_number <= max_batches:
        remaining = requested_count - len(generated)
        batch = await _generate_batch(
            bounded,
            min(requested_batch_size, remaining),
            batch_number,
        )
        for item in batch:
            case = _validated_case(item, bounded, len(generated) + 1)
            question_key = case["question"].casefold() if case else ""
            if case is not None and question_key not in seen_questions:
                seen_questions.add(question_key)
                generated.append(case)
            if len(generated) >= requested_count:
                break
        if not batch:
            break
        batch_number += 1
        if len(generated) < requested_count:
            await asyncio.sleep(settings.evaluation_request_delay_seconds)

    if len(generated) < min(5, requested_count):
        raise RuntimeError(
            f"Only {len(generated)} valid evaluation cases were generated; "
            "add more READY document content or retry generation"
        )
    return generated
