"""Evaluation dataset schema and JSONL loading."""

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationSample:
    question: str
    expected_answer: str
    expected_document_ids: list[str]
    expected_pages: list[int]
    query_type: str
    sample_id: str = ""
    selected_document_ids: list[str] = field(default_factory=list)
    expected_sources: list[dict] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict) -> "EvaluationSample":
        question = value["question"]
        expected_sources = list(value.get("expected_sources", []))
        expected_document_ids = list(value.get("expected_document_ids", []))
        expected_pages = [int(page) for page in value.get("expected_pages", [])]
        for source in expected_sources:
            if not isinstance(source, dict):
                continue
            document_id = source.get("document_id")
            if document_id and document_id not in expected_document_ids:
                expected_document_ids.append(str(document_id))
            if source.get("page") is not None:
                page = int(source["page"])
                page_end = int(source.get("page_end", page))
                expected_pages.extend(
                    candidate
                    for candidate in range(page, page_end + 1)
                    if candidate not in expected_pages
                )
            if source.get("page_start") is not None:
                page = int(source["page_start"])
                page_end = int(source.get("page_end", page))
                expected_pages.extend(
                    candidate
                    for candidate in range(page, page_end + 1)
                    if candidate not in expected_pages
                )
        selected_document_ids = list(
            value.get("selected_documents", value.get("selected_document_ids", []))
        )
        if not selected_document_ids:
            selected_document_ids = list(expected_document_ids)
        return cls(
            question=question,
            expected_answer=value.get("ground_truth", value.get("expected_answer", "")),
            expected_document_ids=expected_document_ids,
            expected_pages=expected_pages,
            query_type=value.get("category", value.get("query_type", "factual")),
            sample_id=str(
                value.get("id")
                or "q-"
                + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
            ),
            selected_document_ids=selected_document_ids,
            expected_sources=expected_sources,
            reference_contexts=list(value.get("reference_contexts", [])),
            conversation=list(value.get("conversation", [])),
        )


def load_jsonl(path: str | Path) -> list[EvaluationSample]:
    return [
        EvaluationSample.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
