"""Evaluation dataset schema and JSONL loading."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationSample:
    question: str
    expected_answer: str
    expected_document_ids: list[str]
    expected_pages: list[int]
    query_type: str

    @classmethod
    def from_dict(cls, value: dict) -> "EvaluationSample":
        return cls(
            question=value["question"],
            expected_answer=value.get("expected_answer", ""),
            expected_document_ids=list(value.get("expected_document_ids", [])),
            expected_pages=[int(page) for page in value.get("expected_pages", [])],
            query_type=value.get("query_type", "factual"),
        )


def load_jsonl(path: str | Path) -> list[EvaluationSample]:
    return [
        EvaluationSample.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
