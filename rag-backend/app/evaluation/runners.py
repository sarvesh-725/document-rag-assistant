"""Offline evaluation runners; never imported by normal chat processing."""

from dataclasses import asdict
from typing import Awaitable, Callable

from app.evaluation.datasets import EvaluationSample
from app.evaluation.metrics import (
    answer_token_f1,
    citation_precision,
    citation_recall,
    document_recall_at_k,
    evidence_support_rate,
    ndcg_at_k,
    no_answer_correct,
    page_recall,
    reciprocal_rank,
)
from app.evaluation.strategies import RAGStrategy


async def run_retrieval_evaluation(
    samples: list[EvaluationSample],
    retrieve: Callable[[EvaluationSample, RAGStrategy], Awaitable[list[dict]]],
    strategy: RAGStrategy,
) -> list[dict]:
    results = []
    for sample in samples:
        evidence = await retrieve(sample, strategy)
        documents = [item.get("document_id") for item in evidence]
        pages = [
            page
            for item in evidence
            for page in (
                item.get("page"),
                item.get("page_start"),
            )
            if page is not None
        ]
        results.append({
            "sample": asdict(sample),
            "strategy": asdict(strategy),
            "document_recall_at_5": (
                None
                if not sample.expected_document_ids
                else document_recall_at_k(documents, sample.expected_document_ids, 5)
            ),
            "document_recall_at_10": (
                None
                if not sample.expected_document_ids
                else document_recall_at_k(documents, sample.expected_document_ids, 10)
            ),
            "mrr": None if not sample.expected_document_ids else reciprocal_rank(documents, sample.expected_document_ids),
            "ndcg_at_10": None if not sample.expected_document_ids else ndcg_at_k(documents, sample.expected_document_ids, 10),
            "page_recall": None if not sample.expected_pages else page_recall(pages, sample.expected_pages),
            "retrieved": evidence,
        })
    return results


async def run_generation_evaluation(
    samples: list[EvaluationSample],
    generate: Callable[[EvaluationSample, RAGStrategy], Awaitable[dict]],
    strategy: RAGStrategy,
) -> list[dict]:
    results = []
    for sample in samples:
        result = await generate(sample, strategy)
        results.append({
            "sample": asdict(sample),
            "strategy": asdict(strategy),
            "answer_token_f1": answer_token_f1(result.get("answer", ""), sample.expected_answer),
            "evidence_support_rate": result.get("evidence_support_rate", evidence_support_rate(result.get("answer", ""), result.get("contexts", []))),
            "citation_precision": result.get("citation_precision", citation_precision(result.get("citations", []), sample.expected_sources)),
            "citation_recall": result.get("citation_recall", citation_recall(result.get("citations", []), sample.expected_sources)),
            "no_answer_correct": no_answer_correct(
                result.get("answer", ""),
                sample.query_type,
                sample.expected_sources,
                bool(result.get("evidence_available", False)),
            ),
            "latency_ms": result.get("latency_ms"),
            "token_usage": result.get("token_usage"),
        })
    return results
