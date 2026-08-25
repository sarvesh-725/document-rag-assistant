"""Offline evaluation runners; never imported by normal chat processing."""

from dataclasses import asdict
from typing import Awaitable, Callable

from app.evaluation.datasets import EvaluationSample
from app.evaluation.metrics import (
    ndcg_at_k,
    page_recall,
    recall_at_k,
    reciprocal_rank,
    string_relevancy,
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
            "recall_at_5": recall_at_k(documents, sample.expected_document_ids, 5),
            "recall_at_10": recall_at_k(documents, sample.expected_document_ids, 10),
            "mrr": reciprocal_rank(documents, sample.expected_document_ids),
            "ndcg_at_10": ndcg_at_k(documents, sample.expected_document_ids, 10),
            "page_recall": page_recall(pages, sample.expected_pages),
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
            "answer_relevancy": string_relevancy(result.get("answer", ""), sample.expected_answer),
            "faithfulness": result.get("faithfulness"),
            "context_precision": result.get("context_precision"),
            "context_recall": result.get("context_recall"),
            "latency_ms": result.get("latency_ms"),
            "token_usage": result.get("token_usage"),
        })
    return results
