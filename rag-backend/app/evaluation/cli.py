"""Small command-line evaluator for comparing the existing RAG strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from statistics import mean

from app.database.connection import AsyncSessionLocal, engine
from app.evaluation.datasets import EvaluationSample, load_jsonl
from app.evaluation.metrics import (
    ndcg_at_k,
    page_recall,
    recall_at_k,
    reciprocal_rank,
    string_relevancy,
)
from app.evaluation.ragas_adapter import evaluate_with_ragas
from app.evaluation.strategies import PROFILE_NAMES, RAGStrategy, get_strategy
from app.services.context_builder import ContextBuilder as PromptContextBuilder
from app.services.document_selection import resolve_selected_documents
from app.services.gemini_generation import GeminiAnswerStreamer
from app.services.intent_classifier import classify_intent
from app.services.retrieval import HybridRetriever, has_sufficient_evidence


NO_GROUNDING_RESPONSE = (
    "The selected documents do not contain enough information to answer this question."
)


def _page_numbers(citations: list[dict]) -> list[int]:
    pages: set[int] = set()
    for citation in citations:
        start = citation.get("page_start", citation.get("page"))
        end = citation.get("page_end", start)
        if start is None:
            continue
        pages.update(range(int(start), int(end or start) + 1))
    return sorted(pages)


def _summary(cases: list[dict]) -> dict:
    numeric_fields = (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "page_recall",
        "answer_relevancy",
        "latency_ms",
    )
    summary = {"case_count": len(cases)}
    for field in numeric_fields:
        values = [case[field] for case in cases if isinstance(case.get(field), (int, float))]
        if values:
            summary[field] = round(mean(values), 4)
    return summary


async def _generate_answer(
    streamer: GeminiAnswerStreamer, messages: list[dict[str, str]]
) -> str:
    pieces: list[str] = []
    async for piece in streamer.stream(messages):
        pieces.append(piece)
    return "".join(pieces)


async def _run_strategy(
    samples: list[EvaluationSample],
    strategy: RAGStrategy,
    user_id: uuid.UUID,
    *,
    retrieval_only: bool,
    use_ragas: bool,
) -> dict:
    retriever = HybridRetriever(strategy=strategy)
    prompt_builder = PromptContextBuilder()
    streamer = GeminiAnswerStreamer()
    cases: list[dict] = []

    async with AsyncSessionLocal() as db:
        for sample in samples:
            started = time.perf_counter()
            selected_ids = sample.selected_document_ids
            resolved = (
                await resolve_selected_documents(
                    db,
                    authenticated_user_id=user_id,
                    selected_document_ids=selected_ids,
                )
                if selected_ids
                else []
            )
            version_ids = [document.version_id for document in resolved]
            retrieval = await retriever.retrieve(
                sample.question,
                user_id,
                version_ids,
                db,
            )
            analysis = classify_intent(sample.question)
            package = prompt_builder.build(
                query_analysis=analysis,
                conversation_summary="",
                recent_messages=sample.conversation,
                retrieved_evidence=retrieval.context,
                current_question=sample.question,
            )
            if has_sufficient_evidence(retrieval):
                answer = (
                    ""
                    if retrieval_only
                    else await _generate_answer(streamer, package.to_llm_messages())
                )
            else:
                answer = "" if retrieval_only else NO_GROUNDING_RESPONSE

            citations = package.citations
            retrieved_documents = [
                candidate.document_id for candidate in retrieval.reranked_candidates
            ]
            contexts = [block.text for block in retrieval.context]
            case = {
                "id": sample.sample_id,
                "question": sample.question,
                "category": sample.query_type,
                "reference": sample.expected_answer,
                "reference_contexts": sample.reference_contexts,
                "contexts": contexts,
                "answer": answer,
                "retrieved_document_ids": retrieved_documents,
                "citations": citations,
                "grounded": has_sufficient_evidence(retrieval),
                "recall_at_5": recall_at_k(
                    retrieved_documents, sample.expected_document_ids, 5
                ),
                "recall_at_10": recall_at_k(
                    retrieved_documents, sample.expected_document_ids, 10
                ),
                "mrr": reciprocal_rank(
                    retrieved_documents, sample.expected_document_ids
                ),
                "ndcg_at_10": ndcg_at_k(
                    retrieved_documents, sample.expected_document_ids, 10
                ),
                "page_recall": page_recall(
                    _page_numbers(citations), sample.expected_pages
                ),
                "answer_relevancy": (
                    None
                    if retrieval_only
                    else string_relevancy(answer, sample.expected_answer)
                ),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "retrieval_counts": {
                    "dense": len(retrieval.dense_candidates),
                    "bm25": len(retrieval.bm25_candidates),
                    "fused": len(retrieval.fused_candidates),
                    "reranked": len(retrieval.reranked_candidates),
                    "context": len(retrieval.context),
                },
            }
            cases.append(case)

    ragas_output = {"cases": [], "summary": {}}
    if use_ragas:
        ragas_output = evaluate_with_ragas(cases)
        for case, ragas_result in zip(cases, ragas_output["cases"]):
            case["ragas"] = ragas_result

    return {
        "strategy": strategy.__dict__,
        "summary": _summary(cases),
        "ragas_summary": ragas_output["summary"] if use_ragas else {},
        "cases": cases,
    }


async def _run(args: argparse.Namespace) -> None:
    samples = load_jsonl(args.dataset)
    if not samples:
        raise ValueError(f"Evaluation dataset is empty: {args.dataset}")
    user_id = uuid.UUID(args.user_id)
    names = list(PROFILE_NAMES) if args.all_strategies else [args.strategy]
    if args.ragas and args.retrieval_only:
        raise ValueError("--ragas requires generated answers; remove --retrieval-only")

    results = {}
    try:
        for name in names:
            results[name] = await _run_strategy(
                samples,
                get_strategy(name),
                user_id,
                retrieval_only=args.retrieval_only,
                use_ragas=args.ragas,
            )
    finally:
        await engine.dispose()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation-{int(time.time())}.json"
    output_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "user_id": str(user_id),
                "strategies": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved evaluation results to {output_path}")
    for name, result in results.items():
        print(f"{name}: {json.dumps(result['summary'], sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--strategy", default="hybrid_parent_rerank")
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
