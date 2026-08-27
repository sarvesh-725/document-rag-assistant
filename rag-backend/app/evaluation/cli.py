"""Small command-line evaluator for comparing the existing RAG strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.database.connection import AsyncSessionLocal, engine
from app.evaluation.datasets import EvaluationSample, load_jsonl
from app.evaluation.dataset_generation import generate_dataset
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
from app.evaluation.langsmith_adapter import evaluate_with_langsmith
from app.evaluation.strategies import PROFILE_NAMES, RAGStrategy, get_strategy
from app.services.context_builder import ContextBuilder as PromptContextBuilder
from app.services.document_selection import resolve_selected_documents
from app.services.gemini_generation import GeminiAnswerStreamer
from app.services.intent_classifier import classify_intent
from app.services.retrieval import (
    AsyncRateLimiter,
    HybridRetriever,
    Reranker,
    has_sufficient_evidence,
)
from app.services import rag_engine


NO_GROUNDING_RESPONSE = (
    "The selected documents do not contain enough information to answer this question."
)


class QueryEmbeddingCache:
    """Reuse one Gemini query embedding while comparing multiple profiles."""

    def __init__(self):
        self._values: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        if query not in self._values:
            self._values[query] = rag_engine._get_embeddings().embed_query(query)
        return self._values[query]


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
        "document_recall_at_5",
        "document_recall_at_10",
        "mrr",
        "ndcg_at_10",
        "page_recall",
        "citation_precision",
        "citation_recall",
        "answer_token_f1",
        "evidence_support_rate",
        "no_answer_correct",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "latency_ms",
    )
    summary = {"case_count": len(cases)}
    for field in numeric_fields:
        values = [case[field] for case in cases if isinstance(case.get(field), (int, float))]
        if values:
            summary[field] = round(mean(values), 4)
    return summary


def _answer_samples(samples: list[EvaluationSample], limit: int) -> list[EvaluationSample]:
    """Keep no-answer cases in the bounded answer set."""
    no_answer = [sample for sample in samples if sample.query_type == "no_answer"]
    answerable = [sample for sample in samples if sample.query_type != "no_answer"]
    return (no_answer + answerable)[:limit]


def _retrieval_screen_key(result: dict) -> tuple[float, float, float, float, float]:
    summary = result["summary"]
    return (
        summary.get("document_recall_at_10", 0.0),
        summary.get("citation_recall", 0.0),
        summary.get("page_recall", 0.0),
        summary.get("mrr", 0.0),
        -summary.get("retrieval_latency_ms", float("inf")),
    )


async def _generate_answer(
    streamer: GeminiAnswerStreamer,
    messages: list[dict[str, str]],
    retry_delay: float = 0.0,
) -> str:
    for attempt in range(3):
        try:
            pieces: list[str] = []
            async for piece in streamer.stream(messages):
                pieces.append(piece)
            return "".join(pieces)
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(retry_delay * (attempt + 1))
    return ""


async def _run_strategy(
    samples: list[EvaluationSample],
    strategy: RAGStrategy,
    user_id: uuid.UUID,
    *,
    retrieval_only: bool,
    use_langsmith: bool,
    embeddings: QueryEmbeddingCache | None = None,
    langsmith_limit: int | None = None,
    langsmith_dataset: str | None = None,
    reset: bool = False,
    rerank_rate_limiter: AsyncRateLimiter | None = None,
    reranker: Reranker | None = None,
    request_delay: float = 0.0,
) -> dict:
    retriever = HybridRetriever(
        strategy=strategy,
        embeddings=embeddings,
        rerank_rate_limiter=rerank_rate_limiter,
        reranker=reranker if strategy.reranker else None,
    )
    prompt_builder = PromptContextBuilder()
    streamer = GeminiAnswerStreamer()
    cases: list[dict] = []

    async with AsyncSessionLocal() as db:
        for sample_index, sample in enumerate(samples):
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
            retrieval_started = time.perf_counter()
            retrieval = await retriever.retrieve(
                sample.question,
                user_id,
                version_ids,
                db,
            )
            retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)
            analysis = classify_intent(sample.question)
            package = prompt_builder.build(
                query_analysis=analysis,
                conversation_summary="",
                recent_messages=sample.conversation,
                retrieved_evidence=retrieval.context,
                current_question=sample.question,
            )
            evidence_available = has_sufficient_evidence(retrieval)
            is_no_answer = sample.query_type == "no_answer"
            generation_started = time.perf_counter()
            if is_no_answer:
                answer = "" if retrieval_only else NO_GROUNDING_RESPONSE
            elif evidence_available:
                answer = (
                    ""
                    if retrieval_only
                    else await _generate_answer(
                        streamer,
                        package.to_llm_messages(),
                        retry_delay=request_delay,
                    )
                )
            else:
                answer = "" if retrieval_only else NO_GROUNDING_RESPONSE
            generation_latency_ms = (
                None
                if retrieval_only
                else round((time.perf_counter() - generation_started) * 1000, 2)
            )

            citations = [] if is_no_answer else package.citations
            retrieved_documents = [
                candidate.document_id for candidate in retrieval.reranked_candidates
            ]
            contexts = [block.text for block in retrieval.context]
            retrieved_candidates = [
                {
                    "identity": candidate.identity,
                    "document_id": candidate.document_id,
                    "version_id": candidate.version_id,
                    "parent_id": candidate.parent_id,
                    "chunk_id": candidate.chunk_id,
                    "page_start": candidate.page_start,
                    "page_end": candidate.page_end,
                    "dense_rank": candidate.dense_rank,
                    "bm25_rank": candidate.bm25_rank,
                    "fusion_score": candidate.fusion_score,
                    "rerank_rank": candidate.rerank_rank,
                    "rerank_score": candidate.rerank_score,
                }
                for candidate in retrieval.reranked_candidates
            ]
            case = {
                "id": sample.sample_id,
                "question": sample.question,
                "category": sample.query_type,
                "reference": sample.expected_answer,
                "selected_document_ids": sample.selected_document_ids,
                "expected_document_ids": sample.expected_document_ids,
                "expected_pages": sample.expected_pages,
                "expected_sources": sample.expected_sources,
                "reference_contexts": sample.reference_contexts,
                "contexts": contexts,
                "selected_context": package.evidence,
                "answer": answer,
                "retrieved_document_ids": retrieved_documents,
                "retrieved_candidate_ids": [
                    candidate["identity"] for candidate in retrieved_candidates
                ],
                "retrieved_candidates": retrieved_candidates,
                "retrieved_pages": _page_numbers(retrieved_candidates),
                "citations": citations,
                "grounded": evidence_available,
                "evidence_available": evidence_available,
                "document_recall_at_5": (
                    None
                    if not sample.expected_document_ids
                    else document_recall_at_k(
                        retrieved_documents, sample.expected_document_ids, 5
                    )
                ),
                "document_recall_at_10": (
                    None
                    if not sample.expected_document_ids
                    else document_recall_at_k(
                        retrieved_documents, sample.expected_document_ids, 10
                    )
                ),
                "mrr": (
                    None
                    if not sample.expected_document_ids
                    else reciprocal_rank(retrieved_documents, sample.expected_document_ids)
                ),
                "ndcg_at_10": (
                    None
                    if not sample.expected_document_ids
                    else ndcg_at_k(
                        retrieved_documents, sample.expected_document_ids, 10
                    )
                ),
                "page_recall": (
                    None
                    if not sample.expected_pages
                    else page_recall(_page_numbers(citations), sample.expected_pages)
                ),
                "citation_precision": citation_precision(citations, sample.expected_sources),
                "citation_recall": citation_recall(citations, sample.expected_sources),
                "answer_token_f1": (
                    None
                    if retrieval_only or is_no_answer
                    else answer_token_f1(answer, sample.expected_answer)
                ),
                "evidence_support_rate": (
                    None
                    if retrieval_only or is_no_answer
                    else evidence_support_rate(answer, contexts)
                ),
                "no_answer_correct": (
                    None
                    if retrieval_only
                    else no_answer_correct(
                        answer,
                        sample.query_type,
                        sample.expected_sources,
                        evidence_available,
                    )
                ),
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
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
            if request_delay and sample_index + 1 < len(samples) and not retrieval_only:
                await asyncio.sleep(request_delay)

    langsmith_output = {"cases": [], "summary": {}}
    if use_langsmith:
        langsmith_output = await evaluate_with_langsmith(
            cases[:langsmith_limit] if langsmith_limit else cases,
            strategy_name=strategy.name,
            dataset_name=langsmith_dataset,
            reset=reset,
        )
        for case, langsmith_result in zip(cases, langsmith_output["cases"]):
            case["langsmith"] = langsmith_result

    return {
        "strategy": strategy.__dict__,
        "summary": _summary(cases),
        "langsmith": langsmith_output,
        "cases": cases,
    }


async def _run(args: argparse.Namespace) -> None:
    user_id = uuid.UUID(args.user_id)
    settings = get_settings()
    if args.case_limit is not None and not 1 <= args.case_limit <= 12:
        raise ValueError("case_limit must be between 1 and 12")
    generated_dataset = False
    if args.auto:
        document_ids = [uuid.UUID(value) for value in args.document_id]
        async with AsyncSessionLocal() as db:
            generated = await generate_dataset(
                db,
                user_id,
                document_ids or None,
                sample_count=args.sample_count or settings.evaluation_sample_count,
                batch_size=args.batch_size or settings.evaluation_batch_size,
            )
        output_dataset = Path(args.generated_dataset)
        output_dataset.parent.mkdir(parents=True, exist_ok=True)
        output_dataset.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in generated) + "\n",
            encoding="utf-8",
        )
        args.dataset = str(output_dataset)
        generated_dataset = True

    samples = load_jsonl(args.dataset)
    if not samples:
        raise ValueError(f"Evaluation dataset is empty: {args.dataset}")
    missing_scope = [sample.sample_id for sample in samples if not sample.selected_document_ids]
    if missing_scope:
        raise ValueError(
            "Every evaluation case must provide selected_documents/retrieval_scope; "
            f"missing: {', '.join(missing_scope)}"
        )
    names = list(PROFILE_NAMES) if args.all_strategies or args.auto else [args.strategy]
    if args.langsmith and args.retrieval_only:
        raise ValueError("--langsmith requires generated answers; remove --retrieval-only")
    if args.auto and args.retrieval_only:
        raise ValueError("--auto includes the selected strategy answer evaluation; remove --retrieval-only")
    if args.langsmith and args.all_strategies and not args.auto:
        raise ValueError(
            "Run retrieval-only comparison first, then run --langsmith for the selected strategy"
        )

    results = {}
    embedding_cache = QueryEmbeddingCache()
    rerank_rate_limiter = AsyncRateLimiter(
        settings.evaluation_cohere_min_interval_seconds
    )
    reranker = Reranker(rate_limiter=rerank_rate_limiter)
    try:
        for name in names:
            evaluation_samples = (
                samples
                if args.retrieval_only or args.auto
                else samples[: args.case_limit or settings.evaluation_case_limit]
            )
            results[name] = await _run_strategy(
                evaluation_samples,
                get_strategy(name),
                user_id,
                retrieval_only=args.retrieval_only or args.auto,
                use_langsmith=args.langsmith and not args.auto,
                embeddings=embedding_cache,
                langsmith_dataset=args.langsmith_dataset,
                reset=args.reset,
                rerank_rate_limiter=rerank_rate_limiter,
                reranker=reranker,
                request_delay=settings.evaluation_request_delay_seconds,
            )
        if args.auto:
            top_names = sorted(
                results, key=lambda name: _retrieval_screen_key(results[name]), reverse=True
            )[:2]
            answer_results = {}
            answer_samples = _answer_samples(
                samples, args.case_limit or settings.evaluation_case_limit
            )
            for index, name in enumerate(top_names):
                answer_results[name] = await _run_strategy(
                    answer_samples,
                    get_strategy(name),
                    user_id,
                    retrieval_only=False,
                    use_langsmith=args.langsmith,
                    embeddings=embedding_cache,
                    langsmith_limit=len(answer_samples),
                    langsmith_dataset=args.langsmith_dataset,
                    reset=args.reset and index == 0,
                    rerank_rate_limiter=rerank_rate_limiter,
                    reranker=reranker,
                    request_delay=settings.evaluation_request_delay_seconds,
                )
            results = {
                "retrieval_comparison": results,
                "top_strategies": top_names,
                "answer_comparison": answer_results,
            }
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
                "generated_dataset": generated_dataset,
                "strategies": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved evaluation results to {output_path}")
    if args.auto:
        print(f"top_strategies: {', '.join(results['top_strategies'])}")
        for name, result in results["retrieval_comparison"].items():
            print(f"{name}: {json.dumps(result['summary'], sort_keys=True)}")
        for name, result in results["answer_comparison"].items():
            print(f"{name}_answers: {json.dumps(result['summary'], sort_keys=True)}")
            langsmith = result.get("langsmith", {})
            if langsmith.get("experiment_url"):
                print(f"{name}_langsmith_experiment: {langsmith['experiment_url']}")
    else:
        for name, result in results.items():
            print(f"{name}: {json.dumps(result['summary'], sort_keys=True)}")
            langsmith = result.get("langsmith", {})
            if langsmith.get("experiment_url"):
                print(f"{name}_langsmith_experiment: {langsmith['experiment_url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/dataset.jsonl")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--strategy", default="hybrid_parent_rerank")
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument("--generated-dataset", default="evaluation/generated_dataset.jsonl")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--langsmith-dataset")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
