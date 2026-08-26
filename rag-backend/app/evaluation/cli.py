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
    ndcg_at_k,
    page_recall,
    recall_at_k,
    reciprocal_rank,
    string_relevancy,
)
from app.evaluation.langsmith_adapter import evaluate_with_langsmith
from app.evaluation.strategies import PROFILE_NAMES, RAGStrategy, get_strategy
from app.services.context_builder import ContextBuilder as PromptContextBuilder
from app.services.document_selection import resolve_selected_documents
from app.services.gemini_generation import GeminiAnswerStreamer
from app.services.intent_classifier import classify_intent
from app.services.retrieval import HybridRetriever, has_sufficient_evidence
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
    request_delay: float = 0.0,
) -> dict:
    retriever = HybridRetriever(strategy=strategy, embeddings=embeddings)
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
                    else await _generate_answer(
                        streamer,
                        package.to_llm_messages(),
                        retry_delay=request_delay,
                    )
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
    if args.case_limit is not None and not 1 <= args.case_limit <= 10:
        raise ValueError("case_limit must be between 1 and 10")
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
                request_delay=settings.evaluation_request_delay_seconds,
            )
        if args.auto:
            best_name = max(
                results,
                key=lambda name: (
                    results[name]["summary"].get("recall_at_10", 0.0) * 0.5
                    + results[name]["summary"].get("page_recall", 0.0) * 0.3
                    + results[name]["summary"].get("mrr", 0.0) * 0.2
                ),
            )
            selected = await _run_strategy(
                samples[: args.case_limit or settings.evaluation_case_limit],
                get_strategy(best_name),
                user_id,
                retrieval_only=False,
                use_langsmith=args.langsmith,
                embeddings=embedding_cache,
                langsmith_limit=args.case_limit or settings.evaluation_case_limit,
                langsmith_dataset=args.langsmith_dataset,
                reset=args.reset,
                request_delay=settings.evaluation_request_delay_seconds,
            )
            results = {
                "retrieval_comparison": results,
                "selected_strategy": best_name,
                "selected_evaluation": selected,
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
        print(f"selected_strategy: {results['selected_strategy']}")
        for name, result in results["retrieval_comparison"].items():
            print(f"{name}: {json.dumps(result['summary'], sort_keys=True)}")
        print(
            "selected_evaluation: "
            + json.dumps(results["selected_evaluation"]["summary"], sort_keys=True)
        )
        langsmith = results["selected_evaluation"].get("langsmith", {})
        if langsmith.get("experiment_url"):
            print(f"langsmith_experiment: {langsmith['experiment_url']}")
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
