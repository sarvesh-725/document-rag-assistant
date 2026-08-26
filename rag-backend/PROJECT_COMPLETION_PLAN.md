# Document RAG Assistant: Completion Plan

## Purpose

This document is the development reference for completing the project without
expanding it into a large production platform. The target is an interview-ready,
production-style MVP with a coherent architecture, deliberate failure handling,
measurable RAG quality, and documentation that matches the implementation.

The project should be considered complete when the main workflow is reliable,
the deployment responsibilities are clear, the retrieval quality is measured,
source citations are correct, and every important resume claim can be demonstrated
or explained honestly. After the final phase, the developer will perform one
complete manual user-flow test and will not expand the project further.

## Scope Guardrails

Keep the existing architecture:

```text
Next.js -> FastAPI -> PostgreSQL
                    -> Outbox -> Publisher -> Redis -> Taskiq Worker
                    -> Qdrant retrieval -> Gemini generation -> SSE response
```

Do not add agents, web search, knowledge graphs, Kafka, Kubernetes, another
vector database, another LLM provider, or a new job-processing framework.

Do not claim exactly-once delivery. The intended reliability model is:

```text
at-least-once dispatch + idempotent processing
```

Do not introduce a new queue, vector database, sparse-search service, LLM
provider, orchestration framework, or evaluation platform. Prefer the existing
Python, PostgreSQL, Redis, Qdrant, Taskiq, Gemini, Cohere, and JSON-based files.

The system must remain fast for its intended MVP scale. Any retrieval strategy
must use explicit candidate and context limits, and any quality improvement must
be justified by evaluation results rather than added speculatively.

## Developer Checkpoints

The development workflow is intentionally limited to the following actions:

- The assistant implements one phase and performs targeted verification relevant to that phase.
- The developer reviews the phase diff and the short completion report, then pushes the successful phase to Git.
- The developer uploads one or more evaluation documents before Phase 2. The assistant generates a small page-linked dataset; the developer may review the generated cases but does not need to hand-write them.
- The developer does not perform the full manual application test between phases.
- After Phase 3, the developer starts the complete system and manually tests every user-facing flow once. This is the final project validation.

## Current Baseline

The following core capabilities are already present:

- JWT authentication, password hashing, login throttling, and ownership checks
- PostgreSQL persistence for users, documents, versions, conversations, query runs, and outbox events
- Next.js chat and upload interface with typed SSE token, source, and status events
- Asynchronous document processing with Taskiq, Redis, Unstructured, chunking, Gemini, and Qdrant
- Dense retrieval, in-process BM25, reciprocal-rank fusion, parent-context expansion, and optional Cohere reranking
- User-global documents, soft deletion, ingestion state tracking, and asynchronous cleanup
- Document/version scoping, deterministic vector upserts, chat-request idempotency, and source citations
- Grounding checks, no-evidence handling, bounded conversation context, and basic chitchat routing

The following areas are incomplete or need deliberate decisions:

- API, worker, publisher, and dependency startup are not yet packaged as one documented local deployment
- Outbox rows need transactional claiming to prevent concurrent duplicate dispatch
- Configuration is inconsistent in some storage and Qdrant paths
- Version records exist, but the product contract for uploading and managing versions is not fully consistent
- BM25 is request-time and in-process; it is not a persistent large-scale sparse index
- Conversation summaries are deterministic aggregation/truncation, not semantic LLM summaries
- Completed chat replay does not fully reproduce the original citation stream
- RAGAS evaluation is not yet operational
- Health/readiness behavior, frontend startup documentation, and project documentation need completion

The current hybrid retrieval path must be reviewed: BM25 is currently derived
from a request-time corpus and must not be treated as a second pass over only
the dense results. The completed design will run dense and lexical retrieval as
independent candidate-generation paths and fuse their rankings with RRF.

## Phase 1: Reliability and Local Operations

### Objective

Make the full system startable, configurable, secure, and safe under ordinary
failure and retry conditions.

### Work Items

1. Remove credentials from tracked or shared configuration, ensure real
   environment files are ignored, and provide an `.env.example` containing only
   empty placeholders. If any real key was exposed, credential rotation is a
   security prerequisite before the final manual run, not an application-code
   task.
2. Centralize runtime configuration for PostgreSQL, Redis, Qdrant, storage,
   upload limits, JWT settings, Gemini, and Cohere. Remove duplicated or
   hardcoded runtime paths and endpoints.
3. Provide a minimal Docker Compose deployment for PostgreSQL, Redis, Qdrant,
   the API, the Taskiq worker, and the outbox publisher. Run the Next.js
   frontend locally with `npm run dev`; document its connection to the API.
   Keep Gemini, Cohere, and Unstructured provider requirements explicit.
4. Add `GET /health` for process liveness and `GET /ready` for required local
   dependency readiness. External model providers should not prevent the API
   process from starting.
5. Change outbox publishing to claim rows transactionally with PostgreSQL row
   locking, such as `FOR UPDATE SKIP LOCKED`. Preserve retryability when
   publishing fails and rely on idempotent worker operations after a crash.
6. Verify the `available_at` schema/model contract on fresh and existing
   databases by applying the Alembic migration before starting the API.
7. Make duplicate chat requests safe at the request level: a completed request
   replays its answer and citations, while an active request is not regenerated.
8. Fix local database initialization and remove stale startup comments or
   instructions that no longer match the current application.

### Completion Criteria

- One documented startup procedure starts the API, worker, publisher,
  PostgreSQL, Redis, and Qdrant, while the frontend has a separate local
  `npm run dev` command.
- API, publisher, and worker responsibilities are separate and visible.
- Two publishers cannot claim the same outbox row concurrently.
- A failed publish remains retryable.
- Repeating a task does not create duplicate vector records or duplicate chat answers.
- Configuration comes from one settings source.
- Health and readiness behavior is documented.
- No real credentials are committed.

### Developer Gate

The developer reviews the changed files and the architecture explanation, then
pushes Phase 1. No complete manual application test is required yet. Phase 1
verification is limited to startup, health/readiness, migration, and outbox
behavior; it is not permission to build or repair a general test suite.

### Interview Result

The upload explanation becomes:

```text
HTTP request -> database state and outbox event -> committed transaction
-> publisher -> Redis -> Taskiq worker -> parse/chunk/embed/index
```

The reliability explanation becomes: durable outbox intent, at-least-once
dispatch, idempotent processing, and retryable failures.

## Phase 2: RAG Quality and Evaluation

### Objective

Measure retrieval and generation quality, then make only targeted improvements
based on observed failures.

### Work Items

1. Add a simple, editable JSONL evaluation dataset template and an automatic
   generator. The developer supplies documents only; generated cases should
   support the following fields:

```json
{
  "id": "q-001",
  "category": "direct_lookup",
  "question": "...",
  "selected_documents": ["document-id"],
  "ground_truth": "...",
  "expected_sources": [{"document_id": "...", "page": 3}],
  "reference_contexts": [],
  "conversation": []
}
```

   Generate 5-20 cases, with 8-10 as the default, across direct lookup,
   multi-hop, cross-section, table/list extraction, ambiguous, no-answer, and
   multi-turn questions when the document evidence supports them. Generate in
   batches of at most 5 requests and apply a delay between Gemini requests.
   `conversation` is only needed for multi-turn cases.
2. Complete the RAGAS evaluation path for context precision, context recall,
   faithfulness, and response relevancy. Store dataset, strategy configuration,
   raw outputs, and summary results so runs are repeatable.
3. Add only a thin strategy configuration layer, not a generalized experiment
   framework. The developer should be able to select profiles such as:

```text
dense_only
hybrid_parallel
hybrid_parallel_parent
hybrid_parallel_parent_rerank
```

   Profiles may vary top-k, candidate limits, RRF weights, parent expansion,
   reranking, and context budgets. A simple command or configuration value must
   compare profiles using the same dataset. Do not add an experiment registry,
   parameter sweeps, result database, dashboard, or profile inheritance.
4. Correct the hybrid retrieval flow without introducing new technology. Dense
   Qdrant retrieval and lexical BM25 retrieval must independently produce
   candidate lists from the selected document corpus. RRF must combine their
   ranks only after both paths complete:

```text
dense candidates  ----\
                       -> RRF -> parent expansion -> optional rerank -> context
BM25 candidates   ----/
```

   BM25 must not be applied only to the dense-result list. Keep it in-process
   for this MVP, bound the lexical corpus/candidate count, and document that a
   persistent sparse index would be a future scaling step.
5. Record retrieved candidates, selected parent context, final citations, final
   answer, latency, and failure category for every evaluation case.
6. Use evaluation results to tune only the necessary layer:

```text
low context recall   -> candidate count, parent expansion, or chunking
low context precision -> filtering, fusion, or reranking
low faithfulness     -> grounding prompt and evidence threshold
low answer relevance -> prompt structure and conversation context
```

7. Preserve explicit no-evidence behavior: when the selected documents do not
   provide enough support, the system must refuse to invent an answer or source.
8. Keep Cohere reranking optional and retain the fused-ranking fallback when the
   provider is unavailable.
9. Keep conversation context deterministic unless evaluation demonstrates that
   it is insufficient. Describe it as rolling conversation context rather than
   claiming semantic summarization.
10. Enforce a citation contract. Page numbers must come from parser metadata and
   must be preserved through child chunks, parent expansion, reranking, and SSE
   output. Cite the page or page range of the selected evidence, deduplicate
   repeated sources, and never infer or invent a page number. For page-less
   formats, show section or element metadata instead of a false page number.
   Fix preservation and presentation of metadata already available from the
   parser only. Do not redesign parsing or add a separate document-layout
   extraction system.
11. Include citation cases in the generated dataset, using only page numbers
    present in parser metadata for PDFs containing paragraphs, tables, and
    multi-page content. Reject generated sources that cannot be validated.

### Completion Criteria

- RAGAS produces repeatable metrics from a saved, generated sample dataset.
- The worst retrieval and generation failures have been manually inspected.
- Candidate limits and no-evidence behavior are tested.
- At least one targeted improvement is justified by evaluation results.
- Dense-only versus parallel hybrid strategies can be compared without code edits or regenerating the dataset.
- Returned citations match the evidence page/page range in the evaluation data.
- The README reports known quality and scalability tradeoffs honestly.

### Interview Result

The RAG explanation becomes:

```text
dense Qdrant search + independent lexical BM25 -> rank fusion -> parent expansion
-> optional reranking -> context budget -> grounded Gemini answer
```

The project can discuss quality using measurements instead of claiming that
the model is simply accurate.

### Developer Gate

Before Phase 2 begins, the developer uploads one or more documents and waits for
them to become READY. The automatic evaluator generates and saves the cases,
then compares retrieval profiles. The developer reviews the comparison and the
selected answer/RAGAS result, then pushes Phase 2. The developer does not need
to generate or hand-write the dataset.

## Phase 3: Product Contract and Interview Polish

### Objective

Remove remaining contradictions between the data model, UI behavior, resume
language, and project documentation.

### Work Items

1. Choose and document one versioning contract. The recommended small scope is
   immutable ingestion snapshots with retrieval against the current READY
   version. Do not add comparison, rollback, or historical-chat screens unless
   the existing use case requires them.
2. If subsequent uploads are intended to update one logical document, add a
   small version-upload path. Otherwise describe the current behavior as
   document ingestion snapshots rather than full version management.
3. Document ownership isolation, soft deletion, cleanup, query-run snapshots,
   deterministic IDs, and request-level idempotency as explicit design choices.
4. Keep basic chitchat routing as a small intent branch that avoids unnecessary
   retrieval for obvious conversational messages. Do not turn it into an agent.
5. Update the README with architecture, request flows, startup instructions,
   environment variables, failure handling, evaluation results, and known
   tradeoffs.
6. Remove stale boilerplate metadata and instructions from the frontend and
   backend documentation.
7. Define the final manual verification flows:

```text
register/login
upload -> process -> READY
selected-document question -> streamed answer and sources
no-evidence question -> grounded refusal
document deletion -> cleanup
same chat request twice -> one logical answer
chitchat -> no unnecessary document retrieval
```

8. Prepare the final manual-test checklist so it covers every user-facing
   action: authentication, upload, duplicate filenames, ingestion status,
   document selection, grounded questions, no-evidence questions, multi-turn
   chat, citations/page numbers, chitchat, deletion, retries, and logout.

### Completion Criteria

- The versioning behavior can be explained in one precise paragraph.
- Resume wording matches actual behavior and fallback paths.
- README startup and architecture instructions work from a clean environment,
  with the frontend run locally rather than containerized.
- The critical user flows have been manually demonstrated.
- Known tradeoffs are documented rather than hidden.

### Developer Gate

The developer reviews the final README, resume wording, evaluation summary, and
known limitations, then pushes Phase 3. Only after this push does the developer
run the complete manual application test described above. That manual test is
the final validation and is not a request for another development phase.

## Definition of Done

Call the project complete for its intended MVP/interview scope when all of the
following are true:

- The API, publisher, worker, PostgreSQL, Redis, and Qdrant startup path is documented and repeatable.
- Upload, outbox, ingestion, indexing, retrieval, generation, and SSE response flows are explainable end to end.
- Outbox dispatch is retryable and protected against concurrent row claiming.
- Duplicate task execution is safe; duplicate chat requests do not start duplicate generation.
- User and document ownership boundaries are enforced.
- Dense retrieval, BM25, fusion, parent expansion, reranking, and grounding behavior are demonstrated.
- Dense and BM25 retrieval operate as independent paths before rank fusion.
- RAGAS produces metrics and the main failures have been reviewed.
- Strategy profiles can be compared using one saved generated dataset.
- Citation pages/page ranges are correct for paginated documents, with honest metadata for page-less documents.
- Health/readiness, configuration, secrets, and database migration behavior are addressed.
- Versioning and summary terminology are accurate.
- The README and resume describe the same system.

After Phase 3, no feature work is allowed unless the final manual test reveals
a correctness defect. Performance tuning, refactoring, or new capabilities do
not reopen the project scope.

## Deliberately Out of Scope

These are valid future production improvements, but they should not block this
completion target:

- Kubernetes or multi-region deployment
- Persistent sparse retrieval infrastructure
- Arbitrary user-facing metadata filter language
- Full document version comparison and rollback
- LLM-based semantic summarization unless evaluation requires it
- Token-level streaming resumability
- Frontend automated-test infrastructure
- Distributed tracing and an elaborate observability stack
- Multi-agent workflows, web search, multimodal RAG, or knowledge graphs

## Recommended Execution Order

Complete Phase 1 before changing retrieval behavior. Complete the initial RAGAS
dataset in Phase 2 before tuning chunking, fusion, or prompts. Specifically,
change BM25 to an independent retrieval path before comparing hybrid strategies.
Complete Phase 3 after behavior and evaluation are stable so the documentation
and resume reflect the final implementation rather than planned features. The
developer then performs the single final manual test and stops development.
