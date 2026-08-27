# Document RAG Assistant Backend

## Local Startup

1. Copy `.env.example` to `.env` and set the required secret and provider values.
2. Start only the infrastructure containers from this directory:

```bash
docker compose up postgres redis qdrant
```

Run FastAPI, the Taskiq worker, and the outbox publisher directly from the
project virtual environment in separate terminals. Exact commands are in
`RUN_AND_EVALUATE.md`. This avoids building the large Python image three times.

Start the Next.js frontend separately from `rag-frontend`:

```bash
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` and calls the API at
`http://localhost:8000`.

## Service Responsibilities

- FastAPI handles authentication, document metadata, chat requests, and SSE.
- PostgreSQL owns document, conversation, ingestion, query-run, and outbox state.
- The outbox publisher sends committed work to Redis.
- The Taskiq worker parses, chunks, embeds, and indexes documents.
- Qdrant stores and searches document vectors.
- Gemini and Cohere are configured external providers; their keys are never committed.

Retrieval runs dense Qdrant search and independent in-process BM25 candidate
generation, then applies application-side RRF, optional Cohere child reranking,
deterministic parent expansion/deduplication, and a bounded context budget.
Qdrant-native sparse retrieval is a future scaling option, not part of this MVP.

Live retrieval configuration belongs in `rag-backend/.env` and is centralized
in `app/config.py` (`DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `RERANK_TOP_K`,
`RERANK_THRESHOLD`, `FINAL_CANDIDATE_COUNT`, and `COHERE_API_KEY`). The file
`evaluation/strategies.json` affects evaluation profiles only.

## Operational Endpoints

- `GET http://localhost:8000/health` checks API liveness.
- `GET http://localhost:8000/ready` checks PostgreSQL, Redis, and Qdrant reachability.

`/health` does not depend on downstream services. `/ready` returns HTTP 503 with
individual dependency results until the local services are reachable.

## Reliability Model

Document state and the outbox event are committed together before background
work is published. Outbox publishing is at-least-once and retryable. Worker
operations use deterministic identities and safe upserts; exactly-once delivery
is not claimed.

For API details, see `API.md`. For the remaining implementation phases, see
`PROJECT_COMPLETION_PLAN.md`.

## RAG Evaluation

For the complete upload-to-results workflow, package migration, rate limits,
manual reset steps, and LangSmith setup, see `EVALUATION_GUIDE.md`. The final
developer-only handoff checklist is in `FINAL_MANUAL_STEPS.md`.

The evaluator accepts a verified JSONL dataset or generates twelve page-aware
cases, including no-answer cases, from two or three uploaded READY PDF/TXT
documents. It compares the existing retrieval profiles, answers only the top
two retrieval profiles, bounds Gemini usage, and uploads repeatable experiments
to LangSmith. Results are also written to
`evaluation/results/`. The profiles run dense and BM25 independently before
RRF; `dense_only`, `hybrid`, `hybrid_parent`, and `hybrid_parent_rerank` remain
the existing strategy set.
