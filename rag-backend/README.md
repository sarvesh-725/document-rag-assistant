# Document RAG Assistant Backend

## Local Startup

1. Copy `.env.example` to `.env` and set the required secret and provider values.
2. Start the backend services from this directory:

```bash
docker compose up --build
```

This starts PostgreSQL, Redis, Qdrant, the FastAPI API, the Taskiq worker, and
the outbox publisher. The API applies Alembic migrations before it starts.

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
