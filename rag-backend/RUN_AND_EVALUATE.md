# Run And Evaluate

Run the commands below in PowerShell. Docker Desktop must be running.

## 1. Prepare The Backend

Run from:

```text
E:\document-assistant\rag-backend
```

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-backend"
if (-not (Test-Path -LiteralPath ".env")) { Copy-Item -LiteralPath ".env.example" -Destination ".env" }
notepad .env
```

In `.env`, set at least:

```text
SECRET_KEY=<random value with at least 32 characters>
GEMINI_API_KEY=<your Gemini API key>
UNSTRUCTURED_API_KEY=<your Unstructured API key>
UNSTRUCTURED_API_URL=<your Unstructured API URL>
```

For the host-side evaluation command, keep these local values unless you
intentionally want to evaluate a remote environment:

```text
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/document_assistant
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
```

Docker internally overrides these three values to use the Compose service names.

`COHERE_API_KEY` is optional. Leave `QDRANT_API_KEY` empty for the local
Qdrant container. Do not commit `.env`.

Remove the retired evaluation packages and install the backend dependencies:

```powershell
& "E:\document-assistant\.venv\Scripts\python.exe" -m pip uninstall -y ragas datasets openai
& "E:\document-assistant\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Do not uninstall `openai` if another project uses it in this same virtual
environment. Set `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and
`LANGSMITH_ENDPOINT` in `.env` before using `--langsmith`.

## 2. Start Infrastructure Only

Remain in:

```text
E:\document-assistant\rag-backend
```

Do not build the API, worker, or publisher images. Start only the local
infrastructure containers and leave this terminal running:

```powershell
docker compose up postgres redis qdrant
```

This starts PostgreSQL, Redis, and Qdrant without building the large Python
application image.

## 3. Start The API, Worker, And Publisher

Open three additional PowerShell windows. Run all commands below from:

```text
E:\document-assistant\rag-backend
```

### API window

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-backend"
& "E:\document-assistant\.venv\Scripts\python.exe" -m alembic upgrade head
& "E:\document-assistant\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Taskiq worker window

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-backend"
& "E:\document-assistant\.venv\Scripts\taskiq.exe" worker app.broker:broker
```

### Outbox publisher window

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-backend"
& "E:\document-assistant\.venv\Scripts\python.exe" -m app.outbox_publisher
```

The API applies no destructive database reset. The explicit Alembic command only
applies pending migrations.

## 4. Start The Frontend

Open another PowerShell window. Run from:

```text
E:\document-assistant\rag-frontend
```

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-frontend"
npm install
npm run dev
```

Run `npm install` only the first time or after dependency changes. Keep this
terminal running.

If the frontend shows `404` or `500` at `/` after an earlier Next.js run, stop
the frontend with `Ctrl+C`, then run from `E:\document-assistant\rag-frontend`:

```powershell
Remove-Item -LiteralPath ".next" -Recurse -Force -ErrorAction SilentlyContinue
npm run dev
```

This clears only generated Next.js cache files; it does not delete source code.

## 5. Check The Application

Open:

```text
http://localhost:3000
```

Optional checks from any PowerShell window:

```powershell
Invoke-RestMethod "http://localhost:8000/health"
Invoke-RestMethod "http://localhost:8000/ready"
```

`/health` should return `status: ok`. `/ready` should return `status: ready`
and successful checks for PostgreSQL, Redis, and Qdrant.

## 6. Upload Evaluation Documents

In the frontend:

1. Create an account or log in.
2. Upload one or more PDF/TXT documents.
3. Wait until each document shows `READY`.
4. Note the `user_id` returned during signup. The evaluation command needs the UUID of the document owner.

## 7. Generate Dataset And Obtain Results

Open another PowerShell window. Run from:

```text
E:\document-assistant\rag-backend
```

```powershell
Set-Location -LiteralPath "E:\document-assistant\rag-backend"
& "E:\document-assistant\.venv\Scripts\python.exe" -m app.evaluation.cli --user-id YOUR_USER_UUID --auto --langsmith --document-id DOCUMENT_UUID_1 --document-id DOCUMENT_UUID_2
```

The command generates twelve page-linked cases, compares all four retrieval
profiles without answer generation, evaluates only the top two profiles, and
writes:

```text
evaluation/generated_dataset.jsonl
evaluation/results/evaluation-<timestamp>.json
```

To evaluate only specific uploaded documents, add `--document-id` once per
document:

```powershell
& "E:\document-assistant\.venv\Scripts\python.exe" -m app.evaluation.cli --user-id YOUR_USER_UUID --document-id DOCUMENT_UUID --auto --langsmith
```

## 8. Repeat After A Strategy Change

Edit one setting at a time in:

```text
E:\document-assistant\rag-backend\evaluation\strategies.json
```

Then compare retrieval without spending requests on answer generation:

```powershell
& "E:\document-assistant\.venv\Scripts\python.exe" -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --all-strategies --retrieval-only
```

Evaluate the strategy being considered:

```powershell
& "E:\document-assistant\.venv\Scripts\python.exe" -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --strategy hybrid_parent_rerank --langsmith --case-limit 12
```

Available profile names are `dense_only`, `hybrid`, `hybrid_parent`, and
`hybrid_parent_rerank`. Change only one of `dense_top_k`, `bm25_top_k`,
`parent_expansion`, `reranker`, or `rerank_top_k` per experiment.

Change global limits such as `BM25_SCAN_LIMIT`, `RRF_K`, and
`RERANK_THRESHOLD` in `.env`. `--case-limit` bounds Gemini answer generation
and LangSmith evaluation. LangSmith evaluators make no Gemini calls.

Changing chunking, parser, or embedding settings requires re-uploading the
documents before evaluation. Changing retrieval limits or reranking settings
does not require a new dataset.

To replace the documents after an evaluation, upload the new PDF/TXT files,
wait for `READY`, and run `--auto --langsmith --reset` with the new document IDs.
Use `--langsmith-dataset NAME` when you want to replace a named remote dataset;
otherwise the content fingerprint creates a new dataset automatically.

## 9. Stop Services

When finished, press `Ctrl+C` in the infrastructure, API, worker, publisher,
and frontend terminals. Then run from
`E:\document-assistant\rag-backend`:

```powershell
docker compose down
```

The persistent local database and Qdrant data remain in Docker volumes.
