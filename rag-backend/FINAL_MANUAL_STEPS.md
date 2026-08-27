# Final Manual Steps

Run these steps after the code changes are reviewed.

1. Rotate any real credentials that were ever exposed in `.env`.
2. In the project virtual environment, remove the retired evaluation packages:

```powershell
python -m pip uninstall -y ragas datasets openai
python -m pip install -r requirements.txt
```

Keep `openai` installed if another project uses the same virtual environment.
The backend no longer imports it.

3. Create a LangSmith API key and set these values in the ignored backend `.env`:

```text
LANGSMITH_API_KEY=<key>
LANGSMITH_PROJECT=document-assistant-evaluation
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Also set the evaluation limits in `.env`:

```text
EVALUATION_SAMPLE_COUNT=12
EVALUATION_BATCH_SIZE=4
EVALUATION_CASE_LIMIT=12
EVALUATION_REQUEST_DELAY_SECONDS=5
EVALUATION_COHERE_MIN_INTERVAL_SECONDS=7
```

4. Start PostgreSQL, Redis, Qdrant, the API, Taskiq worker, outbox publisher,
   and frontend using `RUN_AND_EVALUATE.md`.
5. Register or log in, upload two or three materially different PDF/TXT files,
   and wait for every file to become `READY`.
6. Run the 12-case benchmark:

```powershell
python -m app.evaluation.cli --user-id UUID --auto --langsmith --document-id ID1 --document-id ID2
```

7. Inspect `evaluation/generated_dataset.jsonl`. Verify retrieval scope contains
   all selected documents, expected sources/pages are correct, and two cases
   are valid no-answer cases.
8. Inspect the two answer-comparison results and LangSmith experiment links.
   Choose the final strategy manually using answer token F1, evidence support,
   citation correctness, retrieval quality, failure cases, and latency.
9. Change one setting at a time in `evaluation/strategies.json`, then rerun the
   saved dataset with `--all-strategies --retrieval-only` or one strategy with
   `--langsmith`. Do not regenerate the dataset for retrieval experiments.
10. For a new document set, upload the replacements, wait for `READY`, and run
    with `--reset` and the new document IDs. Keep the previous results.
11. After the strategy is frozen, manually test register/login, upload and
    processing, selected-document questions, citations, no-answer refusal,
    multi-turn chat, chitchat, deletion, retry, duplicate request, and logout.
12. After this final test, Phase 4 is complete and no further feature work is
    planned.

## Production Configuration

Change live RAG behavior in `rag-backend/.env`, which is read by
`app/config.py`. The main production retrieval settings are:

```text
DENSE_TOP_K
BM25_TOP_K
BM25_SCAN_LIMIT
RRF_K
RERANK_TOP_K
RERANK_THRESHOLD
FINAL_CANDIDATE_COUNT
COHERE_API_KEY
RERANKER_MODEL
```

`evaluation/strategies.json` changes evaluation profiles only. Evaluation-only
settings such as `EVALUATION_CASE_LIMIT` and
`EVALUATION_COHERE_MIN_INTERVAL_SECONDS` do not change production behavior.
