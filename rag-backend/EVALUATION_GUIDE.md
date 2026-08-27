# RAG Evaluation Guide

Evaluation is an offline command-line workflow. LangSmith stores the dataset
and experiment results; it is never enabled in the production chat, upload, or
worker path.

## Manual Setup

Run these commands in the backend virtual environment. These are the packages
to uninstall from this project environment:

```powershell
python -m pip uninstall -y ragas datasets openai
python -m pip install -r requirements.txt
```

Do not uninstall `openai` if another application in the same virtual environment
uses it. The backend has no runtime import of `ragas`, `datasets`, or `openai`.

Create a LangSmith API key and set these values in `.env`:

```text
LANGSMITH_API_KEY=<your LangSmith API key>
LANGSMITH_PROJECT=document-assistant-evaluation
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Use `EVALUATION_SAMPLE_COUNT=12`, `EVALUATION_BATCH_SIZE=4`,
`EVALUATION_CASE_LIMIT=12`, and `EVALUATION_REQUEST_DELAY_SECONDS=5` for the
recommended bounded run.

Keep the key out of Git. A LangSmith workspace/project is external; no local
LangSmith container is required.

## First Run

1. Start the application using `RUN_AND_EVALUATE.md`.
2. Create an account, upload two or three PDF/TXT documents, and wait for every
   selected document to show `READY`.
3. Note the owner UUID and document UUIDs.
4. Run the rate-safe automatic workflow:

```powershell
python -m app.evaluation.cli --user-id YOUR_USER_UUID --auto --langsmith --document-id DOCUMENT_UUID_1 --document-id DOCUMENT_UUID_2
```

The command generates twelve validated page-aware cases, compares all retrieval
profiles without answer generation, retains the top two retrieval profiles,
generates answers only for those profiles (no generation for no-answer cases),
and uploads one LangSmith experiment per profile. Results are also
saved locally under `evaluation/results/`.

Use a supplied JSONL dataset instead of generation:

```powershell
python -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/dataset.jsonl --strategy hybrid_parent_rerank --langsmith
```

The JSONL shape is documented in `evaluation/dataset.template.jsonl`. Verify
ground-truth answers, document IDs, and page metadata yourself.

## Repeat Experiments

Reuse the same dataset to make comparisons fair and avoid Gemini dataset
generation requests. Change one setting at a time in `evaluation/strategies.json`:

```text
dense_top_k
bm25_top_k
parent_expansion
reranker
rerank_top_k
```

Then run retrieval-only comparisons, which do not call Gemini for answers:

```powershell
python -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --all-strategies --retrieval-only
```

Run one candidate strategy with a bounded answer set and a LangSmith experiment:

```powershell
python -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --strategy hybrid_parent_rerank --langsmith --case-limit 12
```

`--case-limit` bounds answer generation and LangSmith evaluation. The default is
12, with no-answer cases handled deterministically without Gemini generation.
Retrieval-only runs still use every row in the saved dataset. Change global limits such as
`BM25_SCAN_LIMIT`, `RRF_K`, and `RERANK_THRESHOLD` in `.env`.

## Rate-Limit Rules

- Dataset generation is sequential, uses batches of at most five, and defaults
  to twelve cases.
- Answer generation is sequential for the top two profiles and skips no-answer
  cases.
- Query embeddings are cached across profile comparisons.
- LangSmith evaluators are deterministic and make no Gemini or embedding calls.
- Keep `EVALUATION_REQUEST_DELAY_SECONDS` at 5 or increase it after a 429. For
  the supplied limits, do not run multiple evaluations concurrently.

The workflow intentionally does not use a Gemini LLM-as-judge. This avoids
consuming the very small Gemini 3.5 Flash quota and keeps repeated experiments
safe. LangSmith records deterministic code evaluators: document recall@5/@10,
MRR, nDCG@10, page recall, citation precision/recall, answer token F1, evidence
support rate, and no-answer correctness. These are heuristics, not semantic
faithfulness judgments.

## Change Documents Or Reset

Changing chunk size, chunk overlap, parser behavior, or embedding settings
requires uploading/re-indexing the documents before evaluation. Retrieval
settings and prompt/model changes do not require a new dataset.

After completing one document set, upload the replacement documents and wait
for `READY`. Generate a fresh dataset and reset a named LangSmith dataset when
needed:

```powershell
python -m app.evaluation.cli --user-id YOUR_USER_UUID --auto --langsmith --reset --langsmith-dataset my-document-assistant-eval --document-id NEW_DOCUMENT_UUID_1 --document-id NEW_DOCUMENT_UUID_2
```

Without `--langsmith-dataset`, the dataset name includes a content fingerprint,
so a changed JSONL dataset automatically gets a new LangSmith dataset while
old experiments remain available for comparison.

## Interpretation

- Low document recall: inspect chunking, top-k, and parent expansion.
- Low citation precision: inspect filtering, fusion, and reranking.
- Low evidence support: inspect the grounding prompt and evidence threshold.
- Low answer token F1: inspect the prompt and conversation context.
- Low page/citation recall: inspect parser metadata preservation and citation propagation.

Keep a change only when retrieval quality improves without an unacceptable
latency or answer-quality regression.
