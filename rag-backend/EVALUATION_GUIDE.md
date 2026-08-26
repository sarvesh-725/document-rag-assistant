# RAG Evaluation Guide

This is a small, rate-safe evaluation workflow. It does not require manually
writing questions and does not add CI, dashboards, tracing, or an experiment
database.

## First Run

1. Start the application and upload one or more documents through the frontend.
2. Wait until every document used for evaluation shows `READY`.
3. Ensure the updated backend dependencies are installed:

```bash
pip install -r requirements.txt
```

4. From `rag-backend`, run the automatic evaluation. Replace the user UUID with
   the account that owns the uploaded documents:

```bash
python -m app.evaluation.cli --user-id YOUR_USER_UUID --auto --ragas
```

The command automatically:

- Finds all READY documents owned by that user. Use `--document-id ID` once per
  document to restrict the source set.
- Generates 5 page-linked evaluation cases in one batch using Gemini.
- Saves the generated cases to `evaluation/generated_dataset.jsonl`.
- Compares `dense_only`, `hybrid`, `hybrid_parent`, and
  `hybrid_parent_rerank` using retrieval metrics only.
- Reuses query embeddings while comparing profiles.
- Selects the strongest retrieval profile using document recall, page recall,
  and MRR.
- Generates answers and runs RAGAS only for the selected profile and up to 5
  cases.
- Saves the complete result to `evaluation/results/`.

Generation and answer requests are run sequentially with a delay. The default
limits are 5 generated cases, a maximum of 20 cases, a batch size of 5, and 5
RAGAS cases. Set `EVALUATION_SAMPLE_COUNT=10` when you want a larger comparison.
Google limits vary by project and must be checked in [AI Studio Rate
Limits](https://aistudio.google.com/rate-limit). If a 429 occurs, wait and rerun
using `EVALUATION_SAMPLE_COUNT=5` and a larger
`EVALUATION_REQUEST_DELAY_SECONDS` in `.env`.

## Repeat After A Strategy Change

Do not generate a new dataset for every experiment. Reuse the saved dataset so
the comparison remains fair and does not spend more Gemini requests.

1. Change only one setting at a time in `evaluation/strategies.json`:

```text
dense_top_k
bm25_top_k
parent_expansion
reranker
rerank_top_k
```

   Global limits such as `BM25_SCAN_LIMIT`, `RRF_K`, and
   `RERANK_THRESHOLD` can be changed in `.env`.
2. Compare all retrieval profiles without answer generation:

```bash
python -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --all-strategies --retrieval-only
```

3. Run answer generation and RAGAS only for the profile being considered:

```bash
python -m app.evaluation.cli --user-id YOUR_USER_UUID --dataset evaluation/generated_dataset.jsonl --strategy hybrid_parent_rerank --ragas
```

4. Keep a change only when retrieval quality improves without an unacceptable
   latency or answer-quality regression.

## What Requires Re-indexing

- Changing top-k, RRF, BM25 scan limits, parent expansion, or reranking only requires rerunning evaluation.
- Changing chunk size, chunk overlap, parser behavior, or embedding settings requires re-uploading/re-indexing the documents before evaluation.
- Changing the generation prompt or Gemini model requires only rerunning answer evaluation.

## Interpreting Results

- Low context recall: inspect chunking, top-k, and parent expansion.
- Low context precision: inspect BM25 limits, RRF, filtering, and reranking.
- Low faithfulness: inspect the grounding prompt and evidence threshold.
- Low answer relevance: inspect the prompt and conversation context.
- Low page recall: inspect parser metadata preservation and citation propagation.

Compare retrieval first, then generation. Do not change multiple pipeline stages
in one experiment.
