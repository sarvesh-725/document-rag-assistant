"""Optional Gemini-backed RAGAS adapter for offline evaluation only."""

import math


def evaluate_with_ragas(rows: list[dict]) -> dict:
    """Evaluate normalized ``question/contexts/answer/reference`` rows.

    This function is never called by the chat router. Importing RAGAS lazily
    keeps normal user requests independent of evaluation dependencies.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional RAGAS evaluation dependencies before using --ragas"
        ) from exc

    from app.config import get_settings
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for RAGAS")

    dataset_rows = [
        {
            "question": row["question"],
            "contexts": row.get("contexts", []),
            "answer": row.get("answer", ""),
            "reference": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
        }
        for row in rows
    ]
    result = evaluate(
        Dataset.from_list(dataset_rows),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            google_api_key=settings.gemini_api_key,
        ),
        embeddings=GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        ),
    )
    if hasattr(result, "to_pandas"):
        case_results = result.to_pandas().to_dict(orient="records")
    else:
        case_results = []

    normalized_cases = []
    for case in case_results:
        normalized_cases.append(
            {
                key: value.item() if hasattr(value, "item") else value
                for key, value in case.items()
            }
        )
    summary = {}
    metric_names = (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    )
    for metric_name in metric_names:
        values = [
            case[metric_name]
            for case in normalized_cases
            if isinstance(case.get(metric_name), (int, float))
            and not math.isnan(float(case[metric_name]))
        ]
        if values:
            summary[metric_name] = round(sum(values) / len(values), 4)
    return {"cases": normalized_cases, "summary": summary}
