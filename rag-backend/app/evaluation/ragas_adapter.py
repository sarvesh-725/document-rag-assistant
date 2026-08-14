"""Optional RAGAS adapter, intentionally outside production request flow."""


def evaluate_with_ragas(result: dict) -> dict:
    """Evaluate a completed result when RAGAS is installed.

    This function is never called by the chat router. Importing RAGAS lazily
    keeps normal user requests independent of evaluation dependencies.
    """
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RAGAS is required only for offline evaluation") from exc
    return {"ragas_pending": True, "result": result}
