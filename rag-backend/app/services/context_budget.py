"""Tokenizer-aware budgeting for the final model context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


def estimate_tokens(text: str) -> int:
    """Estimate model tokens with tiktoken when available, never by message count."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(re.findall(r"\S+", text)) * 4 // 3)


def _truncate_tokens(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    if estimate_tokens(text) <= token_limit:
        return text
    words = text.split()
    output: list[str] = []
    for word in words:
        candidate = " ".join(output + [word])
        if estimate_tokens(candidate) > token_limit:
            break
        output.append(word)
    return " ".join(output)


def _message_text(message: Any) -> str:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, dict):
        role = message.get("role", "user")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    return f"{role}: {content or ''}"


def _evidence_text(evidence: Any) -> str:
    text = getattr(evidence, "text", None)
    if text is not None:
        return str(text)
    candidate = getattr(evidence, "candidate", None)
    return str(getattr(candidate, "text", ""))


@dataclass(frozen=True)
class BudgetedContext:
    summary: str
    recent_messages: list[Any]
    evidence: list[Any]
    estimated_tokens: int


class ContextBudgetManager:
    """Reserve system/question/output space before filling history/evidence."""

    def __init__(
        self,
        *,
        model_context_limit: int = 32768,
        expected_output_tokens: int = 1024,
        summary_token_budget: int = 512,
        history_token_budget: int = 2048,
    ):
        if model_context_limit <= expected_output_tokens:
            raise ValueError("model context limit must leave room for output")
        self.model_context_limit = model_context_limit
        self.expected_output_tokens = expected_output_tokens
        self.summary_token_budget = summary_token_budget
        self.history_token_budget = history_token_budget

    def fit(
        self,
        *,
        system_prompt: str,
        current_question: str,
        summary: str,
        recent_messages: Sequence[Any],
        evidence: Sequence[Any],
    ) -> BudgetedContext:
        available = (
            self.model_context_limit
            - self.expected_output_tokens
            - estimate_tokens(system_prompt)
            - estimate_tokens(current_question)
        )
        if available <= 0:
            return BudgetedContext("", [], [], self.model_context_limit - self.expected_output_tokens)

        summary = _truncate_tokens(summary, min(self.summary_token_budget, available))
        available -= estimate_tokens(summary)

        kept_messages: list[Any] = []
        message_tokens = 0
        for message in reversed(list(recent_messages)):
            cost = estimate_tokens(_message_text(message))
            if message_tokens + cost > min(self.history_token_budget, available):
                break
            kept_messages.append(message)
            message_tokens += cost
        kept_messages.reverse()
        available -= message_tokens

        kept_evidence: list[Any] = []
        evidence_tokens = 0
        # Candidates are already rerank/fusion ordered. Dropping from the end
        # therefore drops the lowest-ranked evidence first.
        for item in evidence:
            cost = estimate_tokens(_evidence_text(item))
            if cost <= available - evidence_tokens:
                kept_evidence.append(item)
                evidence_tokens += cost
            else:
                break
        estimated = (
            estimate_tokens(system_prompt)
            + estimate_tokens(current_question)
            + estimate_tokens(summary)
            + message_tokens
            + evidence_tokens
        )
        return BudgetedContext(summary, kept_messages, kept_evidence, estimated)
