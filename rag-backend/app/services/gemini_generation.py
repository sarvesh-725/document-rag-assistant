"""Async Gemini answer streaming adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any


class GeminiAnswerStreamer:
    """Stream Gemini text from the already-built, sectioned prompt package."""

    def __init__(self, model: str | None = None, llm: Any = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.llm = llm

    def _get_llm(self):
        if self.llm is not None:
            return self.llm
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini generation")
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
        )
        return self.llm

    async def stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Yield only text while preserving all prompt metadata externally."""
        async for chunk in self._get_llm().astream(list(messages)):
            content = getattr(chunk, "content", chunk)
            if isinstance(content, str):
                if content:
                    yield content
                continue
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
                if text:
                    yield text
