"""Async Gemini answer streaming adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

logger = logging.getLogger("gemini_generation")


class GeminiGenerationError(RuntimeError):
    """Provider setup or streaming failed after the request reached Gemini."""


class GeminiAnswerStreamer:
    """Stream Gemini text from the already-built, sectioned prompt package."""

    def __init__(self, model: str | None = None, llm: Any = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.llm = llm

    def _get_llm(self):
        if self.llm is not None:
            return self.llm
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiGenerationError("A Gemini API key is required for generation")
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=api_key,
        )
        return self.llm

    async def stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Yield only text while preserving all prompt metadata externally."""
        try:
            llm = self._get_llm()
        except GeminiGenerationError:
            raise
        except Exception as exc:
            logger.exception("Gemini client initialization failed", extra={"model": self.model})
            raise GeminiGenerationError("Gemini client initialization failed") from exc

        for attempt in range(2):
            emitted = False
            try:
                async for chunk in llm.astream(list(messages)):
                    content = getattr(chunk, "content", chunk)
                    text = content if isinstance(content, str) else ""
                    if isinstance(content, list):
                        text = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    if text:
                        emitted = True
                        yield text
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if emitted or attempt == 1:
                    logger.exception(
                        "Gemini streaming failed",
                        extra={"model": self.model, "attempt": attempt + 1},
                    )
                    raise GeminiGenerationError("Gemini streaming failed") from exc
                await asyncio.sleep(0.25)
