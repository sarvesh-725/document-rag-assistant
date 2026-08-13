"""Derived conversation memory and chronological recent-message loading."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import ChatSession, ConversationSummary, Message
from app.database.repositories import (
    get_or_create_conversation_summary,
    get_recent_session_messages,
)

logger = logging.getLogger("conversation_context")


@dataclass(frozen=True)
class HistoryConfig:
    recent_messages: int = 4
    summary_token_budget: int = 512
    history_token_budget: int = 2048

    @classmethod
    def from_settings(cls) -> "HistoryConfig":
        settings = get_settings()
        return cls(
            recent_messages=settings.history_recent_messages,
            summary_token_budget=settings.summary_token_budget,
            history_token_budget=settings.history_token_budget,
        )


@dataclass(frozen=True)
class ConversationContext:
    summary: str
    last_summarized_message_id: Optional[uuid.UUID]
    recent_messages: list[Message]


class ConversationSummarizer:
    """Safe derived-state summarizer; canonical messages remain untouched."""

    def summarize(
        self,
        messages: Sequence[Message],
        previous_summary: str = "",
        *,
        token_budget: int = 512,
    ) -> str:
        lines = []
        if previous_summary.strip():
            lines.append(previous_summary.strip())
        for message in messages:
            content = (message.content or "").strip().replace("\n", " ")
            if content:
                lines.append(f"{message.role}: {content}")
        text = "\n".join(lines)
        # A conservative character approximation is only used to bound the
        # derived summary; model context budgeting uses a tokenizer.
        return text[: max(1, token_budget * 4)]


async def maybe_update_summary(
    db: AsyncSession,
    session_id: uuid.UUID,
    current_message_id: uuid.UUID,
    config: HistoryConfig,
    summarizer: Optional[ConversationSummarizer] = None,
) -> None:
    """Summarize messages older than the retained recent window.

    Failure is intentionally swallowed: summary is derived state and recent
    canonical messages remain sufficient for the query.
    """
    summarizer = summarizer or ConversationSummarizer()
    try:
        current = await db.get(Message, current_message_id)
        if current is None:
            return
        await db.execute(
            select(ChatSession.id).where(ChatSession.id == session_id).with_for_update()
        )
        result = await db.execute(
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.sequence_number < current.sequence_number,
            )
            .order_by(Message.sequence_number.asc())
        )
        conversation = list(result.scalars().all())
        if len(conversation) <= config.recent_messages:
            return
        older = conversation[: -config.recent_messages] if config.recent_messages else conversation
        if not older:
            return

        summary = summarizer.summarize(
            older,
            token_budget=config.summary_token_budget,
        )
        await get_or_create_conversation_summary(
            db,
            session_id,
            summary,
            older[-1].id,
        )
    except Exception:
        logger.exception("Conversation summary update failed; continuing without it")


async def load_conversation_context(
    db: AsyncSession,
    session_id: uuid.UUID,
    current_message_id: uuid.UUID,
    *,
    config: Optional[HistoryConfig] = None,
) -> ConversationContext:
    config = config or HistoryConfig.from_settings()
    await maybe_update_summary(db, session_id, current_message_id, config)
    summary_result = await db.execute(
        select(
            ConversationSummary.summary,
            ConversationSummary.last_summarized_message_id,
        ).where(ConversationSummary.session_id == session_id)
    )
    summary_row = summary_result.first()
    recent_messages = await get_recent_session_messages(
        db,
        session_id,
        limit=config.recent_messages,
        exclude_message_id=current_message_id,
    )
    return ConversationContext(
        summary=(summary_row[0] or "") if summary_row else "",
        last_summarized_message_id=summary_row[1] if summary_row else None,
        recent_messages=recent_messages,
    )
