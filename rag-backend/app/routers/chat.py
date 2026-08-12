"""
Chat routes — Phase 1 stub.

The streaming query endpoint is stubbed pending Phase 2+ rewrite
with Message, QueryRun, and QueryRunDocument models.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.database.models import User
from app.auth.security import get_current_user

logger = logging.getLogger("chat_router")
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatQueryRequest(BaseModel):
    session_id: str
    question: str
    selected_document_ids: List[str] = Field(default_factory=list)


@router.post("/query")
async def query_chat_stream(
    request: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
):
    """
    STUB: Chat query endpoint pending Phase 2+ rewrite.
    Will create Message rows, QueryRun, perform hybrid retrieval,
    rerank, stream via SSE, and persist with status tracking.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Chat query pending Phase 2+ rewrite with Message/QueryRun/hybrid retrieval pipeline."
    )
