import json
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.database.models import User, ChatSession, SessionFile
from app.auth.security import get_current_user

from app.services.rag_engine import (
    _get_embeddings,
    client as qdrant_client,
    QDRANT_COLLECTION_NAME,
    init_qdrant
)
from qdrant_client.http import models as qdrant_models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, messages_from_dict, messages_to_dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

logger = logging.getLogger("chat_router")
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatQueryRequest(BaseModel):
    session_id: str
    question: str
    include_prev_files: bool = True
    explicit_files: List[str] = Field(default_factory=list)


async def chat_stream_generator(request: ChatQueryRequest, current_user: User, db: AsyncSession):
    """
    Async generator that performs similarity search on Qdrant, runs reranking,
    calls Gemini stream, yields SSE data chunks, and commits updates to PostgreSQL.
    """
    question = request.question.strip()
    session_id = request.session_id.strip()

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    chat_session = session_result.scalars().first()
    if not chat_session:
        yield f"data: {json.dumps({'error': 'Session not found or unauthorized'})}\n\n"
        return

    file_result = await db.execute(
        select(SessionFile).where(SessionFile.session_id == session_id)
    )
    session_files = file_result.scalars().all()

    just_uploaded_names = {f.filename for f in session_files if f.just_uploaded}
    committed_names = {f.filename for f in session_files if f.is_committed}

    if request.include_prev_files:
        base_set = committed_names | set(request.explicit_files)
    else:
        base_set = set(request.explicit_files)

    resolved_set = base_set | just_uploaded_names
    resolved_files_list = list(resolved_set)

    if not resolved_files_list:
        yield f"data: {json.dumps({'error': 'At least one active document must be selected or uploaded to start querying.'})}\n\n"
        return


    user_message = {
        "role": "user",
        "content": question,
        "bound_files": resolved_files_list
    }
    chat_session.messages = list(chat_session.messages) + [user_message]
    db.add(chat_session)
    await db.commit()

    await db.refresh(chat_session)

    must_conditions = [
        qdrant_models.FieldCondition(
            key="user_id",
            match=qdrant_models.MatchValue(value=str(current_user.id))
        )
    ]

    if resolved_files_list:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="filename",
                match=qdrant_models.MatchAny(any=resolved_files_list)
            )
        )

    search_filter = qdrant_models.Filter(must=must_conditions)

    try:
        embeddings = _get_embeddings()
        query_vector = embeddings.embed_query(question)

        await init_qdrant()

        search_result = await qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=10
        )
    except Exception as e:
        logger.exception("Failed Qdrant vector database similarity lookup")
        yield f"data: {json.dumps({'error': f'Failed to query vector database: {str(e)}'})}\n\n"
        return

    documents = [hit.payload.get("text", "") for hit in search_result.points if hit.payload]

    if not documents:
        ai_message = {
            "role": "assistant",
            "content": "I cannot find the answer in the provided documents.",
            "bound_files": resolved_files_list
        }
        chat_session.messages = list(chat_session.messages) + [ai_message]
        db.add(chat_session)

        existing_filenames = {f.filename for f in session_files}
        for filename in resolved_files_list:
            if filename in existing_filenames:
                f = next(sf for sf in session_files if sf.filename == filename)
                f.is_committed = True
                f.just_uploaded = False
                db.add(f)
            else:
                hash_res = await db.execute(
                    select(SessionFile.file_hash).where(
                        SessionFile.filename == filename,
                        SessionFile.status == "completed"
                    ).limit(1)
                )
                file_hash = hash_res.scalar() or "legacy"
                new_assoc = SessionFile(
                    session_id=session_id,
                    filename=filename,
                    file_hash=file_hash,
                    is_committed=True,
                    just_uploaded=False,
                    status="completed"
                )
                db.add(new_assoc)

        for f in session_files:
            if f.filename not in resolved_files_list:
                f.is_committed = False
                f.just_uploaded = False
                db.add(f)

        await db.commit()
        yield f"data: {json.dumps({'text': 'I cannot find the answer in the provided documents.'})}\n\n"
        return

    retrieved_docs = [Document(page_content=doc) for doc in documents]
    try:
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
        ranked_docs = reranker.compress_documents(retrieved_docs, question)
        context_docs = [doc.page_content for doc in ranked_docs]
    except Exception as cohere_err:
        logger.warning(f"Cohere Reranker failed: {cohere_err}. Falling back to top 5 initial Qdrant results.")
        context_docs = documents[:5]

    context = "\n\n".join(context_docs)

    db_messages = chat_session.messages[:-1]
    langchain_messages = []
    for msg in db_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "human"):
            langchain_messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            langchain_messages.append(AIMessage(content=content))

    trimmed_messages = trim_messages(
        langchain_messages, max_tokens=4, strategy="last", token_counter=len
    )

    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using ONLY the provided context and the conversation history. "
                "If the answer cannot be found in the context or history, say 'I cannot find the answer in the provided documents.' "
                "Do not make things up.\n\n"
                "Context:\n{context}",
            ),
            *trimmed_messages,
            ("human", "{query}"),
        ]
    )

    chain = prompt | model

    full_response_text = ""
    try:
        async for chunk in chain.astream({"context": context, "query": question}):
            content = chunk.content
            if isinstance(content, list):
                content = "".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
            else:
                content = str(content)
            full_response_text += content

            yield f"data: {json.dumps({'text': content})}\n\n"
    except Exception as stream_err:
        logger.exception("Error occurred during text response generation streaming loop")
        yield f"data: {json.dumps({'error': f'Streaming interrupted: {str(stream_err)}'})}\n\n"
        return

    ai_message = {
        "role": "assistant",
        "content": full_response_text,
        "bound_files": resolved_files_list
    }
    chat_session.messages = list(chat_session.messages) + [ai_message]

    existing_filenames = {f.filename for f in session_files}
    for filename in resolved_files_list:
        if filename in existing_filenames:
            f = next(sf for sf in session_files if sf.filename == filename)
            f.is_committed = True
            f.just_uploaded = False
            db.add(f)
        else:
            hash_res = await db.execute(
                select(SessionFile.file_hash).where(
                    SessionFile.filename == filename,
                    SessionFile.status == "completed"
                ).limit(1)
            )
            file_hash = hash_res.scalar() or "legacy"
            new_assoc = SessionFile(
                session_id=session_id,
                filename=filename,
                file_hash=file_hash,
                is_committed=True,
                just_uploaded=False,
                status="completed"
            )
            db.add(new_assoc)

    for f in session_files:
        if f.filename not in resolved_files_list:
            f.is_committed = False
            f.just_uploaded = False
            db.add(f)

    db.add(chat_session)
    await db.commit()
    logger.info("Saved session chat history exchange and committed files successfully.")


@router.post("/query")
async def query_chat_stream(
    request: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Streams the query response token-by-token using SSE.
    """
    return StreamingResponse(
        chat_stream_generator(request, current_user, db),
        media_type="text/event-stream"
    )
