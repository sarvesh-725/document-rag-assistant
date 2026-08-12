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
from app.services.intent_classifier import classify_intent

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

    # Step 1: Verify the session exists and belongs to the authenticated user
    # This prevents users from accessing or modifying chat sessions that are not theirs.
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

    # Step 2: Retrieve all files that are completed and belong to ANY of the user's sessions (global files)
    # This allows users to re-use previously uploaded files across different sessions.
    global_file_result = await db.execute(
        select(SessionFile).join(ChatSession).where(
            ChatSession.user_id == current_user.id,
            SessionFile.status == "completed"
        )
    )
    global_files = global_file_result.scalars().all()

    # Step 3: Determine the set of "active" files for the query context
    # Simply use the explicit files sent by the frontend, but verify they are completed and belong to the user.
    valid_global_filenames = {f.filename for f in global_files}
    
    # Filter explicit files to only those that are valid and completed
    resolved_set = {f for f in request.explicit_files if f in valid_global_filenames}
    resolved_files_list = list(resolved_set)

    # Step 4: Save the user's question to the database history immediately
    # We store the question first so that even if the AI generation fails, the user's input is preserved.
    user_message = {
        "role": "user",
        "content": question,
        "bound_files": resolved_files_list
    }
    chat_session.messages = list(chat_session.messages) + [user_message]
    db.add(chat_session)
    await db.commit()

    # Refresh the session object from DB to ensure we have the latest state (e.g. updated messages array)
    await db.refresh(chat_session)

    # Step 5: Perform Vector Search (if active files are selected)
    documents = []
    if resolved_files_list:
        # Construct Qdrant metadata filters to ensure tenant isolation (user_id) 
        # and limit the search exclusively to the selected active files (filename).
        must_conditions = [
            qdrant_models.FieldCondition(
                key="user_id",
                match=qdrant_models.MatchValue(value=str(current_user.id))
            ),
            qdrant_models.FieldCondition(
                key="filename",
                match=qdrant_models.MatchAny(any=resolved_files_list)
            )
        ]

        search_filter = qdrant_models.Filter(must=must_conditions)

        try:
            # Generate the vector embedding for the user's question
            logger.info("Calling Gemini API (gemini-embedding-2-preview) to embed query...")
            embeddings = _get_embeddings()
            query_vector = embeddings.embed_query(question)

            # Check intent
            intent = classify_intent(question)
            
            if intent == "chitchat":
                # Skip RAG, directly reply.
                pass # We handle this by setting documents = [] and moving on. Wait, if it's chitchat, we should still use LLM but without context.
                # Actually, let's break out of vector search.
            else:
                # Ensure Qdrant connection is active
                await init_qdrant()
    
                # Execute the similarity search against Qdrant, retrieving top 15 closest chunks
                search_result = await qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=15
                )
                
                # Filter by similarity threshold 0.60
                valid_hits = [hit for hit in search_result.points if hit.payload and hit.score >= 0.60]
                
                # Extract the required text dynamically based on intent
                for hit in valid_hits:
                    if intent == "knowledge_specific":
                        text_content = hit.payload.get("child_text", hit.payload.get("text", ""))
                    else: # knowledge_broad
                        text_content = hit.payload.get("parent_text", hit.payload.get("text", ""))
                    
                    documents.append(text_content)
        except Exception as e:
            logger.exception("Failed Qdrant vector database similarity lookup")
            yield f"data: {json.dumps({'error': f'Failed. Try again.'})}\n\n"
            return

    # Step 6: Rerank the retrieved chunks (if vector search yielded results)
    context = ""
    intent = classify_intent(question)

    if resolved_files_list:
        if not documents and intent != "chitchat":
            # If active files were selected but Qdrant found NO relevant chunks whatsoever (or below threshold),
            # we short-circuit the process and inform the user directly to save LLM tokens.
            msg = "I can only answer questions related to our documents. The document does not contain this information."
            ai_message = {
                "role": "assistant",
                "content": msg,
                "bound_files": resolved_files_list
            }
            chat_session.messages = list(chat_session.messages) + [ai_message]
            db.add(chat_session)
    
            await db.commit()
            yield f"data: {json.dumps({'text': msg})}\n\n"
            return
            
        if documents:
            # Convert raw strings back to Langchain Document objects for the Reranker
            retrieved_docs = [Document(page_content=doc) for doc in documents]
            try:
                # Use Cohere's Rerank API to re-order the top 15 chunks down to the top 3-5 most highly relevant ones.
                reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
                ranked_docs = reranker.compress_documents(retrieved_docs, question)
                context_docs = [doc.page_content for doc in ranked_docs]
            except Exception as cohere_err:
                logger.warning(f"Cohere Reranker failed: {cohere_err}. Falling back to top 5 initial Qdrant results.")
                context_docs = documents[:5]
        
            # Merge the final reranked chunks into a single context string for the LLM prompt
            context = "\n\n---\n\n".join(context_docs)

    if not resolved_files_list and intent != "chitchat":
        msg = "I can only answer knowledge questions based on documents. Please select a document first."
        ai_message = {
            "role": "assistant",
            "content": msg,
            "bound_files": []
        }
        chat_session.messages = list(chat_session.messages) + [ai_message]
        db.add(chat_session)
        await db.commit()
        yield f"data: {json.dumps({'text': msg})}\n\n"
        return

    # Step 7: Prepare Chat History for the LLM
    # Exclude the very last message since it's the current user question we just appended in Step 4
    db_messages = chat_session.messages[:-1]
    langchain_messages = []
    for msg in db_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        # Map database JSON roles to LangChain's strict Human/AI message classes
        if role in ("user", "human"):
            langchain_messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            langchain_messages.append(AIMessage(content=content))

    # Trim the conversation history to the last 4 messages (2 exchanges) to save tokens and prevent context bloat
    trimmed_messages = trim_messages(
        langchain_messages, max_tokens=4, strategy="last", token_counter=len
    )

    # Initialize the LLM (Gemini 3.1 Flash Lite) with a low temperature for more factual, less creative responses
    logger.info("Calling Gemini API (gemini-3.1-flash-lite) to generate answer...")
    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    
    # Step 8: Build the Dynamic System Prompt
    if resolved_files_list and intent != "chitchat":
        # If files are selected, forcefully instruct the LLM to strictly rely on the provided context
        system_prompt = (
            "Answer using ONLY the provided context. If the answer is not in the text, "
            "reply 'The document does not contain this information.' Do not use outside knowledge.\n\n"
            "Context:\n{context}"
        )
    else:
        # Chitchat intent or no files selected
        system_prompt = (
            "You are a conversational assistant. You may engage in greetings, small talk, and casual conversation. "
            "However, you MUST NOT answer any factual, historical, or general knowledge questions. "
            "If the user asks for facts or general knowledge, politely decline and tell them to upload and select a document to search for that information."
        )

    # Assemble the final prompt chain combining System Instructions, Chat History, and the new User Question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            *trimmed_messages,
            ("human", "{query}"),
        ]
    )

    chain = prompt | model

    # Step 9: Execute LLM generation and stream the output to the client via Server-Sent Events (SSE)
    full_response_text = ""
    try:
        # 'astream' yields text chunks as soon as Gemini produces them, reducing perceived latency
        async for chunk in chain.astream({"context": context, "query": question}):
            content = chunk.content
            # Handle potential multimodal response formats defensively (though Gemini text usually yields strings)
            if isinstance(content, list):
                content = "".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
            else:
                content = str(content)
            
            # Accumulate the full text to save into the database later
            full_response_text += content

            # Yield the chunk back to the client immediately in SSE JSON format
            yield f"data: {json.dumps({'text': content})}\n\n"
    except Exception as stream_err:
        logger.exception("Error occurred during text response generation streaming loop")
        yield f"data: {json.dumps({'error': f'Streaming interrupted: {str(stream_err)}'})}\n\n"
        return

    # Step 10: Persist the AI's response in PostgreSQL
    ai_message = {
        "role": "assistant",
        "content": full_response_text,
        "bound_files": resolved_files_list
    }
    chat_session.messages = list(chat_session.messages) + [ai_message]

    db.add(chat_session)
    await db.commit()
    logger.info("Saved session chat history exchange successfully.")


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

