import os
from typing import List
import io
import hashlib
import uuid
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages, messages_from_dict, messages_to_dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.intent_classifier import classify_intent

load_dotenv(override=True)

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "document_chunks")

MODEL_NAME = "models/gemini-embedding-2-preview"

client = AsyncQdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

_embeddings = None
_qdrant_initialized = False


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the embedding model instance, fallback to a standard model if
    the requested one fails, and cache it.
    """
    global MODEL_NAME, _embeddings
    if _embeddings is not None:
        return _embeddings

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set in the environment or .env file."
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_NAME)
        embeddings.embed_query("test connection")
        _embeddings = embeddings
        return _embeddings
    except Exception as e:
        print(f"Model initialization failed for {MODEL_NAME} with: {e}")
        fallback_model = "models/text-embedding-004"
        print(f"Attempting fallback to: {fallback_model}")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=fallback_model)
            embeddings.embed_query("test connection")
            MODEL_NAME = fallback_model
            _embeddings = embeddings
            return _embeddings
        except Exception as fallback_err:
            print(f"Fallback model also failed: {fallback_err}")
            raise e


async def init_qdrant():
    """Checks if collection exists in Qdrant and programmatically creates it if not."""
    global _qdrant_initialized
    if _qdrant_initialized:
        return

    try:
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if QDRANT_COLLECTION_NAME not in collection_names:
            embeddings = _get_embeddings()
            sample_vector = embeddings.embed_query("test connection")
            vector_size = len(sample_vector)

            await client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size, distance=qdrant_models.Distance.COSINE
                ),
            )
            print(
                f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created with size {vector_size}."
            )

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="user_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            print(f"Index creation for 'user_id' failed (may already exist): {idx_err}")

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="filename",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            print(
                f"Index creation for 'filename' failed (may already exist): {idx_err}"
            )

        try:
            await client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="file_hash",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_err:
            print(
                f"Index creation for 'file_hash' failed (may already exist): {idx_err}"
            )

        _qdrant_initialized = True
    except Exception as e:
        print(f"Failed to check or initialize Qdrant collection: {e}")


from langchain_text_splitters import TokenTextSplitter

async def process_and_store_document(
    file_bytes: bytes, filename: str, user_id: str
) -> int:
    """
    Takes raw file bytes from an upload, parses the text using unstructured,
    generates an MD5 hash of the bytes, chunks the text into Parent and Child chunks,
    generates embeddings for Children, and saves them to Qdrant Cloud.
    """
    if not file_bytes:
        raise ValueError("File content is empty.")

    await init_qdrant()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    try:
        loader = UnstructuredLoader(
            file=io.BytesIO(file_bytes),
            api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
            url=os.environ.get("UNSTRUCTURED_API_URL"),
            partition_via_api=True,
            strategy="hi_res",
        )
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        if not text.strip():
            raise ValueError("Extracted text is empty or could not be parsed.")
    except Exception as e:
        raise RuntimeError(f"Error parsing document with Unstructured loader: {e}")

    # Parent-Child Chunking Strategy
    parent_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=100)
    child_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)

    parents = parent_splitter.split_text(text)
    
    child_chunks = []
    parent_texts = []
    
    for parent in parents:
        children = child_splitter.split_text(parent)
        for child in children:
            child_chunks.append(child)
            parent_texts.append(parent)

    if not child_chunks:
        return 0

    embeddings = _get_embeddings()
    embeddings_list = embeddings.embed_documents(child_chunks)

    try:
        purge_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
                ),
                qdrant_models.FieldCondition(
                    key="filename", match=qdrant_models.MatchValue(value=filename)
                ),
            ],
            must_not=[
                qdrant_models.FieldCondition(
                    key="file_hash", match=qdrant_models.MatchValue(value=file_hash)
                )
            ],
        )
        await client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=purge_filter),
        )
    except Exception as e:
        print(f"Qdrant cleanup of outdated chunks failed or was empty: {e}")

    points = []
    for i, (child, parent, vector) in enumerate(zip(child_chunks, parent_texts, embeddings_list)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_hash}_{i}"))

        payload = {
            "user_id": str(user_id),
            "filename": filename,
            "file_hash": file_hash,
            "source": filename,
            "chunk_id": i,
            "text": child,
            "child_text": child,
            "parent_text": parent,
            "intent_tag": "knowledge"
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    await client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    return len(child_chunks)


async def query_rag_system(
    question: str,
    user_id: int,
    session_id: str,
    db: AsyncSession,
    include_prev_files: bool = True,
    explicit_files: List[str] = None,
) -> str:
    """
    Takes a user query, retrieves relevant chunks from Qdrant based on session/explicit files,
    constructs the prompt template including sliding window memory,
    sends it to gemini-2.5-flash, saves history to DB, and returns the final answer text.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if explicit_files is None:
        explicit_files = []

    from app.database.models import ChatSession, SessionFile

    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
    )
    chat_session = result.scalars().first()
    if not chat_session:
        raise ValueError(f"Session {session_id} not found or unauthorized.")

    file_result = await db.execute(
        select(SessionFile).where(SessionFile.session_id == session_id)
    )
    session_files = file_result.scalars().all()

    just_uploaded_names = {f.filename for f in session_files if f.just_uploaded}
    committed_names = {f.filename for f in session_files if f.is_committed}

    if include_prev_files:
        base_set = committed_names | set(explicit_files)
    else:
        base_set = set(explicit_files)

    resolved_set = base_set | just_uploaded_names
    resolved_files_list = list(resolved_set)

    must_conditions = [
        qdrant_models.FieldCondition(
            key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
        )
    ]

    if resolved_files_list:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="filename", match=qdrant_models.MatchAny(any=resolved_files_list)
            )
        )

    search_filter = qdrant_models.Filter(must=must_conditions)

    embeddings = _get_embeddings()
    query_vector = embeddings.embed_query(question)

    await init_qdrant()
    
    intent = classify_intent(question)
    
    if intent == "chitchat":
        # Fall back to empty documents and conversational response
        documents = []
    else:
        try:
            search_result = await client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_vector,
                query_filter=search_filter,
                limit=15,
            )
        except Exception as e:
            raise ValueError(
                f"Could not query Qdrant collection '{QDRANT_COLLECTION_NAME}'. Has ingestion been run? Detail: {e}"
            )
    
        valid_hits = [hit for hit in search_result.points if hit.payload and hit.score >= 0.60]
        
        documents = []
        for hit in valid_hits:
            if intent == "knowledge_specific":
                text_content = hit.payload.get("child_text", hit.payload.get("text", ""))
            else:
                text_content = hit.payload.get("parent_text", hit.payload.get("text", ""))
            documents.append(text_content)
            
    if not documents and intent != "chitchat":
        for f in session_files:
            if f.just_uploaded:
                f.just_uploaded = False
                f.is_committed = True
        db.add_all(session_files)
        await db.commit()
        return "I can only answer questions related to our documents. The document does not contain this information."

    if documents:
        retrieved_docs = [Document(page_content=doc) for doc in documents]
    
        try:
            reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
            ranked_docs = reranker.compress_documents(retrieved_docs, question)
            context_docs = [doc.page_content for doc in ranked_docs]
        except Exception as cohere_err:
            print(
                f"Cohere Reranker failed: {cohere_err}. Falling back to top 5 initial Qdrant results."
            )
            context_docs = documents[:5]
    
        context = "\n\n---\n\n".join(context_docs)
    else:
        context = ""

    db_messages = chat_session.messages
    langchain_messages = messages_from_dict(db_messages)
    history = InMemoryChatMessageHistory(messages=langchain_messages)

    trimmed_messages = trim_messages(
        history.messages, max_tokens=4, strategy="last", token_counter=len
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    if intent != "chitchat":
        system_prompt = (
            "Answer using ONLY the provided context. If the answer is not in the text, "
            "reply 'The document does not contain this information.' Do not use outside knowledge.\n\n"
            "Context:\n{context}"
        )
    else:
        system_prompt = "You are a helpful assistant. Answer the user's question conversationally."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            *trimmed_messages,
            ("human", "{query}"),
        ]
    )

    chain = prompt | model
    response = await chain.ainvoke({"context": context, "query": question})

    answer_text = response.content
    if isinstance(answer_text, list):
        texts = [
            item.get("text", "")
            for item in answer_text
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        answer_text = "".join(texts)
    else:
        answer_text = str(answer_text)

    history.add_user_message(question)
    history.add_ai_message(answer_text)

    chat_session.messages = messages_to_dict(history.messages)

    for f in session_files:
        if f.just_uploaded:
            f.just_uploaded = False
            f.is_committed = True

    db.add(chat_session)
    db.add_all(session_files)
    await db.commit()

    return answer_text


async def delete_file_from_db(user_id: str, filename: str):
    """
    Deletes all chunks associated with user_id and filename from Qdrant.
    """
    await init_qdrant()
    try:
        delete_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id", match=qdrant_models.MatchValue(value=str(user_id))
                ),
                qdrant_models.FieldCondition(
                    key="filename", match=qdrant_models.MatchValue(value=filename)
                ),
            ]
        )
        await client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=delete_filter),
        )
    except Exception as e:
        print(f"Error deleting file {filename} for user {user_id} from Qdrant: {e}")


async def undo_file_upload(
    user_id: int, session_id: str, filename: str, db: AsyncSession
) -> bool:
    """
    If the file has not been committed yet (just_uploaded is True),
    delete it from database and remove its chunks from Qdrant.
    Otherwise, return False.
    """
    from app.database.models import SessionFile

    result = await db.execute(
        select(SessionFile).where(
            SessionFile.session_id == session_id, SessionFile.filename == filename
        )
    )
    session_file = result.scalars().first()

    if session_file and session_file.just_uploaded:
        await db.delete(session_file)
        await db.commit()
        await delete_file_from_db(str(user_id), filename)
        return True
    return False
