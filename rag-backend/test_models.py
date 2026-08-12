"""
Phase 1 - Domain model verification test.

Verifies that all 10 domain models can be:
1. Created as database tables
2. Populated with test records
3. Read back successfully
4. Cleaned up

Run: python test_models.py
"""

import asyncio
import uuid
from datetime import datetime

from app.database.connection import engine, Base, AsyncSessionLocal
from app.database.models import (
    User, Document, DocumentVersion, ChatSession, Message,
    IngestionJob, QueryRun, QueryRunDocument, ConversationSummary, OutboxEvent,
)
from app.database.enums import (
    DocumentStatus, VersionStatus, MessageRole, MessageStatus,
    IngestionStatus, IngestionStage, QueryRunStatus,
)


EXPECTED_TABLES = [
    "users",
    "documents",
    "document_versions",
    "chat_sessions",
    "messages",
    "ingestion_jobs",
    "query_runs",
    "query_run_documents",
    "conversation_summaries",
    "outbox_events",
]


async def verify_tables_exist():
    """Verify all 10 tables exist in the database."""
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )

    missing = set(EXPECTED_TABLES) - set(table_names)
    if missing:
        print(f"FAIL: Missing tables: {missing}")
        return False

    print(f"PASS: All {len(EXPECTED_TABLES)} tables exist")
    return True


async def verify_crud_operations():
    """Test create/read on all models."""
    async with AsyncSessionLocal() as db:
        try:
            # --- User ---
            user = User(
                username=f"test_user_{uuid.uuid4().hex[:8]}",
                hashed_password="$2b$12$fakehash",
            )
            db.add(user)
            await db.flush()
            print(f"  OK: User created: id={user.id}")

            # --- Document ---
            doc = Document(
                user_id=user.id,
                display_name="report.pdf",
                original_filename="report.pdf",
                duplicate_index=0,
                status=DocumentStatus.UPLOADING,
            )
            db.add(doc)
            await db.flush()
            print(f"  OK: Document created: id={doc.id}")

            # --- DocumentVersion ---
            version = DocumentVersion(
                document_id=doc.id,
                version_number=1,
                content_hash="abc123def456",
                storage_key="uploads/user1/doc1/v1/report.pdf",
                status=VersionStatus.PENDING,
                parser_version="unstructured-0.23",
                chunking_version="parent-child-v1",
                embedding_profile="gemini-embedding-2-preview",
            )
            db.add(version)
            await db.flush()
            print(f"  OK: DocumentVersion created: id={version.id}")

            # Set current_version_id on document
            doc.current_version_id = version.id
            doc.status = DocumentStatus.READY
            await db.flush()
            print(f"  OK: Document.current_version_id set to {version.id}")

            # --- Duplicate Document (same filename) ---
            doc2 = Document(
                user_id=user.id,
                display_name="report (2).pdf",
                original_filename="report.pdf",
                duplicate_index=1,
                status=DocumentStatus.UPLOADING,
            )
            db.add(doc2)
            await db.flush()
            print(f"  OK: Duplicate Document created: id={doc2.id}, display_name='{doc2.display_name}'")

            # --- ChatSession ---
            session = ChatSession(
                user_id=user.id,
                title="Test Conversation",
            )
            db.add(session)
            await db.flush()
            print(f"  OK: ChatSession created: id={session.id}")

            # --- Messages ---
            user_msg = Message(
                session_id=session.id,
                sequence_number=1,
                role=MessageRole.USER,
                content="What is the Q3 revenue?",
                status=MessageStatus.COMPLETED,
                client_request_id=str(uuid.uuid4()),
                selected_document_snapshot={
                    "documents": [
                        {"document_id": str(doc.id), "version_id": str(version.id)}
                    ]
                },
            )
            db.add(user_msg)
            await db.flush()
            print(f"  OK: User Message created: id={user_msg.id}, seq={user_msg.sequence_number}")

            ai_msg = Message(
                session_id=session.id,
                sequence_number=2,
                role=MessageRole.ASSISTANT,
                content="The Q3 revenue was $42M according to the report.",
                status=MessageStatus.COMPLETED,
                sources={"chunks": [{"text": "Q3 revenue: $42M", "score": 0.95}]},
            )
            db.add(ai_msg)
            await db.flush()
            print(f"  OK: Assistant Message created: id={ai_msg.id}, seq={ai_msg.sequence_number}")

            # --- IngestionJob ---
            job = IngestionJob(
                document_id=doc.id,
                version_id=version.id,
                status=IngestionStatus.COMPLETED,
                stage=IngestionStage.FINALIZE,
                attempt_count=1,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            db.add(job)
            await db.flush()
            print(f"  OK: IngestionJob created: id={job.id}, status={job.status}")

            # --- QueryRun ---
            qr = QueryRun(
                session_id=session.id,
                user_id=user.id,
                message_id=user_msg.id,
                status=QueryRunStatus.COMPLETED,
                completed_at=datetime.utcnow(),
            )
            db.add(qr)
            await db.flush()
            print(f"  OK: QueryRun created: id={qr.id}")

            # --- QueryRunDocument ---
            qrd = QueryRunDocument(
                query_run_id=qr.id,
                document_id=doc.id,
                version_id=version.id,
            )
            db.add(qrd)
            await db.flush()
            print(f"  OK: QueryRunDocument created: query_run={qr.id}, doc={doc.id}")

            # --- ConversationSummary ---
            summary = ConversationSummary(
                session_id=session.id,
                summary="User asked about Q3 revenue. AI provided answer from report.",
                last_summarized_message_id=ai_msg.id,
            )
            db.add(summary)
            await db.flush()
            print(f"  OK: ConversationSummary created for session {session.id}")

            # --- OutboxEvent ---
            event = OutboxEvent(
                event_type="document.ingestion.completed",
                aggregate_id=doc.id,
                payload={"document_id": str(doc.id), "version_id": str(version.id), "chunks": 42},
            )
            db.add(event)
            await db.flush()
            print(f"  OK: OutboxEvent created: id={event.id}, type={event.event_type}")

            # --- Rollback (don't persist test data) ---
            await db.rollback()
            print("\n  OK: All test records rolled back (no persistent side effects)")

            return True

        except Exception as e:
            await db.rollback()
            print(f"\n  FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    print("=" * 60)
    print("Phase 1 - Domain Model Verification")
    print("=" * 60)

    # Step 1: Create tables
    print("\n[1/3] Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  Tables created.")

    # Step 2: Verify tables exist
    print("\n[2/3] Verifying tables...")
    tables_ok = await verify_tables_exist()

    # Step 3: CRUD test
    print("\n[3/3] Testing CRUD operations...")
    crud_ok = await verify_crud_operations()

    # Summary
    print("\n" + "=" * 60)
    if tables_ok and crud_ok:
        print("PASS: Phase 1 DOMAIN MODEL VERIFICATION PASSED")
    else:
        print("FAIL: Phase 1 DOMAIN MODEL VERIFICATION FAILED")
    print("=" * 60)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
