"""Add database cascades for deleting a session aggregate.

Revision ID: d7a31cc5f6e2
Revises: 6ef774397fac
"""

from alembic import op


revision = "d7a31cc5f6e2"
down_revision = "6ef774397fac"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("messages_session_id_fkey", "messages", type_="foreignkey")
    op.create_foreign_key("messages_session_id_fkey", "messages", "chat_sessions", ["session_id"], ["id"], ondelete="CASCADE")
    op.drop_constraint("query_runs_session_id_fkey", "query_runs", type_="foreignkey")
    op.create_foreign_key("query_runs_session_id_fkey", "query_runs", "chat_sessions", ["session_id"], ["id"], ondelete="CASCADE")
    op.drop_constraint("conversation_summaries_session_id_fkey", "conversation_summaries", type_="foreignkey")
    op.create_foreign_key("conversation_summaries_session_id_fkey", "conversation_summaries", "chat_sessions", ["session_id"], ["id"], ondelete="CASCADE")
    op.drop_constraint("query_run_documents_query_run_id_fkey", "query_run_documents", type_="foreignkey")
    op.create_foreign_key("query_run_documents_query_run_id_fkey", "query_run_documents", "query_runs", ["query_run_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    op.drop_constraint("query_run_documents_query_run_id_fkey", "query_run_documents", type_="foreignkey")
    op.create_foreign_key("query_run_documents_query_run_id_fkey", "query_run_documents", "query_runs", ["query_run_id"], ["id"])
    op.drop_constraint("conversation_summaries_session_id_fkey", "conversation_summaries", type_="foreignkey")
    op.create_foreign_key("conversation_summaries_session_id_fkey", "conversation_summaries", "chat_sessions", ["session_id"], ["id"])
    op.drop_constraint("query_runs_session_id_fkey", "query_runs", type_="foreignkey")
    op.create_foreign_key("query_runs_session_id_fkey", "query_runs", "chat_sessions", ["session_id"], ["id"])
    op.drop_constraint("messages_session_id_fkey", "messages", type_="foreignkey")
    op.create_foreign_key("messages_session_id_fkey", "messages", "chat_sessions", ["session_id"], ["id"])
