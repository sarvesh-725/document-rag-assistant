"""Index QueryRun lifecycle reconciliation queries."""

from alembic import op


revision = "a4c8e1f2b6d0"
down_revision = "f3b7c9d1e245"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_query_runs_status_created",
        "query_runs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_query_runs_status_created", table_name="query_runs")
