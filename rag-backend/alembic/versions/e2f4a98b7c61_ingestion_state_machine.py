"""Normalize ingestion state-machine statuses."""

from alembic import op
import sqlalchemy as sa


revision = "e2f4a98b7c61"
down_revision = "d7a31cc5f6e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Status columns are VARCHAR in the frozen schema; normalize legacy rows.
    op.execute(
        sa.text(
            "UPDATE document_versions SET status = 'PROCESSING', updated_at = CURRENT_TIMESTAMP "
            "WHERE status = 'PENDING'"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE document_versions SET status = 'PENDING', updated_at = CURRENT_TIMESTAMP "
            "WHERE status = 'PROCESSING'"
        )
    )
