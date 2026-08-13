"""Store canonical parent chunks for each document version."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f3b7c9d1e245"
down_revision = "e2f4a98b7c61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_parents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("parent_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(), nullable=True),
        sa.Column("element_type", sa.String(), nullable=True),
        sa.Column("source_position", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("parser_version", sa.String(length=50), nullable=False),
        sa.Column("chunking_version", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id", "parent_index", name="uq_document_parent_index"),
    )
    op.create_index("ix_document_parents_version_id", "document_parents", ["version_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_document_parents_version_id", table_name="document_parents")
    op.drop_table("document_parents")
