"""Link assistant messages to their originating user message."""

from alembic import op
import sqlalchemy as sa


revision = "c7d9e2f4a1b3"
down_revision = "a4c8e1f2b6d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("parent_message_id", sa.UUID(), nullable=True))
    op.create_index("ix_messages_parent_message_id", "messages", ["parent_message_id"], unique=False)
    op.create_foreign_key(
        "messages_parent_message_id_fkey",
        "messages",
        "messages",
        ["parent_message_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("messages_parent_message_id_fkey", "messages", type_="foreignkey")
    op.drop_index("ix_messages_parent_message_id", table_name="messages")
    op.drop_column("messages", "parent_message_id")
