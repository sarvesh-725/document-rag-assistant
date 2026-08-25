"""Add durable scheduling time to outbox events."""

from alembic import op
import sqlalchemy as sa


revision = "d1e5f7a9c3b4"
down_revision = "c7d9e2f4a1b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("outbox_events")}

    if "available_at" not in columns:
        op.add_column(
            "outbox_events",
            sa.Column(
                "available_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
        )
    else:
        # Some databases may already contain this column from an earlier deploy.
        op.alter_column(
            "outbox_events",
            "available_at",
            existing_type=sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        )

    index_names = {
        index["name"] for index in inspector.get_indexes("outbox_events")
    }
    if "ix_outbox_unpublished" in index_names:
        op.drop_index("ix_outbox_unpublished", table_name="outbox_events")
    op.create_index(
        "ix_outbox_unpublished",
        "outbox_events",
        ["published_at", "available_at", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_unpublished", table_name="outbox_events")
    op.create_index(
        "ix_outbox_unpublished",
        "outbox_events",
        ["published_at", "created_at"],
        unique=False,
    )
    op.drop_column("outbox_events", "available_at")
