"""initial: projects and datasets

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-20
"""
import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("stored_filename", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("encoding", sa.String(32), nullable=False),
        sa.Column("delimiter", sa.String(4), nullable=False),
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("n_cols", sa.Integer(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("target_column", sa.String(255), nullable=True),
        sa.Column("task_type", sa.String(32), nullable=True),
        sa.Column("class_distribution", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_datasets_project_id", "datasets", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_datasets_project_id", table_name="datasets")
    op.drop_table("datasets")
    op.drop_table("projects")
