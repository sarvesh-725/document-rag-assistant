"""Lifecycle transitions and reconciliation for query executions."""

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.enums import QueryRunStatus
from app.database.models import QueryRun
from app.database.repositories import transition_query_run_status


def _now() -> datetime:
    return datetime.utcnow()


async def reconcile_stale_query_runs(
    db: AsyncSession,
    *,
    stale_after: timedelta = timedelta(minutes=15),
    limit: int = 100,
) -> int:
    """Cancel RUNNING query runs that have exceeded the execution deadline."""
    cutoff = _now() - stale_after
    result = await db.execute(
        select(QueryRun)
        .where(
            QueryRun.status == QueryRunStatus.RUNNING.value,
            QueryRun.created_at < cutoff,
        )
        .order_by(QueryRun.created_at.asc())
        .limit(limit)
        .with_for_update()
    )
    stale_runs = list(result.scalars().all())
    for query_run in stale_runs:
        transition_query_run_status(query_run, QueryRunStatus.CANCELLED.value)
    if stale_runs:
        await db.commit()
    return len(stale_runs)
