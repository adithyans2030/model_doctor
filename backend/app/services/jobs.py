"""Background job execution.

Jobs run in FastAPI's background-task thread pool after the response is sent,
each with its own DB session. Status is persisted so the dashboard can poll
``GET /jobs/{id}``. In tests, Starlette runs background tasks before the
request returns, which makes job tests deterministic.

For production scale this is the seam to swap in a real queue (Celery/RQ/Arq):
``run_job`` is already a plain function of (session factory, settings, job id).
"""

import logging
from collections.abc import Callable

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import DomainError, NotFoundError
from app.db.models import Dataset, Job, utcnow
from app.monitoring.metrics import REGISTRY

log = logging.getLogger("modeldoctor.jobs")

Worker = Callable[[Session, Settings, Job], dict]


def create_job(db: Session, dataset: Dataset, kind: str, params: dict) -> Job:
    job = Job(dataset_id=dataset.id, kind=kind, params=params, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"Job '{job_id}' not found.")
    return job


def set_progress(db: Session, job: Job, message: str) -> None:
    job.progress = message[:255]
    db.commit()


def run_job(session_factory, settings: Settings, job_id: str, worker: Worker) -> None:
    db: Session = session_factory()
    job = db.get(Job, job_id)
    if job is None:
        db.close()
        return
    try:
        job.status, job.started_at = "running", utcnow()
        db.commit()
        job.result = worker(db, settings, job)
        job.status, job.progress = "completed", "done"
    except DomainError as exc:
        db.rollback()
        job.status, job.error = "failed", exc.message
    except Exception as exc:  # noqa: BLE001 - a job must never crash the worker silently
        db.rollback()
        log.exception("job %s failed", job_id)
        job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        job.finished_at = utcnow()
        db.commit()
        REGISTRY.inc("modeldoctor_jobs_total", {"kind": job.kind, "status": job.status},
                     help_="Finished background jobs by kind and status.")
        db.close()


def recover_interrupted(session_factory) -> int:
    """Mark jobs left running by a previous process as failed."""
    with session_factory() as db:
        res = db.execute(
            update(Job)
            .where(Job.status.in_(["queued", "running"]))
            .values(status="failed", error="Interrupted by a server restart.", finished_at=utcnow())
        )
        db.commit()
        return res.rowcount or 0
