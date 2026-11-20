"""Computing and caching stored reports."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Dataset, Report
from app.diagnostics.data_doctor import run_data_doctor
from app.services.datasets import load_dataframe


def _data_key(dataset: Dataset) -> str:
    return f"target={dataset.target_column or ''}"


def latest_report(db: Session, dataset_id: str, kind: str, *, model_id: str | None = None) -> Report | None:
    q = select(Report).where(Report.dataset_id == dataset_id, Report.kind == kind)
    if model_id:
        q = q.where(Report.model_id == model_id)
    return db.scalars(q.order_by(Report.created_at.desc()).limit(1)).first()


def get_data_report(db: Session, settings: Settings, dataset: Dataset, *, refresh: bool = False) -> Report:
    """Return the cached data report, recomputing it if missing, stale or requested."""
    key = _data_key(dataset)
    report = latest_report(db, dataset.id, "data")
    if report is not None and report.key == key and not refresh:
        return report
    df = load_dataframe(dataset, settings)
    payload = run_data_doctor(df, dataset, settings)
    report = Report(dataset_id=dataset.id, kind="data", key=key, payload=payload)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def data_report_out(report: Report) -> dict:
    return {"id": report.id, "dataset_id": report.dataset_id, "created_at": report.created_at, **report.payload}
