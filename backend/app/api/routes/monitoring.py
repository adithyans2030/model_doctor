from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, Query, UploadFile, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.api.deps import DbDep, SettingsDep
from app.db.models import AuditLog, Job, MLModel, Report
from app.monitoring.metrics import REGISTRY, gauge_lines
from app.services import monitoring as svc
from app.services.training import get_model

router = APIRouter(tags=["monitoring"])
metrics_router = APIRouter(tags=["monitoring"])


class DriftReportOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    model_id: str
    created_at: datetime
    report: dict[str, Any]


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    actor: str
    method: str
    path: str
    status_code: int
    client_ip: str
    created_at: datetime


def _out(r: Report) -> dict:
    return {"id": r.id, "model_id": r.model_id, "created_at": r.created_at, "report": r.payload}


@router.post("/models/{model_id}/monitor", response_model=DriftReportOut, status_code=status.HTTP_201_CREATED)
def monitor(model_id: str, db: DbDep, settings: SettingsDep, file: UploadFile = File(...)):
    """Upload a production batch (CSV with the model's feature columns, target optional) to check drift."""
    return _out(svc.monitor_batch(db, settings, get_model(db, model_id), file))


@router.get("/models/{model_id}/drift", response_model=list[DriftReportOut])
def drift(model_id: str, db: DbDep, limit: int = Query(20, ge=1, le=100)):
    return [_out(r) for r in svc.drift_history(db, get_model(db, model_id), limit)]


@router.get("/audit-logs", response_model=list[AuditOut])
def audit_logs(db: DbDep, limit: int = Query(100, ge=1, le=1000), actor: str | None = None):
    q = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if actor:
        q = q.where(AuditLog.actor == actor)
    return list(db.scalars(q))


@metrics_router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics(db: DbDep):
    """Prometheus scrape endpoint."""
    lines = REGISTRY.render()
    stage_counts = db.execute(select(MLModel.stage, func.count()).group_by(MLModel.stage)).all()
    lines += gauge_lines("modeldoctor_models", "Registered models by stage.",
                         [({"stage": s}, n) for s, n in stage_counts])
    job_counts = db.execute(select(Job.kind, Job.status, func.count()).group_by(Job.kind, Job.status)).all()
    lines += gauge_lines("modeldoctor_jobs", "Jobs by kind and current status.",
                         [({"kind": k, "status": s}, n) for k, s, n in job_counts])
    watched = list(db.scalars(select(MLModel).where(MLModel.stage.in_(["production", "approved"]))))
    lines += gauge_lines("modeldoctor_model_f1_macro", "Test macro-F1 of approved/production models.",
                         [({"model": m.name, "model_id": m.id, "stage": m.stage}, m.primary_score) for m in watched])
    drift_samples = []
    for m in watched:
        r = db.scalars(select(Report).where(Report.model_id == m.id, Report.kind == "drift")
                       .order_by(Report.created_at.desc()).limit(1)).first()
        if r:
            drift_samples.append(({"model": m.name, "model_id": m.id}, r.payload["score"]))
    lines += gauge_lines("modeldoctor_drift_score", "Latest drift score (100 = no drift).", drift_samples)
    return "\n".join(lines) + "\n"
