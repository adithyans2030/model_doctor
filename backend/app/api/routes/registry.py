from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.api.deps import DbDep
from app.api.routes.models import model_summary
from app.db.models import MLModel
from app.schemas import ModelSummary, Stage
from app.services import registry as svc
from app.services.datasets import get_project
from app.services.training import get_model

router = APIRouter(tags=["registry"])


class StageRequest(BaseModel):
    stage: Stage
    note: str | None = Field(default=None, max_length=2000)
    actor: str | None = Field(default=None, max_length=120)
    acknowledge_risks: bool = False


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    model_id: str
    from_stage: str | None
    to_stage: str
    actor: str
    note: str | None
    created_at: datetime


class RegistryOut(BaseModel):
    project_id: str
    production: ModelSummary | None
    models: list[ModelSummary]
    events: list[EventOut]


def _actor(request: Request, body_actor: str | None) -> str:
    # An authenticated identity (Phase 8) always wins over a self-declared name.
    return getattr(request.state, "actor", None) or (body_actor or "").strip() or "anonymous"


@router.post("/models/{model_id}/stage", response_model=ModelSummary)
def change_stage(model_id: str, body: StageRequest, request: Request, db: DbDep):
    """Move a model through the registry. Production requires prior human approval."""
    model = get_model(db, model_id)
    svc.change_stage(db, model, body.stage, actor=_actor(request, body.actor), note=body.note,
                     acknowledge_risks=body.acknowledge_risks)
    return model_summary(model)


@router.get("/models/{model_id}/events", response_model=list[EventOut])
def model_events(model_id: str, db: DbDep):
    return get_model(db, model_id).events


@router.get("/projects/{project_id}/registry", response_model=RegistryOut)
def registry(project_id: str, db: DbDep):
    get_project(db, project_id)
    models = list(db.scalars(select(MLModel).where(MLModel.project_id == project_id).order_by(MLModel.version.desc())))
    production = next((m for m in models if m.stage == "production"), None)
    events = sorted((e for m in models for e in m.events), key=lambda e: e.created_at, reverse=True)
    return {"project_id": project_id, "production": production and model_summary(production),
            "models": [model_summary(m) for m in models], "events": events[:100]}


@router.get("/models/compare/{a_id}/{b_id}")
def compare(a_id: str, b_id: str, db: DbDep) -> dict[str, Any]:
    """Compare two model versions trained on the same dataset (b relative to a)."""
    return svc.compare_models(get_model(db, a_id), get_model(db, b_id))
