from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Request, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select

from app.api.deps import DbDep, SettingsDep
from app.core.errors import DomainError
from app.db.models import Job
from app.diagnostics.findings import INTERVENTIONS
from app.schemas import JobOut
from app.services import jobs as job_svc
from app.services.datasets import get_dataset
from app.services.experiments import experiment_worker
from app.services.training import get_model

router = APIRouter(tags=["optimisation"])


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"budget": "standard"}})

    budget: Literal["quick", "standard"] = "standard"
    interventions: list[str] | None = None  # restrict the plan; None = everything the diagnosis suggests

    @field_validator("interventions")
    @classmethod
    def _known(cls, v):
        if v is not None:
            unknown = sorted(set(v) - INTERVENTIONS.keys())
            if unknown:
                raise ValueError(f"unknown interventions: {', '.join(unknown)}")
        return v


@router.post("/models/{model_id}/experiments", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def start_experiment(model_id: str, body: ExperimentRequest, request: Request, background: BackgroundTasks,
                     db: DbDep, settings: SettingsDep):
    """Run the corrective experiments the diagnosis recommends. Poll `GET /jobs/{id}`."""
    model = get_model(db, model_id)
    if model.source == "uploaded":
        raise DomainError("Experiments retrain the model, so they need a ModelDoctor-trained model. "
                          "Train baselines on this dataset and optimise one of those.", code="invalid_request")
    dataset = get_dataset(db, model.dataset_id)
    job = job_svc.create_job(db, dataset, "experiment", {"base_model_id": model.id, **body.model_dump()})
    background.add_task(job_svc.run_job, request.app.state.session_factory, settings, job.id, experiment_worker)
    return job


@router.get("/models/{model_id}/experiments", response_model=list[JobOut])
def list_experiments(model_id: str, db: DbDep):
    model = get_model(db, model_id)
    jobs = db.scalars(
        select(Job).where(Job.dataset_id == model.dataset_id, Job.kind == "experiment").order_by(Job.created_at.desc())
    )
    return [j for j in jobs if j.params.get("base_model_id") == model.id]


@router.get("/interventions")
def interventions():
    """The catalogue of interventions the optimiser can run."""
    return [{"key": k, "label": v} for k, v in INTERVENTIONS.items()]
