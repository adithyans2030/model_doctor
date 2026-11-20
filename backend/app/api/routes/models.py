from fastapi import APIRouter, BackgroundTasks, Query, Request, status
from sqlalchemy import select

from app.api.deps import DbDep, SettingsDep
from app.db.models import Job, MLModel
from app.schemas import (
    DiagnosisOut, JobOut, ModelOut, ModelSummary, PredictRequest, PredictResponse, TrainRequest,
)
from app.services import datasets as ds_svc
from app.services import jobs as job_svc
from app.services import training as svc
from app.services.diagnosis import build_diagnosis

router = APIRouter(tags=["model doctor"])


def model_summary(m: MLModel) -> dict:
    t = m.metrics["test"]
    return {
        "id": m.id, "project_id": m.project_id, "dataset_id": m.dataset_id, "job_id": m.job_id,
        "name": m.name, "algorithm": m.algorithm, "version": m.version, "source": m.source,
        "stage": m.stage, "created_at": m.created_at, "n_findings": len(m.findings),
        "scores": {
            "f1_macro": t["f1_macro"], "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"],
            "roc_auc": t["roc_auc"], "worst_class": t["worst_class"],
        },
    }


def model_detail(m: MLModel) -> dict:
    return {
        **model_summary(m), "config": m.config, "feature_spec": m.feature_spec, "metrics": m.metrics,
        "findings": m.findings, "importances": m.importances,
    }


@router.post("/datasets/{dataset_id}/train", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def train(dataset_id: str, body: TrainRequest, request: Request, background: BackgroundTasks,
          db: DbDep, settings: SettingsDep):
    """Train baseline models in the background. Poll `GET /jobs/{id}` for progress."""
    dataset = ds_svc.get_dataset(db, dataset_id)
    svc.validate_train_request(dataset, body.drop_columns)
    job = job_svc.create_job(db, dataset, "training", body.model_dump())
    background.add_task(job_svc.run_job, request.app.state.session_factory, settings, job.id, svc.training_worker)
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: DbDep):
    return job_svc.get_job(db, job_id)


@router.get("/datasets/{dataset_id}/jobs", response_model=list[JobOut])
def list_jobs(dataset_id: str, db: DbDep, kind: str | None = None):
    ds_svc.get_dataset(db, dataset_id)
    q = select(Job).where(Job.dataset_id == dataset_id)
    if kind:
        q = q.where(Job.kind == kind)
    return list(db.scalars(q.order_by(Job.created_at.desc())))


@router.get("/datasets/{dataset_id}/models", response_model=list[ModelSummary])
def list_models(dataset_id: str, db: DbDep):
    return [model_summary(m) for m in ds_svc.get_dataset(db, dataset_id).models]


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: str, db: DbDep):
    return model_detail(svc.get_model(db, model_id))


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, body: PredictRequest, db: DbDep, settings: SettingsDep):
    model = svc.get_model(db, model_id)
    return {"model_id": model.id, "predictions": svc.predict_records(settings, model, body.records)}


@router.get("/datasets/{dataset_id}/diagnosis", response_model=DiagnosisOut)
def diagnosis(dataset_id: str, db: DbDep, settings: SettingsDep, model_id: str | None = Query(None)):
    """Full diagnosis: health score, all findings, recommendations and a plain-English summary."""
    return build_diagnosis(db, settings, ds_svc.get_dataset(db, dataset_id), model_id)
