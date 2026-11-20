from fastapi import APIRouter, File, Query, UploadFile, status

from app.api.deps import DbDep, SettingsDep
from app.schemas import DatasetOut, DatasetPreview, DatasetSummary, TargetRequest, TargetResponse
from app.services import datasets as svc

router = APIRouter(tags=["datasets"])


@router.post(
    "/projects/{project_id}/datasets",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
)
def upload_dataset(project_id: str, db: DbDep, settings: SettingsDep, file: UploadFile = File(...)):
    project = svc.get_project(db, project_id)
    return svc.ingest_upload(db, settings, project, file)


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetSummary])
def list_datasets(project_id: str, db: DbDep):
    return svc.get_project(db, project_id).datasets


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: DbDep):
    return svc.get_dataset(db, dataset_id)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    db: DbDep,
    settings: SettingsDep,
    rows: int = Query(20, ge=1, le=100),
):
    return svc.preview(svc.get_dataset(db, dataset_id), settings, rows)


@router.post("/datasets/{dataset_id}/target", response_model=TargetResponse)
def set_target(dataset_id: str, body: TargetRequest, db: DbDep, settings: SettingsDep):
    dataset = svc.get_dataset(db, dataset_id)
    analysis = svc.set_target(db, settings, dataset, body.column)
    return TargetResponse(dataset=DatasetOut.model_validate(dataset), warnings=analysis.warnings)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, db: DbDep, settings: SettingsDep):
    svc.delete_dataset(db, settings, svc.get_dataset(db, dataset_id))
