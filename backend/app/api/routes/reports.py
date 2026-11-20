from fastapi import APIRouter, Query

from app.api.deps import DbDep, SettingsDep
from app.schemas import DataReportOut
from app.services import datasets as ds_svc
from app.services import reports as svc

router = APIRouter(tags=["data doctor"])


@router.get("/datasets/{dataset_id}/data-report", response_model=DataReportOut)
def data_report(dataset_id: str, db: DbDep, settings: SettingsDep, refresh: bool = Query(False)):
    """Run (or return the cached) Data Doctor report for a dataset.

    The report is recomputed automatically when the target column changes.
    """
    dataset = ds_svc.get_dataset(db, dataset_id)
    return svc.data_report_out(svc.get_data_report(db, settings, dataset, refresh=refresh))
