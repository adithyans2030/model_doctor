"""Dataset use cases: ingest, load, preview, set target, delete."""

import json

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import InvalidTargetError, NotFoundError
from app.db.models import Dataset, Project
from app.services import ingestion, profiling
from app.services.target import TargetAnalysis, analyze_target
from app.ml.artifacts import delete_bundle
from app.storage import delete_file, save_stream


def get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError(f"Project '{project_id}' not found.")
    return project


def get_dataset(db: Session, dataset_id: str) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"Dataset '{dataset_id}' not found.")
    return dataset


def load_dataframe(
    dataset: Dataset, settings: Settings, *, usecols: list[str] | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Load a stored dataset. Later phases (Data/Model Doctor) use this as their entry point."""
    path = settings.uploads_dir / dataset.stored_filename
    if not path.exists():
        raise NotFoundError(f"The file for dataset '{dataset.id}' is missing from storage.")
    fmt = ingestion.CsvFormat(
        encoding=dataset.encoding,
        delimiter=dataset.delimiter,
        header=ingestion.read_header(path, dataset.encoding, dataset.delimiter),
    )
    return ingestion.read_csv(path, fmt, usecols=usecols, nrows=nrows)


def ingest_upload(db: Session, settings: Settings, project: Project, upload: UploadFile) -> Dataset:
    filename = ingestion.sanitize_filename(upload.filename)
    stored = save_stream(upload.file, settings.uploads_dir, settings.max_upload_bytes)
    try:
        df, fmt = ingestion.parse_upload(stored.path, stored.head, settings)
        dataset = Dataset(
            project_id=project.id,
            original_filename=filename,
            stored_filename=stored.stored_filename,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            encoding=fmt.encoding,
            delimiter=fmt.delimiter,
            n_rows=len(df),
            n_cols=df.shape[1],
            columns=profiling.profile_columns(df),
        )
        db.add(dataset)
        db.commit()
    except BaseException:
        db.rollback()
        delete_file(settings.uploads_dir, stored.stored_filename)
        raise
    db.refresh(dataset)
    return dataset


def preview(dataset: Dataset, settings: Settings, n: int) -> dict:
    df = load_dataframe(dataset, settings, nrows=n)
    data = json.loads(df.to_json(orient="split", date_format="iso", default_handler=str))
    return {
        "columns": data["columns"],
        "rows": data["data"],
        "n_rows_shown": len(df),
        "total_rows": dataset.n_rows,
    }


def set_target(db: Session, settings: Settings, dataset: Dataset, column: str) -> TargetAnalysis:
    profile = next((c for c in dataset.columns if c["name"] == column), None)
    if profile is None:
        raise InvalidTargetError(f"Column '{column}' does not exist in this dataset.")
    series = load_dataframe(dataset, settings, usecols=[column])[column]
    analysis = analyze_target(series, profile, settings)

    dataset.target_column = column
    dataset.task_type = analysis.task_type
    dataset.class_distribution = analysis.class_distribution
    db.commit()
    db.refresh(dataset)
    return analysis


def _artifacts(datasets: list[Dataset]) -> list[str]:
    return [m.artifact for d in datasets for m in d.models]


def delete_dataset(db: Session, settings: Settings, dataset: Dataset) -> None:
    stored, artifacts = dataset.stored_filename, _artifacts([dataset])
    db.delete(dataset)
    db.commit()
    delete_file(settings.uploads_dir, stored)
    for a in artifacts:
        delete_bundle(settings.models_dir, a)


def delete_project(db: Session, settings: Settings, project: Project) -> None:
    stored = [d.stored_filename for d in project.datasets]
    artifacts = _artifacts(project.datasets)
    db.delete(project)
    db.commit()
    for name in stored:
        delete_file(settings.uploads_dir, name)
    for a in artifacts:
        delete_bundle(settings.models_dir, a)
