"""Register a user-supplied model (skops file) and diagnose it like any other model."""

from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import InvalidFileError, InvalidTargetError
from app.db.models import Dataset, MLModel
from app.diagnostics.findings import finding, sort_findings
from app.diagnostics.model_doctor import diagnose_model
from app.ml.external import ExternalModel, load_uploaded_model
from app.ml.pipeline import TrainConfig
from app.ml.training import TrainResult, dummy_scores, evaluate_pipeline, prepare_training_data
from app.services.datasets import load_dataframe
from app.services.reports import get_data_report
from app.services.training import register_model
from app.storage import delete_file, save_stream

_CONTROL = {chr(i) for i in range(32)}


def _filename(name: str | None) -> str:
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = "".join(ch for ch in base if ch not in _CONTROL)
    if Path(base).suffix.lower() != ".skops":
        raise InvalidFileError(
            "Upload a .skops file (save with skops.io.dump). Pickle/joblib files are refused because "
            "loading them can run arbitrary code."
        )
    return base[:255]


def register_uploaded_model(db: Session, settings: Settings, dataset: Dataset, upload: UploadFile,
                            name: str | None = None) -> MLModel:
    if not dataset.target_column:
        raise InvalidTargetError("Select a target column before uploading a model.")
    filename = _filename(upload.filename)
    incoming = settings.data_dir / "incoming"
    stored = save_stream(upload.file, incoming, settings.max_upload_bytes)
    try:
        estimator = load_uploaded_model(stored.path)
    finally:
        delete_file(incoming, stored.stored_filename)  # never keep the raw upload

    df = load_dataframe(dataset, settings)
    data = prepare_training_data(dataset, df, settings)
    wrapper = ExternalModel(estimator, data.task, data.classes)
    cfg = TrainConfig(algorithm="uploaded")
    try:
        test, train = evaluate_pipeline(wrapper, data.spec, cfg, data)
    except InvalidFileError:
        raise
    except Exception as exc:  # noqa: BLE001 - the model doesn't fit this dataset's columns
        raise InvalidFileError(
            f"The uploaded model failed to predict on this dataset's feature columns "
            f"({', '.join(data.spec.features[:6])}…): {type(exc).__name__}: {exc}"[:600]
        ) from exc

    metrics = {
        "task": data.task, "primary_metric": data.primary, "test": test, "train": train, "cv": None,
        "dummy": dummy_scores(data), "split": data.split_info(), "positive_label": data.positive_label,
        "classes": data.classes, "n_features": len(data.spec.features), "fit_seconds": 0.0,
        "external": {"filename": filename, "estimator": type(estimator).__name__},
    }
    report = get_data_report(db, settings, dataset)
    result = TrainResult(cfg, wrapper, data.spec, metrics, None, None)
    model = register_model(db, settings, dataset, result, data, report.payload["findings"], job_id=None,
                           source="uploaded", name=name or f"Uploaded {type(estimator).__name__}")
    model.findings = sort_findings(model.findings + [finding(
        "model.external_evaluation", "model", "info",
        "Uploaded model: evaluated on ModelDoctor's held-out split",
        {"filename": filename, "estimator": type(estimator).__name__, "n_test": test["n"]},
        impact=("ModelDoctor doesn't know which rows this model was trained on. If it saw the test rows "
                "during training, these scores are optimistic and overfitting can't be measured."),
        recommendation="Compare it with a ModelDoctor-trained baseline on the same split (registry → compare).",
    )])
    model.name = f"{model.name} v{model.version}" if not name else model.name
    db.commit()
    db.refresh(model)
    return model
