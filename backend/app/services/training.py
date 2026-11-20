"""Phase 3 use cases: baseline training, model registration, prediction."""

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import DomainError, InvalidTargetError, NotFoundError
from app.db.models import Dataset, Job, MLModel, new_id
from app.diagnostics.model_doctor import diagnose_model
from app.ml.algorithms import ALGORITHMS
from app.ml.artifacts import load_bundle, save_bundle
from app.ml.evaluation import predict_labels
from app.ml.features import FeatureSpec, prepare_frame
from app.ml.pipeline import TrainConfig
from app.ml.training import TrainingData, TrainResult, dummy_scores, prepare_training_data, train_and_evaluate
from app.services.datasets import get_dataset, load_dataframe
from app.services import tracking
from app.services.jobs import set_progress
from app.services.reports import get_data_report


def get_model(db: Session, model_id: str) -> MLModel:
    model = db.get(MLModel, model_id)
    if model is None:
        raise NotFoundError(f"Model '{model_id}' not found.")
    return model


def next_version(db: Session, project_id: str) -> int:
    current = db.scalar(select(func.max(MLModel.version)).where(MLModel.project_id == project_id))
    return (current or 0) + 1


def validate_train_request(dataset: Dataset, drop_columns: list[str]) -> None:
    if not dataset.target_column:
        raise InvalidTargetError("Select a target column before training.")
    names = {c["name"] for c in dataset.columns}
    unknown = [c for c in drop_columns if c not in names]
    if unknown:
        raise DomainError(f"Unknown columns in drop_columns: {', '.join(unknown)}.", code="invalid_request")
    if dataset.target_column in drop_columns:
        raise DomainError("The target column can't be dropped.", code="invalid_request")


def register_model(
    db: Session, settings: Settings, dataset: Dataset, result: TrainResult, data: TrainingData,
    data_findings: list[dict], *, job_id: str | None, source: str, name: str | None = None,
) -> MLModel:
    version = next_version(db, dataset.project_id)
    model_id = new_id()
    artifact = save_bundle(
        settings.models_dir, model_id,
        pipeline=result.pipeline, config=result.config.to_dict(), spec=result.spec.to_dict(),
        classes=data.classes, positive_index=data.positive_index,
    )
    model = MLModel(
        id=model_id,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        job_id=job_id,
        name=name or f"{ALGORITHMS[result.config.algorithm].label} v{version}",
        algorithm=result.config.algorithm,
        version=version,
        source=source,
        stage="candidate",
        config=result.config.to_dict(),
        feature_spec=result.spec.to_dict(),
        metrics=result.metrics,
        findings=diagnose_model(result.metrics, result.importances, data_findings),
        importances=result.importances,
        artifact=artifact,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    tracking.log_model(settings, model)
    return model


def leaderboard_row(m: MLModel) -> dict:
    t = m.metrics["test"]
    return {
        "model_id": m.id, "name": m.name, "version": m.version, "algorithm": m.algorithm,
        "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"], "f1_macro": t["f1_macro"],
        "roc_auc": t["roc_auc"], "worst_class": t["worst_class"],
        "cv_f1_macro": (m.metrics.get("cv") or {}).get("val", {}).get("f1_macro"),
    }


def training_worker(db: Session, settings: Settings, job: Job) -> dict:
    p = job.params
    dataset = get_dataset(db, job.dataset_id)
    set_progress(db, job, "Loading data and running the Data Doctor")
    data_report = get_data_report(db, settings, dataset)
    df = load_dataframe(dataset, settings)
    data = prepare_training_data(dataset, df, settings, seed=p["seed"], test_size=p["test_size"])
    dummy = dummy_scores(data)

    models, skipped = [], []
    for i, algo in enumerate(p["algorithms"], 1):
        if algo == "svm" and len(data.y_train) > settings.svm_max_rows:
            skipped.append({"algorithm": algo, "reason": f"more than {settings.svm_max_rows} training rows"})
            continue
        set_progress(db, job, f"Training {ALGORITHMS[algo].label} ({i}/{len(p['algorithms'])})")
        cfg = TrainConfig(algorithm=algo, drop_columns=p.get("drop_columns", []))
        result = train_and_evaluate(cfg, data, cv_folds=p["cv_folds"], n_jobs=settings.n_jobs, dummy=dummy)
        models.append(register_model(
            db, settings, dataset, result, data, data_report.payload["findings"], job_id=job.id, source="baseline",
        ))
    if not models:
        raise DomainError("No models were trained.", code="nothing_trained")

    best = max(models, key=lambda m: m.primary_score)
    return {
        "model_ids": [m.id for m in models],
        "best_model_id": best.id,
        "leaderboard": sorted((leaderboard_row(m) for m in models), key=lambda r: -r["f1_macro"]),
        "skipped": skipped,
        "split": data.split_info(),
        "dummy": dummy,
        "primary_metric": "f1_macro",
    }


# ----------------------------------------------------------------------------- inference
def load_model_bundle(settings: Settings, model: MLModel) -> dict:
    return load_bundle(settings.models_dir, model.artifact)


def predict_records(settings: Settings, model: MLModel, records: list[dict]) -> list[dict]:
    bundle = load_model_bundle(settings, model)
    spec = FeatureSpec.from_dict(bundle["spec"])
    df = pd.DataFrame.from_records(records)
    missing = [c for c in spec.features if c not in df.columns]
    if missing:
        raise DomainError(f"Records are missing feature columns: {', '.join(missing)}.", code="invalid_request")
    proba = bundle["pipeline"].predict_proba(prepare_frame(df, spec))
    pred = predict_labels(proba, bundle["positive_index"], bundle["config"].get("threshold"))
    classes = bundle["classes"]
    return [
        {
            "prediction": classes[int(k)],
            "confidence": round(float(proba[i].max()), 4),
            "probabilities": {c: round(float(proba[i, j]), 4) for j, c in enumerate(classes)},
        }
        for i, k in enumerate(np.asarray(pred))
    ]
