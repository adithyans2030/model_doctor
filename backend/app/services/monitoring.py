"""Phase 8 use cases: drift reports for production batches."""

import pandas as pd
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import DomainError
from app.db.models import MLModel, Report
from app.ml.evaluation import classification_metrics, predict_labels
from app.ml.features import FeatureSpec, prepare_frame
from app.monitoring.drift import class_mix_psi, drift_report, feature_drift
from app.services import ingestion
from app.services.explanations import holdout
from app.services.target import normalize_labels
from app.storage import delete_file, save_stream


def _read_batch(settings: Settings, upload: UploadFile) -> tuple[str, pd.DataFrame]:
    name = ingestion.sanitize_filename(upload.filename)
    batches = settings.data_dir / "batches"
    stored = save_stream(upload.file, batches, settings.max_upload_bytes)
    try:
        df, _ = ingestion.parse_upload(stored.path, stored.head, settings)
    finally:
        # Only the computed report is kept; raw production data isn't retained.
        delete_file(batches, stored.stored_filename)
    return name, df


def monitor_batch(db: Session, settings: Settings, model: MLModel, upload: UploadFile) -> Report:
    filename, batch = _read_batch(settings, upload)
    dataset, _, bundle, data = holdout(db, settings, model)
    spec = FeatureSpec.from_dict(bundle["spec"])

    missing = [c for c in spec.features if c not in batch.columns]
    if missing:
        raise DomainError(f"The batch is missing feature columns the model needs: {', '.join(missing)}.",
                          code="schema_mismatch")
    target = spec.target
    expected = set(spec.features) | set(spec.excluded) | {target}  # excluded-by-design columns are fine
    extra = sorted(set(batch.columns) - expected)
    schema = {"missing_columns": [], "extra_columns": extra, "has_labels": target in batch.columns}

    cur = prepare_frame(batch, spec)
    ref = data.X_train[spec.features]
    features = feature_drift(ref, cur, spec)

    pipe, pos, classes = bundle["pipeline"], bundle["positive_index"], bundle["classes"]
    threshold = bundle["config"].get("threshold")
    test_pred = predict_labels(pipe.predict_proba(data.X_test[spec.features]), pos, threshold)
    proba = pipe.predict_proba(cur)
    batch_pred = predict_labels(proba, pos, threshold)
    prediction = class_mix_psi(test_pred, batch_pred, classes)

    performance = None
    if schema["has_labels"] and batch[target].notna().sum() >= 10:
        profile = next(c for c in dataset.columns if c["name"] == target)
        labelled = batch[target].notna()
        labels = normalize_labels(batch.loc[labelled, target], profile, settings)
        known = labels.isin(classes)
        code = {c: i for i, c in enumerate(classes)}
        y = labels[known].map(code).to_numpy()
        if len(y) >= 10:
            mask = labelled.to_numpy().nonzero()[0][known.to_numpy()]
            m = classification_metrics(y, proba[mask], classes, pos, threshold)
            test_f1 = model.metrics["test"]["f1_macro"]
            performance = {
                "n_labelled": int(len(y)), "unknown_labels": int((~known).sum()),
                "test_f1_macro": test_f1, "batch_f1_macro": m["f1_macro"],
                "delta_f1_macro": round(m["f1_macro"] - test_f1, 4),
                "batch_accuracy": m["accuracy"], "batch_balanced_accuracy": m["balanced_accuracy"],
                "batch_minority_recall": (m.get("positive_class") or m["worst_class"])["recall"],
            }

    payload = drift_report(features, prediction, performance, schema)
    payload.update(batch={"filename": filename, "n_rows": int(len(batch))}, model_id=model.id)
    report = Report(dataset_id=dataset.id, model_id=model.id, kind="drift", key=filename, payload=payload)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def drift_history(db: Session, model: MLModel, limit: int = 20) -> list[Report]:
    return list(db.scalars(
        select(Report).where(Report.model_id == model.id, Report.kind == "drift")
        .order_by(Report.created_at.desc()).limit(limit)
    ))
