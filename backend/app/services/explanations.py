"""Phase 4 use cases: cached model explanations and per-prediction explanations."""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import DomainError, NotFoundError
from app.db.models import MLModel, utcnow
from app.diagnostics.findings import sort_findings
from app.ml.error_analysis import analyse_errors, group_performance, prediction_findings
from app.ml.evaluation import predict_labels
from app.ml.explain import explain_row, global_importance, reference_values
from app.ml.features import FeatureSpec, prepare_frame
from app.ml.training import TrainingData, prepare_training_data
from app.services.datasets import get_dataset, load_dataframe
from app.services.target import normalize_labels
from app.services.training import load_model_bundle


def holdout(db: Session, settings: Settings, model: MLModel):
    """Rebuild the exact train/test split the model was evaluated on."""
    dataset = get_dataset(db, model.dataset_id)
    df = load_dataframe(dataset, settings)
    split = model.metrics["split"]
    data = prepare_training_data(
        dataset, df, settings, seed=split["seed"], test_size=split["test_size"], target=model.feature_spec["target"]
    )
    return dataset, df, load_model_bundle(settings, model), data


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def compute_explanations(db: Session, settings: Settings, model: MLModel, *, refresh: bool = False) -> dict:
    if model.explanations and not refresh:
        return model.explanations
    _, _, bundle, data = holdout(db, settings, model)
    spec = FeatureSpec.from_dict(bundle["spec"])
    pos, classes = bundle["positive_index"], bundle["classes"]
    X_test = data.X_test
    proba = bundle["pipeline"].predict_proba(X_test[spec.features])
    pred = predict_labels(proba, pos, bundle["config"].get("threshold"))
    row_ids = [int(i) for i in X_test.index]

    errors = analyse_errors(X_test, spec, data.y_test, pred, proba, classes, pos, row_ids)
    fairness = group_performance(X_test, spec, data.y_test, pred, classes, pos)
    samples = [
        {"row_id": row_ids[i], "true": classes[int(data.y_test[i])], "predicted": classes[int(pred[i])],
         "confidence": round(float(proba[i].max()), 4), "correct": bool(pred[i] == data.y_test[i])}
        for i in range(min(200, len(row_ids)))
    ]
    model.explanations = {
        "computed_at": utcnow().isoformat(),
        "global": global_importance(bundle, X_test, data.y_test),
        "error_analysis": errors,
        "fairness": fairness,
        "findings": sort_findings(prediction_findings(errors, fairness)),
        "samples": samples,
        "reference": {k: _py(v) for k, v in reference_values(data.X_train, spec).items()},
    }
    db.commit()
    db.refresh(model)
    return model.explanations


def explain_prediction(db: Session, settings: Settings, model: MLModel, *,
                       row_id: int | None = None, record: dict | None = None) -> dict:
    explanations = compute_explanations(db, settings, model)
    bundle = load_model_bundle(settings, model)
    spec = FeatureSpec.from_dict(bundle["spec"])
    actual = None
    if row_id is not None:
        dataset = get_dataset(db, model.dataset_id)
        df = load_dataframe(dataset, settings)
        if row_id not in df.index:
            raise NotFoundError(f"Row {row_id} does not exist in the dataset.")
        raw = df.loc[[row_id]]
        target = model.feature_spec["target"]
        if pd.notna(raw.iloc[0][target]):
            profile = next(c for c in dataset.columns if c["name"] == target)
            actual = normalize_labels(raw[target], profile, settings).iloc[0]
    else:
        raw = pd.DataFrame.from_records([record])
        missing = [c for c in spec.features if c not in raw.columns]
        if missing:
            raise DomainError(f"Record is missing feature columns: {', '.join(missing)}.", code="invalid_request")
    prepared = prepare_frame(raw, spec)
    result = explain_row(bundle, prepared, explanations["reference"], raw.iloc[0].to_dict())
    in_test = row_id is not None and any(s["row_id"] == row_id for s in explanations["samples"])
    return {"model_id": model.id, "row_id": row_id, "actual": actual, "in_test_sample": in_test, **result}
