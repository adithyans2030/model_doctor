"""Reproducible train/test splitting, cross-validation, fitting and evaluation.

Task-aware: classification (binary / multiclass) and regression share one code
path. Primary metric: macro-F1 for classification, R² for regression.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.core.errors import InvalidTargetError
from app.ml.algorithms import task_family
from app.ml.evaluation import (
    classification_metrics, predict_labels, quick_scores, regression_metrics, regression_quick,
)
from app.ml.features import FeatureSpec, build_feature_spec, labelled_rows, original_feature, prepare_frame
from app.ml.pipeline import TrainConfig, build_pipeline, config_spec, fit_params, underlying_model

PRIMARY = {"classification": "f1_macro", "regression": "r2"}
CV_METRICS = {"classification": ["accuracy", "balanced_accuracy", "f1_macro"], "regression": ["r2", "rmse", "mae"]}


@dataclass
class TrainingData:
    spec: FeatureSpec
    classes: list[str]  # empty for regression
    positive_index: int | None  # binary only: index of the minority ("positive") class
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    n_labelled: int
    sampled_from: int | None  # original labelled row count if we sampled down
    seed: int
    test_size: float
    task: str = "classification"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def positive_label(self) -> str | None:
        return None if self.positive_index is None else self.classes[self.positive_index]

    @property
    def primary(self) -> str:
        return PRIMARY[self.task]

    def split_info(self) -> dict:
        return {
            "seed": self.seed, "test_size": self.test_size,
            "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
            "n_labelled": self.n_labelled, "sampled_from": self.sampled_from,
        }


def _quantile_bins(y: pd.Series, n: int = 10) -> pd.Series | None:
    """Stratification labels for a continuous target, so train/test cover the same range."""
    try:
        bins = pd.qcut(y, q=n, labels=False, duplicates="drop")
    except ValueError:
        return None
    return bins if bins.value_counts().min() >= 2 else None


def prepare_training_data(
    dataset, df: pd.DataFrame, settings, *, seed: int = 42, test_size: float = 0.2, target: str | None = None
) -> TrainingData:
    """Deterministic for a given (data, seed, test_size, target): later phases rebuild the
    exact same test set from the values stored in a model's ``metrics.split``."""
    target = target or dataset.target_column
    if not target:
        raise InvalidTargetError("Select a target column before training.")
    profile = next(c for c in dataset.columns if c["name"] == target)
    frame, y_raw = labelled_rows(df, target, profile, settings)
    task = "regression" if pd.api.types.is_float_dtype(y_raw) else "classification"

    strat = _quantile_bins(y_raw) if task == "regression" else y_raw
    sampled_from = None
    if len(frame) > settings.max_train_rows:
        sampled_from = len(frame)
        keep, _ = train_test_split(frame.index, train_size=settings.max_train_rows, stratify=strat,
                                   random_state=seed)
        frame, y_raw = frame.loc[keep], y_raw.loc[keep]
        strat = _quantile_bins(y_raw) if task == "regression" else y_raw

    spec = build_feature_spec(dataset.columns, target)
    if not spec.features:
        raise InvalidTargetError("No usable feature columns remain after excluding IDs, text and constants.")
    X = prepare_frame(frame, spec)

    if task == "regression":
        y = y_raw.to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=None if strat is None else strat.to_numpy(), random_state=seed
        )
        return TrainingData(spec, [], None, X_train, X_test, y_train, y_test, len(frame), sampled_from,
                            seed, test_size, task)

    counts = y_raw.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise InvalidTargetError(
            f"Classes {', '.join(map(repr, too_small.index))} have fewer than 2 rows, so they can't be split "
            "into train and test sets."
        )
    classes = sorted(counts.index)
    code = {c: i for i, c in enumerate(classes)}
    positive_index = code[counts.idxmin()] if len(classes) == 2 else None
    y = y_raw.map(code).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return TrainingData(spec, classes, positive_index, X_train, X_test, y_train, y_test,
                        len(frame), sampled_from, seed, test_size, task)


def _training_rows(cfg: TrainConfig, X: pd.DataFrame, y: np.ndarray):
    if not cfg.dedupe:
        return X, y
    keep = ~X.assign(__y=y).duplicated().to_numpy()
    return X.loc[keep], y[keep]


def fit_config(cfg: TrainConfig, X: pd.DataFrame, y: np.ndarray, spec: FeatureSpec, n_jobs: int = -1,
               task: str = "classification"):
    spec_cfg = config_spec(spec, cfg)
    X, y = _training_rows(cfg, X[spec_cfg.features], y)
    pipe = build_pipeline(cfg, spec_cfg, y, n_jobs, task)
    pipe.fit(X, y, **fit_params(cfg, y, task))
    return pipe, spec_cfg


def predict_outputs(pipe, X: pd.DataFrame, data_or_task, positive_index=None, threshold=None):
    """(predictions, probabilities-or-None) for either task."""
    task = data_or_task if isinstance(data_or_task, str) else data_or_task.task
    if task == "regression":
        return np.asarray(pipe.predict(X), dtype=float), None
    proba = pipe.predict_proba(X)
    return predict_labels(proba, positive_index, threshold), proba


def _scores(task: str, y, pred, n_classes: int) -> dict:
    return regression_quick(y, pred) if task == "regression" else quick_scores(y, pred, n_classes)


def cross_validate_config(cfg: TrainConfig, data: TrainingData, folds: int, n_jobs: int = -1) -> dict | None:
    """Manual K-fold (stratified for classification) so we also get out-of-fold predictions."""
    task = data.task
    if task == "classification":
        folds = min(folds, int(np.bincount(data.y_train).min()))
    folds = min(folds, len(data.y_train) // 2)
    if folds < 2:
        return None
    if task == "classification":
        splitter = StratifiedKFold(folds, shuffle=True, random_state=data.seed)
        oof = np.zeros((len(data.y_train), data.n_classes))
    else:
        splitter = KFold(folds, shuffle=True, random_state=data.seed)
        oof = np.zeros(len(data.y_train))
    rows = []
    for tr, va in splitter.split(data.X_train, data.y_train):
        Xtr, ytr = data.X_train.iloc[tr], data.y_train[tr]
        Xva, yva = data.X_train.iloc[va], data.y_train[va]
        pipe, spec_cfg = fit_config(cfg, Xtr, ytr, data.spec, n_jobs, task)
        pred_va, proba_va = predict_outputs(pipe, Xva[spec_cfg.features], data, data.positive_index, cfg.threshold)
        oof[va] = proba_va if proba_va is not None else pred_va
        pred_tr, _ = predict_outputs(pipe, Xtr[spec_cfg.features], data, data.positive_index, cfg.threshold)
        rows.append({"val": _scores(task, yva, pred_va, data.n_classes),
                     "train": _scores(task, ytr, pred_tr, data.n_classes)})

    def agg(part, metric):
        vals = np.array([r[part][metric] for r in rows if r[part][metric] is not None], dtype=float)
        return {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)} if len(vals) else None

    metrics = CV_METRICS[task]
    return {
        "folds": folds,
        "val": {m: agg("val", m) for m in metrics},
        "train": {m: agg("train", m) for m in metrics},
        "oof_proba": oof,
    }


def dummy_scores(data: TrainingData) -> dict:
    if data.task == "regression":
        dummy = DummyRegressor(strategy="mean").fit(data.X_train, data.y_train)
        return regression_quick(data.y_test, dummy.predict(data.X_test))
    dummy = DummyClassifier(strategy="prior").fit(data.X_train, data.y_train)
    proba = dummy.predict_proba(data.X_test)
    pred = predict_labels(proba, data.positive_index, None)
    return quick_scores(data.y_test, pred, data.n_classes)


def native_importances(pipe, spec_cfg: FeatureSpec) -> list[dict] | None:
    if not hasattr(pipe, "named_steps") or "prep" not in pipe.named_steps:
        return None  # e.g. uploaded models with their own preprocessing
    model = underlying_model(pipe)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.atleast_2d(np.asarray(model.coef_, dtype=float))).mean(axis=0)
    else:
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    if len(names) != len(values):
        return None
    agg: dict[str, float] = {}
    for name, v in zip(names, values):
        orig = original_feature(name, spec_cfg.features)
        agg[orig] = agg.get(orig, 0.0) + float(v)
    total = sum(agg.values()) or 1.0
    return sorted(
        ({"feature": k, "importance": round(v / total, 4)} for k, v in agg.items()),
        key=lambda d: -d["importance"],
    )


@dataclass
class TrainResult:
    config: TrainConfig
    pipeline: object
    spec: FeatureSpec
    metrics: dict
    importances: list[dict] | None
    oof_proba: np.ndarray | None


def evaluate_pipeline(pipe, spec_cfg: FeatureSpec, cfg: TrainConfig, data: TrainingData) -> tuple[dict, dict]:
    """(test metrics, train quick scores) for a fitted pipeline."""
    pred_test, proba_test = predict_outputs(pipe, data.X_test[spec_cfg.features], data, data.positive_index, cfg.threshold)
    pred_train, _ = predict_outputs(pipe, data.X_train[spec_cfg.features], data, data.positive_index, cfg.threshold)
    if data.task == "regression":
        return regression_metrics(data.y_test, pred_test), regression_quick(data.y_train, pred_train)
    test = classification_metrics(data.y_test, proba_test, data.classes, data.positive_index, cfg.threshold)
    return test, quick_scores(data.y_train, pred_train, data.n_classes)


def train_and_evaluate(
    cfg: TrainConfig, data: TrainingData, *, cv_folds: int = 5, n_jobs: int = -1, dummy: dict | None = None
) -> TrainResult:
    started = time.perf_counter()
    cv = cross_validate_config(cfg, data, cv_folds, n_jobs) if cv_folds >= 2 else None
    pipe, spec_cfg = fit_config(cfg, data.X_train, data.y_train, data.spec, n_jobs, data.task)
    test, train = evaluate_pipeline(pipe, spec_cfg, cfg, data)
    oof = cv.pop("oof_proba") if cv is not None else None
    metrics = {
        "task": data.task,
        "primary_metric": data.primary,
        "test": test,
        "train": train,
        "cv": cv,
        "dummy": dummy,
        "split": data.split_info(),
        "positive_label": data.positive_label,
        "classes": data.classes,
        "n_features": len(spec_cfg.features),
        "fit_seconds": round(time.perf_counter() - started, 2),
    }
    return TrainResult(cfg, pipe, spec_cfg, metrics, native_importances(pipe, spec_cfg), oof)


def model_task(metrics: dict) -> str:
    return metrics.get("task") or "classification"


def primary_value(metrics: dict) -> float:
    return metrics["test"][metrics.get("primary_metric", "f1_macro")]


__all__ = [
    "TrainingData", "TrainResult", "prepare_training_data", "fit_config", "cross_validate_config",
    "train_and_evaluate", "dummy_scores", "native_importances", "predict_outputs", "evaluate_pipeline",
    "model_task", "primary_value", "task_family", "clone", "PRIMARY",
]
