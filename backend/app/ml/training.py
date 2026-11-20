"""Reproducible train/test splitting, cross-validation, fitting and evaluation."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.core.errors import InvalidTargetError
from app.ml.evaluation import classification_metrics, predict_labels, quick_scores
from app.ml.features import FeatureSpec, build_feature_spec, labelled_rows, prepare_frame
from app.ml.pipeline import TrainConfig, build_pipeline, config_spec, fit_params, underlying_model


@dataclass
class TrainingData:
    spec: FeatureSpec
    classes: list[str]
    positive_index: int | None  # binary only: index of the minority ("positive") class
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    n_labelled: int
    sampled_from: int | None  # original labelled row count if we sampled down
    seed: int
    test_size: float

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def positive_label(self) -> str | None:
        return None if self.positive_index is None else self.classes[self.positive_index]

    def split_info(self) -> dict:
        return {
            "seed": self.seed, "test_size": self.test_size,
            "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
            "n_labelled": self.n_labelled, "sampled_from": self.sampled_from,
        }


def prepare_training_data(
    dataset, df: pd.DataFrame, settings, *, seed: int = 42, test_size: float = 0.2, target: str | None = None
) -> TrainingData:
    """Deterministic for a given (data, seed, test_size, target): later phases rebuild the
    exact same test set from the values stored in a model's ``metrics.split``."""
    target = target or dataset.target_column
    if not target:
        raise InvalidTargetError("Select a target column before training.")
    profile = next(c for c in dataset.columns if c["name"] == target)
    frame, labels = labelled_rows(df, target, profile, settings)

    sampled_from = None
    if len(frame) > settings.max_train_rows:
        sampled_from = len(frame)
        keep, _ = train_test_split(
            frame.index, train_size=settings.max_train_rows, stratify=labels, random_state=seed
        )
        frame, labels = frame.loc[keep], labels.loc[keep]

    counts = labels.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise InvalidTargetError(
            f"Classes {', '.join(map(repr, too_small.index))} have fewer than 2 rows, so they can't be split "
            "into train and test sets."
        )

    classes = sorted(counts.index)
    code = {c: i for i, c in enumerate(classes)}
    positive_index = code[counts.idxmin()] if len(classes) == 2 else None

    spec = build_feature_spec(dataset.columns, target)
    if not spec.features:
        raise InvalidTargetError("No usable feature columns remain after excluding IDs, text and constants.")
    X = prepare_frame(frame, spec)
    y = labels.map(code).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return TrainingData(spec, classes, positive_index, X_train, X_test, y_train, y_test,
                        len(frame), sampled_from, seed, test_size)


def _training_rows(cfg: TrainConfig, X: pd.DataFrame, y: np.ndarray):
    if not cfg.dedupe:
        return X, y
    keep = ~X.assign(__y=y).duplicated().to_numpy()
    return X.loc[keep], y[keep]


def fit_config(cfg: TrainConfig, X: pd.DataFrame, y: np.ndarray, spec: FeatureSpec, n_jobs: int = -1):
    spec_cfg = config_spec(spec, cfg)
    X, y = _training_rows(cfg, X[spec_cfg.features], y)
    pipe = build_pipeline(cfg, spec_cfg, y, n_jobs)
    pipe.fit(X, y, **fit_params(cfg, y))
    return pipe, spec_cfg


def cross_validate_config(cfg: TrainConfig, data: TrainingData, folds: int, n_jobs: int = -1) -> dict | None:
    """Manual stratified K-fold so we also get out-of-fold probabilities."""
    folds = min(folds, int(np.bincount(data.y_train).min()))
    if folds < 2:
        return None
    skf = StratifiedKFold(folds, shuffle=True, random_state=data.seed)
    oof = np.zeros((len(data.y_train), data.n_classes))
    rows = []
    for tr, va in skf.split(data.X_train, data.y_train):
        Xtr, ytr = data.X_train.iloc[tr], data.y_train[tr]
        Xva, yva = data.X_train.iloc[va], data.y_train[va]
        pipe, spec_cfg = fit_config(cfg, Xtr, ytr, data.spec, n_jobs)
        proba_va = pipe.predict_proba(Xva[spec_cfg.features])
        oof[va] = proba_va
        pred_va = predict_labels(proba_va, data.positive_index, cfg.threshold)
        pred_tr = predict_labels(pipe.predict_proba(Xtr[spec_cfg.features]), data.positive_index, cfg.threshold)
        rows.append({
            "val": quick_scores(yva, pred_va, data.n_classes),
            "train": quick_scores(ytr, pred_tr, data.n_classes),
        })

    def agg(part, metric):
        vals = np.array([r[part][metric] for r in rows], dtype=float)
        return {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}

    metrics = ["accuracy", "balanced_accuracy", "f1_macro"]
    return {
        "folds": folds,
        "val": {m: agg("val", m) for m in metrics},
        "train": {m: agg("train", m) for m in metrics},
        "oof_proba": oof,
    }


def dummy_scores(data: TrainingData) -> dict:
    dummy = DummyClassifier(strategy="prior").fit(data.X_train, data.y_train)
    proba = dummy.predict_proba(data.X_test)
    pred = predict_labels(proba, data.positive_index, None)
    return quick_scores(data.y_test, pred, data.n_classes)


def native_importances(pipe, spec_cfg: FeatureSpec) -> list[dict] | None:
    from app.ml.features import original_feature

    model = underlying_model(pipe)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
    else:
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
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


def train_and_evaluate(
    cfg: TrainConfig, data: TrainingData, *, cv_folds: int = 5, n_jobs: int = -1, dummy: dict | None = None
) -> TrainResult:
    started = time.perf_counter()
    cv = cross_validate_config(cfg, data, cv_folds, n_jobs) if cv_folds >= 2 else None
    pipe, spec_cfg = fit_config(cfg, data.X_train, data.y_train, data.spec, n_jobs)
    proba_test = pipe.predict_proba(data.X_test[spec_cfg.features])
    test = classification_metrics(data.y_test, proba_test, data.classes, data.positive_index, cfg.threshold)
    pred_train = predict_labels(pipe.predict_proba(data.X_train[spec_cfg.features]), data.positive_index, cfg.threshold)

    oof = None
    if cv is not None:
        oof = cv.pop("oof_proba")
    metrics = {
        "primary_metric": "f1_macro",
        "test": test,
        "train": quick_scores(data.y_train, pred_train, data.n_classes),
        "cv": cv,
        "dummy": dummy,
        "split": data.split_info(),
        "positive_label": data.positive_label,
        "classes": data.classes,
        "n_features": len(spec_cfg.features),
        "fit_seconds": round(time.perf_counter() - started, 2),
    }
    return TrainResult(cfg, pipe, spec_cfg, metrics, native_importances(pipe, spec_cfg), oof)


def predict_proba_frame(pipe, spec: FeatureSpec, df: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(prepare_frame(df, spec))


__all__ = [
    "TrainingData", "TrainResult", "prepare_training_data", "fit_config", "cross_validate_config",
    "train_and_evaluate", "dummy_scores", "native_importances", "predict_proba_frame", "clone",
]
