"""The algorithms ModelDoctor can train, with defaults and search spaces."""

from collections.abc import Callable
from dataclasses import dataclass

from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    factory: Callable[..., object]
    defaults: dict
    search_space: dict
    supports_class_weight: bool = True  # False -> balanced sample weights are used instead


def _lr(**p):
    return LogisticRegression(**p)


def _rf(**p):
    return RandomForestClassifier(**p)


def _xgb(**p):
    return XGBClassifier(**p)


def _svm(**p):
    # SVC(probability=True) is deprecated in scikit-learn 1.9; Platt scaling via
    # CalibratedClassifierCV(ensemble=False) is the documented replacement.
    p = {k.removeprefix("estimator__"): v for k, v in p.items()}
    return CalibratedClassifierCV(SVC(**p), method="sigmoid", cv=3, ensemble=False)


ALGORITHMS: dict[str, Algorithm] = {
    "logistic_regression": Algorithm(
        "logistic_regression", "Logistic Regression", _lr,
        {"max_iter": 2000, "C": 1.0},
        {"C": loguniform(1e-2, 1e2)},
    ),
    "random_forest": Algorithm(
        "random_forest", "Random Forest", _rf,
        {"n_estimators": 200, "min_samples_leaf": 2, "random_state": SEED},
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 6, 10, 16],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.5],
        },
    ),
    "xgboost": Algorithm(
        "xgboost", "XGBoost", _xgb,
        {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9,
            "colsample_bytree": 0.9, "tree_method": "hist", "random_state": SEED, "verbosity": 0,
        },
        {
            "n_estimators": [150, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": loguniform(0.02, 0.3),
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": loguniform(0.1, 10),
        },
        supports_class_weight=False,
    ),
    "svm": Algorithm(
        "svm", "SVM (RBF)", _svm,
        {"C": 1.0, "kernel": "rbf", "random_state": SEED},
        {"estimator__C": loguniform(0.1, 100), "estimator__gamma": ["scale", 0.01, 0.1]},
    ),
}

DEFAULT_ALGORITHMS = ["logistic_regression", "random_forest", "xgboost", "svm"]


def make_estimator(key: str, params: dict | None, class_weight: str | None, n_jobs: int):
    algo = ALGORITHMS[key]
    p = {**algo.defaults, **(params or {})}
    if key in {"random_forest", "xgboost"}:
        p.setdefault("n_jobs", n_jobs)
    if class_weight and algo.supports_class_weight:
        p["class_weight"] = class_weight
    return algo.factory(**p)
