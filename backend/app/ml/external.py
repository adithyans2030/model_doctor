"""Bring-your-own-model: safe loading of uploaded models and an adapter for them.

Security: pickle/joblib files execute code when loaded, so they are never accepted.
Uploads must be **skops** files. Before loading, skops lists every type the file would
instantiate; the file is rejected unless *all* of them come from an allowlist of
numerical/ML libraries (scikit-learn, XGBoost, LightGBM, imbalanced-learn, NumPy, SciPy
and builtins). Only then is it loaded, trusting exactly those types.
"""

from pathlib import Path

import numpy as np

from app.core.errors import InvalidFileError

ALLOWED_PREFIXES = (
    "sklearn.", "xgboost.", "lightgbm.", "imblearn.", "numpy.", "scipy.", "builtins.", "collections.",
    "functools.partial", "math.", "operator.",
)


def _allowed(type_name: str) -> bool:
    return type_name.startswith(ALLOWED_PREFIXES)


def load_uploaded_model(path: Path):
    try:
        from skops.io import get_untrusted_types, load
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise InvalidFileError("Model upload needs the 'skops' package on the server.") from exc
    try:
        untrusted = get_untrusted_types(file=path)
    except Exception as exc:  # noqa: BLE001 - not a valid skops archive
        raise InvalidFileError(f"Not a valid skops file: {type(exc).__name__}.") from exc
    rejected = sorted(t for t in untrusted if not _allowed(t))
    if rejected:
        raise InvalidFileError(
            "The model file contains types outside the allowed ML libraries, so it was not loaded: "
            + ", ".join(rejected[:8]) + ("…" if len(rejected) > 8 else "")
        )
    try:
        return load(path, trusted=untrusted)
    except Exception as exc:  # noqa: BLE001
        raise InvalidFileError(f"The model file could not be loaded: {exc}") from exc


class ExternalModel:
    """Adapts an uploaded estimator to ModelDoctor's conventions.

    * Inputs: the prepared feature frame (same columns as the dataset's usable features).
    * Classification: ``predict_proba`` columns are reordered to ModelDoctor's sorted
      class labels, whatever label type the model was trained with (e.g. 0/1 ints).
    """

    def __init__(self, estimator, task: str, classes: list[str] | None = None):
        self.estimator = estimator
        self.task = task
        self.classes = classes or []
        if task == "classification":
            if not hasattr(estimator, "predict_proba"):
                raise InvalidFileError("Uploaded classifiers must implement predict_proba.")
            model_labels = [self._norm(c) for c in getattr(estimator, "classes_", [])]
            missing = [c for c in self.classes if c not in model_labels]
            if not model_labels or missing:
                raise InvalidFileError(
                    f"The model's classes {model_labels} don't match the dataset's classes {self.classes}."
                )
            self.order = [model_labels.index(c) for c in self.classes]
        elif not hasattr(estimator, "predict"):
            raise InvalidFileError("Uploaded models must implement predict.")

    @staticmethod
    def _norm(label) -> str:
        if isinstance(label, (bool, np.bool_)):
            return str(label).lower()
        if isinstance(label, (float, np.floating)) and float(label).is_integer():
            return str(int(label))
        return str(label).strip()

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=float)[:, self.order]

    def predict(self, X):
        if self.task == "regression":
            return np.asarray(self.estimator.predict(X), dtype=float)
        return self.predict_proba(X).argmax(axis=1)
