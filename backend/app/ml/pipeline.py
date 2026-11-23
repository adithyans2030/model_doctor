"""Training configuration and the sklearn/imblearn pipeline built from it.

The same ``TrainConfig`` drives baseline training (Phase 3) and every
optimisation trial (Phase 5), so an experiment differs from its baseline only in
the fields it changes.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.ml.algorithms import ALGORITHMS, SEED, make_estimator
from app.ml.features import FeatureSpec


@dataclass
class TrainConfig:
    algorithm: str
    params: dict = field(default_factory=dict)
    class_weight: str | None = None  # "balanced"
    resampling: str | None = None  # "smote"
    drop_columns: list[str] = field(default_factory=list)
    dedupe: bool = False  # drop duplicate rows from the training split
    threshold: float | None = None  # binary decision threshold (None = argmax / 0.5)
    calibration: str | None = None  # "sigmoid" | "isotonic"
    target_transform: str | None = None  # regression: "log1p" trains on log(1 + y)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe(self) -> str:
        parts = [ALGORITHMS[self.algorithm].label if self.algorithm in ALGORITHMS else "Uploaded model"]
        if self.class_weight:
            parts.append("class weights")
        if self.resampling:
            parts.append(self.resampling.upper())
        if self.dedupe:
            parts.append("deduplicated")
        if self.drop_columns:
            cols = self.drop_columns
            parts.append(f"without {', '.join(cols)}" if len(cols) <= 2 else f"without {len(cols)} features")
        if self.params:
            parts.append("tuned")
        if self.calibration:
            parts.append(f"{self.calibration} calibration")
        if self.threshold is not None:
            parts.append(f"threshold {self.threshold:.2f}")
        if self.target_transform:
            parts.append(f"{self.target_transform} target")
        return " + ".join(parts)


def config_spec(spec: FeatureSpec, cfg: TrainConfig) -> FeatureSpec:
    drop = set(cfg.drop_columns)
    return FeatureSpec(
        target=spec.target,
        numeric=[c for c in spec.numeric if c not in drop],
        categorical=[c for c in spec.categorical if c not in drop],
        datetime=[c for c in spec.datetime if c not in drop],
        excluded={**spec.excluded, **{c: "dropped by configuration" for c in drop}},
    )


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []
    if spec.numeric_like:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            spec.numeric_like,
        ))
    if spec.categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=5,
                    max_categories=30, sparse_output=False,
                )),
            ]),
            spec.categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(cfg: TrainConfig, spec: FeatureSpec, y_train: np.ndarray, n_jobs: int = -1,
                   task: str = "classification"):
    estimator = make_estimator(cfg.algorithm, cfg.params, cfg.class_weight, n_jobs, task)
    if task == "regression":
        if cfg.target_transform == "log1p":
            # Only the model sees log(1 + y); predictions are mapped back with expm1.
            estimator = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
        return Pipeline([("prep", make_preprocessor(spec)), ("model", estimator)])
    if cfg.calibration:
        estimator = CalibratedClassifierCV(estimator, method=cfg.calibration, cv=3)
    steps = [("prep", make_preprocessor(spec))]
    if cfg.resampling == "smote":
        minority = int(np.bincount(y_train).min())
        k = max(1, min(5, minority - 1))
        steps.append(("resample", SMOTE(k_neighbors=k, random_state=SEED)))
        return ImbPipeline(steps + [("model", estimator)])
    return Pipeline(steps + [("model", estimator)])


def fit_params(cfg: TrainConfig, y_train: np.ndarray, task: str = "classification") -> dict:
    """Algorithms without a class_weight option get balanced sample weights instead."""
    if task == "classification" and cfg.class_weight and not ALGORITHMS[cfg.algorithm].supports_class_weight:
        return {"model__sample_weight": compute_sample_weight(cfg.class_weight, y_train)}
    return {}


def underlying_model(pipe):
    model = pipe.named_steps["model"]
    while isinstance(model, TransformedTargetRegressor):  # log target and/or scaled target (SVR)
        model = model.regressor_
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def search_prefix(cfg: TrainConfig) -> str:
    """Where the estimator's hyperparameters live inside the pipeline."""
    if cfg.target_transform:
        return "model__regressor__"
    if cfg.calibration:
        return "model__estimator__"
    return "model__"
