"""The algorithms ModelDoctor can train, per task, with defaults and search spaces.

Optional libraries (LightGBM, PyTorch) are offered only when they can be imported,
so the core works with scikit-learn and XGBoost alone.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from app.ml.torch_models import TORCH_AVAILABLE, TorchMLPClassifier, TorchMLPRegressor

SEED = 42
TASKS = ("classification", "regression")

try:  # pragma: no cover - availability depends on the environment
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:  # noqa: BLE001
    LIGHTGBM_AVAILABLE = False


def task_family(task_type: str | None) -> str:
    """'binary_classification' / 'multiclass_classification' -> 'classification'."""
    return "regression" if task_type == "regression" else "classification"


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    factories: dict[str, Callable[..., object]]  # task -> estimator factory
    defaults: dict[str, dict]
    search_space: dict[str, dict]
    supports_class_weight: bool = True  # False -> balanced sample weights are used instead
    default_for: tuple[str, ...] = TASKS  # tasks where it is trained by default
    available: bool = True
    note: str = ""
    n_jobs_param: str | None = "n_jobs"
    extra: dict = field(default_factory=dict)

    def tasks(self) -> list[str]:
        return [t for t in TASKS if t in self.factories]


def _svc(**p):
    # SVC(probability=True) is deprecated in scikit-learn 1.9; Platt scaling via
    # CalibratedClassifierCV(ensemble=False) is the documented replacement.
    p = {k.removeprefix("estimator__"): v for k, v in p.items()}
    return CalibratedClassifierCV(SVC(**p), method="sigmoid", cv=3, ensemble=False)


def _svr(**p):
    # SVR is sensitive to the scale of the target (house prices in the 100,000s break the
    # default epsilon/C), so the target is standardised inside the estimator.
    p = {k.removeprefix("regressor__"): v for k, v in p.items()}
    p.pop("random_state", None)
    p.pop("n_jobs", None)
    return TransformedTargetRegressor(regressor=SVR(**p), transformer=StandardScaler())


XGB_SPACE = {
    "n_estimators": [150, 300, 500], "max_depth": [3, 4, 5, 6, 8], "learning_rate": loguniform(0.02, 0.3),
    "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
    "reg_lambda": loguniform(0.1, 10),
}
XGB_DEFAULTS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9,
                "colsample_bytree": 0.9, "tree_method": "hist", "random_state": SEED, "verbosity": 0}
RF_SPACE = {"n_estimators": [100, 200, 400], "max_depth": [None, 6, 10, 16],
            "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", 0.5]}
LGBM_SPACE = {"n_estimators": [150, 300, 500], "num_leaves": [15, 31, 63], "learning_rate": loguniform(0.02, 0.3),
              "min_child_samples": [5, 20, 50], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
              "reg_lambda": loguniform(0.01, 10)}
LGBM_DEFAULTS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9,
                 "subsample_freq": 1, "colsample_bytree": 0.9, "random_state": SEED, "verbose": -1}
MLP_SPACE = {"hidden": [(64,), (64, 32), (128, 64), (128, 64, 32)], "lr": loguniform(3e-4, 3e-2),
             "weight_decay": loguniform(1e-6, 1e-2), "epochs": [40, 60, 100], "dropout": [0.0, 0.1, 0.3]}
MLP_DEFAULTS = {"hidden": (64, 32), "epochs": 60, "lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                "dropout": 0.1, "random_state": SEED}

ALGORITHMS: dict[str, Algorithm] = {
    "logistic_regression": Algorithm(
        "logistic_regression", "Logistic Regression", {"classification": LogisticRegression},
        {"classification": {"max_iter": 2000, "C": 1.0}},
        {"classification": {"C": loguniform(1e-2, 1e2)}},
        n_jobs_param=None,
    ),
    "ridge": Algorithm(
        "ridge", "Ridge Regression", {"regression": Ridge},
        {"regression": {"alpha": 1.0}},
        {"regression": {"alpha": loguniform(1e-3, 1e3)}},
        n_jobs_param=None,
    ),
    "random_forest": Algorithm(
        "random_forest", "Random Forest",
        {"classification": RandomForestClassifier, "regression": RandomForestRegressor},
        {t: {"n_estimators": 200, "min_samples_leaf": 2, "random_state": SEED} for t in TASKS},
        {t: RF_SPACE for t in TASKS},
    ),
    "xgboost": Algorithm(
        "xgboost", "XGBoost", {"classification": XGBClassifier, "regression": XGBRegressor},
        {t: XGB_DEFAULTS for t in TASKS}, {t: XGB_SPACE for t in TASKS},
        supports_class_weight=False,
    ),
    "lightgbm": Algorithm(
        "lightgbm", "LightGBM",
        {"classification": LGBMClassifier, "regression": LGBMRegressor} if LIGHTGBM_AVAILABLE else {},
        {t: LGBM_DEFAULTS for t in TASKS}, {t: LGBM_SPACE for t in TASKS},
        available=LIGHTGBM_AVAILABLE, note="" if LIGHTGBM_AVAILABLE else "install lightgbm to enable",
    ),
    "svm": Algorithm(
        "svm", "SVM (RBF)", {"classification": _svc, "regression": _svr},
        {"classification": {"C": 1.0, "kernel": "rbf", "random_state": SEED},
         "regression": {"C": 1.0, "kernel": "rbf"}},
        {"classification": {"estimator__C": loguniform(0.1, 100), "estimator__gamma": ["scale", 0.01, 0.1]},
         "regression": {"regressor__C": loguniform(0.1, 100), "regressor__gamma": ["scale", 0.01, 0.1],
                        "regressor__epsilon": [0.01, 0.1, 0.5]}},
        n_jobs_param=None,
    ),
    "pytorch_mlp": Algorithm(
        "pytorch_mlp", "Neural network (PyTorch MLP)",
        {"classification": TorchMLPClassifier, "regression": TorchMLPRegressor} if TORCH_AVAILABLE else {},
        {t: MLP_DEFAULTS for t in TASKS}, {t: MLP_SPACE for t in TASKS},
        supports_class_weight=False, default_for=(), available=TORCH_AVAILABLE,
        note="opt-in (slower)" if TORCH_AVAILABLE else "install torch to enable",
    ),
}


def available_algorithms(task: str) -> list[Algorithm]:
    return [a for a in ALGORITHMS.values() if a.available and task in a.factories]


def default_algorithms(task: str) -> list[str]:
    return [a.key for a in available_algorithms(task) if task in a.default_for]


def make_estimator(key: str, params: dict | None, class_weight: str | None, n_jobs: int, task: str = "classification"):
    algo = ALGORITHMS[key]
    if task not in algo.factories:
        raise ValueError(f"{algo.label} is not available for {task}")
    p = {**algo.defaults[task], **(params or {})}
    if algo.n_jobs_param:
        p.setdefault(algo.n_jobs_param, n_jobs)
    if key == "pytorch_mlp":
        p["n_jobs"] = max(1, n_jobs) if n_jobs and n_jobs > 0 else 1
        if isinstance(p.get("hidden"), list):
            p["hidden"] = tuple(p["hidden"])  # JSON round-trips tuples as lists
    if class_weight and algo.supports_class_weight and task == "classification":
        p["class_weight"] = class_weight
    return algo.factories[task](**p)
