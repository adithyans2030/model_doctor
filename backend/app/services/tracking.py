"""Optional MLflow experiment tracking (Phase 7).

ModelDoctor keeps its own registry in the database, which is the source of
truth. When ``MD_MLFLOW_TRACKING_URI`` is set and ``mlflow`` is installed, every
trained model and every optimisation trial is also logged to MLflow, so runs
can be compared in the MLflow UI. Tracking failures are logged and never break
training.
"""

import logging

from app.core.config import Settings

log = logging.getLogger("modeldoctor.tracking")


def _mlflow(settings: Settings):
    if not settings.mlflow_tracking_uri:
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("MD_MLFLOW_TRACKING_URI is set but mlflow is not installed; tracking disabled")
        return None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    return mlflow


def _flat_metrics(test: dict) -> dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "pr_auc", "log_loss", "ece"]
    return {k: float(test[k]) for k in keys if test.get(k) is not None}


def _params(config: dict) -> dict[str, str]:
    out = {k: str(v) for k, v in config.items() if k != "params"}
    out.update({f"hp.{k}": str(v) for k, v in (config.get("params") or {}).items()})
    return out


def log_model(settings: Settings, model) -> str | None:
    mlflow = _mlflow(settings)
    if mlflow is None:
        return None
    try:
        with mlflow.start_run(run_name=model.name) as run:
            mlflow.set_tags({"modeldoctor.model_id": model.id, "modeldoctor.dataset_id": model.dataset_id,
                             "modeldoctor.version": str(model.version), "modeldoctor.source": model.source})
            mlflow.log_params(_params(model.config))
            mlflow.log_metrics(_flat_metrics(model.metrics["test"]))
            mlflow.log_dict({"findings": model.findings}, "diagnosis.json")
            return run.info.run_id
    except Exception:  # noqa: BLE001
        log.exception("MLflow logging failed for model %s", model.id)
        return None


def log_experiment(settings: Settings, job_id: str, base_model, trials: list[dict]) -> None:
    mlflow = _mlflow(settings)
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=f"optimise {base_model.name}") as parent:
            mlflow.set_tags({"modeldoctor.job_id": job_id, "modeldoctor.base_model_id": base_model.id})
            for t in trials:
                with mlflow.start_run(run_name=t["name"], nested=True):
                    mlflow.log_params(_params(t["config"]))
                    mlflow.log_metrics({**{k: float(v) for k, v in t["metrics"].items() if v is not None},
                                        "cv_f1_macro": float(t["cv_f1_macro"])})
                    mlflow.set_tag("kept", str(t["kept"]))
            mlflow.set_tag("modeldoctor.parent_run", parent.info.run_id)
    except Exception:  # noqa: BLE001
        log.exception("MLflow logging failed for experiment %s", job_id)
