"""The combined diagnosis: data + model + prediction + drift findings and the health score."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.db.models import Dataset, Job, MLModel
from app.diagnostics.findings import sort_findings, summarize_counts
from app.diagnostics.health import compute_health
from app.diagnostics.narrative import model_narrative, recommendations
from app.services.reports import get_data_report, latest_report


def select_model(db: Session, dataset: Dataset, model_id: str | None = None) -> MLModel | None:
    if model_id:
        model = db.get(MLModel, model_id)
        if model is None or model.dataset_id != dataset.id:
            raise NotFoundError(f"Model '{model_id}' not found for this dataset.")
        return model
    models = list(db.scalars(select(MLModel).where(MLModel.dataset_id == dataset.id)))
    if not models:
        return None
    production = [m for m in models if m.stage == "production"]
    pool = production or [m for m in models if m.stage not in {"rejected", "archived"}] or models
    return max(pool, key=lambda m: m.primary_score)


def latest_experiment(db: Session, model: MLModel) -> dict | None:
    jobs = db.scalars(
        select(Job)
        .where(Job.dataset_id == model.dataset_id, Job.kind == "experiment", Job.status == "completed")
        .order_by(Job.created_at.desc())
    )
    for job in jobs:
        if job.params.get("base_model_id") == model.id or (job.result or {}).get("best_model_id") == model.id:
            return {"job_id": job.id, **(job.result or {})}
    return None


def resolve_for_model(data_findings: list[dict], model: MLModel) -> list[dict]:
    """Data findings about columns this model doesn't use are kept but marked resolved."""
    used = set(model.feature_spec.get("numeric", []) + model.feature_spec.get("categorical", [])
               + model.feature_spec.get("datetime", []))
    out = []
    for f in data_findings:
        if f["check"] in {"feature.leakage", "feature.low_relevance"} and f["columns"] and not (set(f["columns"]) & used):
            f = {**f, "severity": "info", "interventions": [],
                 "title": f["title"] + " (excluded from this model)", "resolved": True}
        out.append(f)
    return out


def build_diagnosis(db: Session, settings: Settings, dataset: Dataset, model_id: str | None = None) -> dict:
    report = get_data_report(db, settings, dataset)
    model = select_model(db, dataset, model_id)
    findings = list(report.payload["findings"])
    fairness_gap = drift_score = None
    experiment = None
    if model:
        findings = resolve_for_model(findings, model) + model.findings
        if model.explanations:
            findings += model.explanations.get("findings", [])
            fairness_gap = (model.explanations.get("fairness") or {}).get("max_gap")
        drift = latest_report(db, dataset.id, "drift", model_id=model.id)
        if drift:
            findings += drift.payload.get("findings", [])
            drift_score = drift.payload.get("score")
        experiment = latest_experiment(db, model)
    findings = sort_findings(findings)

    health = compute_health(
        report.payload["score"]["value"], model.metrics if model else None, fairness_gap, drift_score
    )
    return {
        "dataset_id": dataset.id,
        "model": None if model is None else {
            "id": model.id, "name": model.name, "version": model.version, "stage": model.stage,
            "algorithm": model.algorithm,
        },
        "health": health,
        "severity_counts": summarize_counts(findings),
        "findings": findings,
        "recommendations": recommendations(findings),
        "narrative": model_narrative(model.name, model.metrics, findings, experiment) if model else [
            "No model trained yet. Data findings are shown; train baselines to diagnose model health."
        ],
        "latest_experiment": experiment and {
            k: experiment.get(k) for k in ("job_id", "best_model_id", "baseline_primary", "primary_metric", "best")
        },
    }
