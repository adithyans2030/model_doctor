"""Phase 5 use cases: run automated corrective experiments for a model."""

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Job
from app.ml.optimize import run_optimisation, trial_summary
from app.ml.pipeline import TrainConfig
from app.ml.training import dummy_scores
from app.services.datasets import get_dataset
from app.services.diagnosis import resolve_for_model
from app.services.explanations import holdout
from app.services.jobs import set_progress
from app.services.reports import get_data_report
from app.services.training import get_model, register_model
from app.services import tracking


def experiment_worker(db: Session, settings: Settings, job: Job) -> dict:
    p = job.params
    base_model = get_model(db, p["base_model_id"])
    dataset = get_dataset(db, base_model.dataset_id)
    set_progress(db, job, "Rebuilding the evaluation split")
    _, _, _, data = holdout(db, settings, base_model)
    report = get_data_report(db, settings, dataset).payload

    findings = resolve_for_model(report["findings"], base_model) + base_model.findings
    if base_model.explanations:
        findings += base_model.explanations.get("findings", [])

    out = run_optimisation(
        TrainConfig.from_dict(base_model.config), data, findings, report,
        budget=p.get("budget", "standard"), allowed=p.get("interventions"),
        hpo_iterations=settings.hpo_iterations, n_jobs=settings.n_jobs, dummy=dummy_scores(data),
        progress=lambda msg: set_progress(db, job, msg),
    )
    baseline, best = out["baseline"], out["best"]
    trials = [trial_summary(t, baseline) for t in out["trials"]]
    tracking.log_experiment(settings, job.id, base_model, trials)

    best_summary = trial_summary(best, baseline)
    best_model = None
    if best.config.to_dict() != base_model.config:
        set_progress(db, job, "Registering the best configuration as a candidate model")
        best_model = register_model(
            db, settings, dataset, best.result, data, report["findings"], job_id=job.id, source="experiment",
            name=None,
        )
        best_model.name = f"{best_model.name} (optimised)"
        db.commit()
        best_summary.update(model_id=best_model.id, version=best_model.version)

    base_f1 = base_model.metrics["test"]["f1_macro"]
    improved = best_model is not None and best.result.metrics["test"]["f1_macro"] > base_f1
    if best_model is None:
        recommendation = "No intervention beat the current model on cross-validation; keep it."
    elif improved:
        recommendation = (f"Promote model v{best_model.version} ({best.config.describe()}) for further validation. "
                          "It requires human approval before it can be used in production.")
    else:
        recommendation = (f"Model v{best_model.version} is the best leak-free configuration. Its test score is lower "
                          "than the original only because the original relied on a leaking feature.")
    return {
        "base_model_id": base_model.id,
        "base_model_f1": base_f1,
        "baseline_f1": baseline.result.metrics["test"]["f1_macro"],
        "plan": out["plan"],
        "trials": trials,
        "best": best_summary,
        "best_model_id": best_model.id if best_model else None,
        "improvement": best_summary["delta_vs_baseline"],
        "selection": out["selection"],
        "seconds": out["seconds"],
        "recommendation": recommendation,
    }
