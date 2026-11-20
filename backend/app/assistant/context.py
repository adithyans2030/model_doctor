"""Build the grounding context the assistant answers from.

Everything here comes from stored, deterministic analysis: the diagnosis,
models, experiments, explanations and drift reports. The LLM only ever sees this
JSON, so it explains results rather than inventing a diagnosis.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Dataset, Job, MLModel
from app.services.diagnosis import build_diagnosis
from app.services.reports import latest_report

MAX_FINDINGS = 25


def _scalar_evidence(ev: dict) -> dict:
    """Keep small scalar evidence values; drop curves, tables and nested lists."""
    out = {}
    for k, v in ev.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, bool)) or x is None for x in v.values()):
            if len(v) <= 8:
                out[k] = v
    return out


def _finding(f: dict) -> dict:
    return {k: v for k, v in {
        "check": f["check"], "severity": f["severity"], "title": f["title"],
        "evidence": _scalar_evidence(f.get("evidence") or {}), "cause": f.get("cause"),
        "recommendation": f.get("recommendation"), "resolved": f.get("resolved") or None,
    }.items() if v not in (None, {}, [])}


def _model_row(m: MLModel) -> dict:
    t = m.metrics["test"]
    return {
        "id": m.id, "name": m.name, "version": m.version, "stage": m.stage, "source": m.source,
        "config": m.config.get("algorithm") and {k: v for k, v in m.config.items() if v not in (None, [], {}, False)},
        "test": {k: t.get(k) for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc", "ece")},
        "minority_class": t.get("positive_class") or t["worst_class"],
        "checks": [f["check"] for f in m.findings],
    }


_TOPICS = {
    "recall": ("model.weak_class", "data.class_imbalance", "model.misleading_accuracy", "model.suboptimal_threshold",
               "prediction.fn_concentration"),
    "drift": ("distribution.",),
    "overfit": ("model.overfitting", "model.high_variance", "model.underfitting"),
    "leak": ("feature.leakage", "model.suspicious_performance", "model.dominant_feature"),
    "fair": ("fairness.",),
    "data": ("data.", "feature."),
}


def compact_context(ctx: dict, question: str, max_findings: int = 8) -> dict:
    """A small, question-focused slice of the context for local models.

    Local CPU inference is slow, and prompt size dominates latency, so only the
    findings related to the question (plus the most severe ones) are kept, and
    evidence is reduced to a few headline numbers.
    """
    q = question.lower()
    prefixes = [p for topic, ps in _TOPICS.items() if topic in q for p in ps]
    if any(w in q for w in ("minority", "false negative", "precision", "miss")):
        prefixes += list(_TOPICS["recall"])
    active = [f for f in ctx["findings"] if not f.get("resolved")]
    related = [f for f in active if prefixes and f["check"].startswith(tuple(prefixes))]
    rest = [f for f in active if f not in related]
    chosen = (related + rest)[:max_findings]

    def slim(f):
        ev = {k: v for k, v in (f.get("evidence") or {}).items() if isinstance(v, (int, float))}
        return {"check": f["check"], "severity": f["severity"], "title": f["title"],
                **({"cause": f["cause"]} if f.get("cause") else {}), **({"numbers": dict(list(ev.items())[:4])} if ev else {})}

    sel = ctx.get("selected_model") or {}
    models = ctx["models"]
    selected = next((m for m in models if m["id"] == sel.get("id")), None)
    out = {
        "dataset": {k: ctx["dataset"][k] for k in ("target", "task_type", "rows")},
        "class_distribution": [{"label": c["label"], "share": c["share"]} for c in (ctx["dataset"].get("class_distribution") or [])],
        "health": ctx["health"],
        "selected_model": selected and {"name": selected["name"], "version": selected["version"], "stage": selected["stage"],
                                        "test": selected["test"], "minority_class": selected["minority_class"]},
        "findings": [slim(f) for f in chosen],
        "recommendations": [r["label"] for r in ctx["recommendations"][:4]],
        "summary": ctx["narrative"][:4],
    }
    if any(w in q for w in ("compare", "version", " vs", "versus")):
        out["models"] = [{"name": m["name"], "version": m["version"], "test": m["test"],
                          "minority_recall": m["minority_class"]["recall"]} for m in models]
    if ctx.get("explanations"):
        e = ctx["explanations"]
        out["top_features"] = e["top_features_shap"][:5] or e["top_features_permutation"][:5]
        out["error_patterns"] = e["error_statements"][:2]
    if ctx.get("experiments"):
        x = ctx["experiments"][0]
        out["latest_experiment"] = {"best": x["best"], "improvement": x["improvement"], "recommendation": x["recommendation"]}
    if ctx.get("drift"):
        out["drift"] = {k: ctx["drift"][k] for k in ("score", "drifted_features", "top")}
    return out


def build_context(db: Session, settings: Settings, dataset: Dataset, model_id: str | None = None) -> dict:
    diagnosis = build_diagnosis(db, settings, dataset, model_id)
    models = list(db.scalars(select(MLModel).where(MLModel.dataset_id == dataset.id).order_by(MLModel.version)))
    selected = next((m for m in models if diagnosis["model"] and m.id == diagnosis["model"]["id"]), None)
    data_report = latest_report(db, dataset.id, "data")

    ctx: dict = {
        "dataset": {
            "id": dataset.id, "filename": dataset.original_filename, "rows": dataset.n_rows,
            "columns": dataset.n_cols, "target": dataset.target_column, "task_type": dataset.task_type,
            "class_distribution": dataset.class_distribution,
        },
        "data_quality": data_report and {
            "score": data_report.payload["score"]["value"],
            "summary": {k: data_report.payload["summary"].get(k) for k in
                        ("missing_cells_pct", "duplicate_rows", "imbalance_ratio", "excluded_columns")},
        },
        "selected_model": diagnosis["model"],
        "health": {"overall": diagnosis["health"]["overall"],
                   "components": {k: v["value"] for k, v in diagnosis["health"]["components"].items()}},
        "findings": [_finding(f) for f in diagnosis["findings"][:MAX_FINDINGS]],
        "recommendations": [{"intervention": r["intervention"], "label": r["label"], "severity": r["severity"],
                             "because": [b["check"] for b in r["because"]]} for r in diagnosis["recommendations"]],
        "narrative": diagnosis["narrative"],
        "models": [_model_row(m) for m in models],
    }

    if selected and selected.explanations:
        e = selected.explanations
        shap = (e["global"].get("shap") or {}).get("values") or []
        ctx["explanations"] = {
            "top_features_shap": [v["feature"] for v in shap[:6]],
            "top_features_permutation": [v["feature"] for v in e["global"]["permutation"]["values"][:6]],
            "error_statements": e["error_analysis"]["statements"],
            "error_summary": e["error_analysis"]["summary"],
            "fairness_max_gap": e["fairness"].get("max_gap"),
            "fairness_worst_feature": e["fairness"].get("worst_feature"),
        }

    jobs = db.scalars(select(Job).where(Job.dataset_id == dataset.id, Job.kind == "experiment",
                                        Job.status == "completed").order_by(Job.created_at.desc()).limit(3))
    ctx["experiments"] = [{
        "job_id": j.id, "base_model_id": j.params.get("base_model_id"),
        "best_model_id": (j.result or {}).get("best_model_id"),
        "best": (j.result or {}).get("best", {}).get("description"),
        "improvement": (j.result or {}).get("improvement"),
        "trials": [{"name": t["name"], "kept": t["kept"], "cv_f1_macro": t["cv_f1_macro"],
                    "test_f1_macro": t["metrics"]["f1_macro"], "minority_recall": t["metrics"]["minority_recall"]}
                   for t in (j.result or {}).get("trials", [])],
        "recommendation": (j.result or {}).get("recommendation"),
    } for j in jobs]

    if selected:
        drift = latest_report(db, dataset.id, "drift", model_id=selected.id)
        if drift:
            p = drift.payload
            ctx["drift"] = {"score": p["score"], "drifted_features": p["drifted_features"],
                            "top": [{"feature": f["feature"], "psi": f["psi"], "severity": f["severity"]}
                                    for f in p["features"][:5]],
                            "performance": p.get("performance")}
    return ctx
