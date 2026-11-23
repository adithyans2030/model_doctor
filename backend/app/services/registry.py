"""Phase 7: model registry with a human-approval gate, and model comparison.

    candidate ──approve──► approved ──promote──► production ──► archived
        │                      │
        └──reject──► rejected  └──reject──► rejected

* Only an **approved** model can be promoted to production (human in the loop).
* Approving a model with unresolved *critical* findings requires explicitly
  acknowledging the risks.
* Promoting archives the project's previous production model.
* Every transition is recorded in ``model_events`` with actor and note.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import DomainError
from app.db.models import MLModel, ModelEvent

TRANSITIONS = {
    "candidate": {"approved", "rejected", "archived"},
    "approved": {"production", "rejected", "archived", "candidate"},
    "production": {"archived"},
    "rejected": {"candidate", "archived"},
    "archived": {"candidate"},
}
NOTE_REQUIRED = {"approved", "rejected", "production"}


class TransitionError(DomainError):
    status_code = 409
    code = "invalid_transition"


def record_event(db: Session, model: MLModel, from_stage: str | None, to_stage: str, actor: str, note: str | None):
    db.add(ModelEvent(model_id=model.id, project_id=model.project_id, from_stage=from_stage,
                      to_stage=to_stage, actor=actor, note=note))


def critical_findings(model: MLModel) -> list[dict]:
    found = list(model.findings)
    if model.explanations:
        found += model.explanations.get("findings", [])
    return [f for f in found if f["severity"] == "critical"]


def change_stage(db: Session, model: MLModel, to_stage: str, *, actor: str, note: str | None,
                 acknowledge_risks: bool = False) -> MLModel:
    current = model.stage
    if to_stage == current:
        raise TransitionError(f"Model is already '{current}'.")
    if to_stage not in TRANSITIONS[current]:
        hint = " Approve it first." if to_stage == "production" else ""
        raise TransitionError(f"Can't move a model from '{current}' to '{to_stage}'.{hint}")
    if to_stage in NOTE_REQUIRED and not (note and note.strip()):
        raise DomainError(f"A note is required to move a model to '{to_stage}'.", code="note_required")
    if to_stage == "approved":
        crit = critical_findings(model)
        if crit and not acknowledge_risks:
            titles = "; ".join(f["title"] for f in crit[:5])
            raise TransitionError(
                f"This model has {len(crit)} critical finding(s): {titles}. "
                "Set acknowledge_risks=true to approve anyway.",
                code="unresolved_critical_findings",
            )

    if to_stage == "production":
        others = db.scalars(select(MLModel).where(
            MLModel.project_id == model.project_id, MLModel.stage == "production", MLModel.id != model.id))
        for other in others:
            record_event(db, other, "production", "archived", actor, f"Replaced by {model.name}.")
            other.stage = "archived"

    record_event(db, model, current, to_stage, actor,
                 (note or "").strip() + (" [critical findings acknowledged]" if acknowledge_risks and to_stage == "approved" else "") or None)
    model.stage = to_stage
    db.commit()
    db.refresh(model)
    return model


# ----------------------------------------------------------------------------- comparison
_METRICS = {
    "classification": ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc", "pr_auc", "ece", "log_loss"],
    "regression": ["r2", "rmse", "mae", "median_ae", "mape", "explained_variance"],
}
_LOWER_IS_BETTER = {"ece", "log_loss", "rmse", "mae", "median_ae", "mape"}


def _minority(m: MLModel) -> dict:
    t = m.metrics["test"]
    return t.get("positive_class") or t["worst_class"]


def compare_models(a: MLModel, b: MLModel) -> dict:
    if a.dataset_id != b.dataset_id:
        raise DomainError("Models must be trained on the same dataset to be compared.", code="invalid_request")
    ta, tb = a.metrics["test"], b.metrics["test"]
    same_split = a.metrics["split"] == b.metrics["split"]
    rows = []
    if a.task != b.task:
        raise DomainError("Models must solve the same task to be compared.", code="invalid_request")
    for k in _METRICS[a.task]:
        va, vb = ta.get(k), tb.get(k)
        delta = None if va is None or vb is None else round(vb - va, 4)
        better = None
        if delta:
            better = "b" if (delta < 0 if k in _LOWER_IS_BETTER else delta > 0) else "a"
        rows.append({"metric": k, "a": va, "b": vb, "delta": delta, "better": better})
    if a.task == "classification":
        ma, mb = _minority(a), _minority(b)
        rows.append({"metric": f"recall ({ma['label']})", "a": ma["recall"], "b": mb["recall"],
                     "delta": round(mb["recall"] - ma["recall"], 4),
                     "better": "b" if mb["recall"] > ma["recall"] else "a" if mb["recall"] < ma["recall"] else None})

    config_diff = {k: {"a": a.config.get(k), "b": b.config.get(k)}
                   for k in sorted(set(a.config) | set(b.config)) if a.config.get(k) != b.config.get(k)}
    checks_a = {f["check"] for f in a.findings}
    checks_b = {f["check"] for f in b.findings}

    winner = b if b.primary_score > a.primary_score else a
    loser = a if winner is b else b
    d = abs(b.primary_score - a.primary_score)
    label = "R²" if a.task == "regression" else "macro-F1"
    summary = [
        f"{winner.name} (v{winner.version}) has the higher {label} ({winner.primary_score:.3f} vs "
        f"{loser.primary_score:.3f}, Δ {d:.3f}).",
    ]
    if a.task == "classification":
        summary.append(f"Minority-class recall: v{a.version} {ma['recall']:.0%} → v{b.version} {mb['recall']:.0%}.")
    else:
        summary.append(f"RMSE: v{a.version} {ta['rmse']:.4g} → v{b.version} {tb['rmse']:.4g}.")
    if checks_a - checks_b:
        summary.append(f"v{b.version} resolves: {', '.join(sorted(checks_a - checks_b))}.")
    if checks_b - checks_a:
        summary.append(f"v{b.version} introduces: {', '.join(sorted(checks_b - checks_a))}.")
    if not same_split:
        summary.append("Warning: the models were evaluated on different test splits, so differences are not directly comparable.")
    return {
        "a": {"id": a.id, "name": a.name, "version": a.version, "stage": a.stage},
        "b": {"id": b.id, "name": b.name, "version": b.version, "stage": b.stage},
        "same_test_split": same_split,
        "metrics": rows,
        "config_diff": config_diff,
        "findings": {"resolved_in_b": sorted(checks_a - checks_b), "new_in_b": sorted(checks_b - checks_a),
                     "shared": sorted(checks_a & checks_b)},
        "winner": winner.id,
        "summary": summary,
    }
