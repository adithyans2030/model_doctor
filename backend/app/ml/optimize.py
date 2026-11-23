"""Phase 5: the optimisation engine.

The findings' ``interventions`` become a plan of experiments that run as a
greedy sequence: each step starts from the best configuration so far, changes
one thing, and is kept only if it improves the **cross-validated primary metric
on the training data** (macro-F1 for classification, R² for regression). The
fixed test set is used for reporting, never for choosing.

    classification: Baseline (leaks removed) → class weights | SMOTE → deduplicate → feature selection
                    → hyperparameter search → calibration → decision threshold
    regression:     Baseline (leaks removed) → log-transform target → deduplicate → feature selection
                    → hyperparameter search
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

from app.diagnostics.findings import INTERVENTIONS
from app.ml.algorithms import ALGORITHMS
from app.ml.evaluation import best_threshold
from app.ml.pipeline import TrainConfig, build_pipeline, config_spec, fit_params, search_prefix
from app.ml.training import (
    TrainingData, TrainResult, _training_rows, cross_validate_config, train_and_evaluate,
)

SELECTION_FOLDS = 3
MIN_GAIN = 0.002
METRIC_LABEL = {"f1_macro": "macro-F1", "r2": "R²"}

CLASSIFICATION_ORDER = ["class_weight", "smote", "drop_duplicates", "feature_selection", "hyperparameter_tuning",
                        "calibration", "threshold_tuning"]
REGRESSION_ORDER = ["target_transform", "drop_duplicates", "feature_selection", "hyperparameter_tuning"]


@dataclass
class Trial:
    step: int
    name: str
    intervention: str | None
    config: TrainConfig
    result: TrainResult
    kept: bool = False
    reason: str = ""

    @property
    def primary(self) -> str:
        return self.result.metrics["primary_metric"]

    @property
    def cv_score(self) -> float:
        cv = self.result.metrics.get("cv")
        if cv and cv["val"].get(self.primary):
            return cv["val"][self.primary]["mean"]
        return self.result.metrics["test"][self.primary]


@dataclass
class Plan:
    leaky_columns: list[str]
    interventions: list[str]
    feature_drops: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def minority_recall(metrics: dict) -> float:
    t = metrics["test"]
    return (t.get("positive_class") or t["worst_class"])["recall"]


def build_plan(findings: list[dict], data_report: dict, base: TrainConfig, binary: bool,
               allowed: list[str] | None = None, task: str = "classification",
               target_non_negative: bool = False) -> Plan:
    wanted = {i for f in findings for i in f["interventions"]}
    if allowed is not None:
        wanted &= set(allowed)
    leaky = sorted({c for f in findings if f["check"] == "feature.leakage" and "drop_leaky_features" in f["interventions"]
                    for c in f["columns"]} - set(base.drop_columns))
    if "drop_leaky_features" not in wanted:
        leaky = []

    # Feature selection candidates: low-relevance features + the weaker column of strongly correlated pairs.
    drops: list[str] = []
    if "feature_selection" in wanted:
        mi = {r["column"]: r.get("mutual_info", 0) for r in data_report.get("feature_target", [])}
        for f in findings:
            if f["check"] == "feature.low_relevance":
                drops += f["columns"]
            elif f["check"] == "feature.multicollinearity" and "feature_selection" in f["interventions"]:
                a, b = f["columns"]
                drops.append(a if mi.get(a, 0) < mi.get(b, 0) else b)
        drops = sorted(set(drops) - set(base.drop_columns) - set(leaky))
        if not drops:
            wanted.discard("feature_selection")

    notes = []
    if task == "regression":
        order = REGRESSION_ORDER
        if "target_transform" in wanted and (not target_non_negative or base.target_transform):
            wanted.discard("target_transform")
            if not target_non_negative:
                notes.append("Log-transform skipped: the target has negative values.")
    else:
        order = CLASSIFICATION_ORDER
        if not binary:
            wanted.discard("threshold_tuning")
        if base.algorithm == "svm":
            wanted.discard("calibration")  # the SVM is already Platt-calibrated
    plan = Plan(leaky_columns=leaky, interventions=[i for i in order if i in wanted], feature_drops=drops, notes=notes)
    if leaky:
        plan.notes.append(f"Leaking features removed from every trial: {', '.join(leaky)}.")
    return plan


def tune_hyperparameters(cfg: TrainConfig, data: TrainingData, n_iter: int, n_jobs: int) -> dict:
    spec_cfg = config_spec(data.spec, cfg)
    X, y = _training_rows(cfg, data.X_train[spec_cfg.features], data.y_train)
    pipe = build_pipeline(cfg, spec_cfg, y, n_jobs, data.task)
    prefix = search_prefix(cfg)
    space = {prefix + k: v for k, v in ALGORITHMS[cfg.algorithm].search_space[data.task].items()}
    if data.task == "regression":
        scoring, cv = "r2", KFold(SELECTION_FOLDS, shuffle=True, random_state=data.seed)
    else:
        scoring, cv = "f1_macro", StratifiedKFold(SELECTION_FOLDS, shuffle=True, random_state=data.seed)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring=scoring, cv=cv,
                                random_state=data.seed, n_jobs=1, error_score=np.nan, refit=False)
    search.fit(X, y, **fit_params(cfg, y, data.task))
    return {k.removeprefix(prefix): _py(v) for k, v in search.best_params_.items()}


def run_optimisation(
    base: TrainConfig, data: TrainingData, findings: list[dict], data_report: dict, *,
    budget: str = "standard", allowed: list[str] | None = None, hpo_iterations: int = 12, n_jobs: int = -1,
    dummy: dict | None = None, progress: Callable[[str], None] | None = None,
) -> dict:
    binary = data.positive_index is not None
    plan = build_plan(findings, data_report, base, binary, allowed, data.task,
                      target_non_negative=bool(data.task == "regression" and np.min(data.y_train) >= 0))
    if budget == "quick":
        if "hyperparameter_tuning" in plan.interventions:
            plan.interventions.remove("hyperparameter_tuning")
            plan.notes.append("Hyperparameter search skipped (quick budget).")
    elif "hyperparameter_tuning" not in plan.interventions and (allowed is None or "hyperparameter_tuning" in allowed):
        # Defaults are rarely optimal, so the standard budget always includes a search.
        after = [i for i in ("calibration", "threshold_tuning") if i in plan.interventions]
        pos = plan.interventions.index(after[0]) if after else len(plan.interventions)
        plan.interventions.insert(pos, "hyperparameter_tuning")
    say = progress or (lambda _: None)
    trials: list[Trial] = []
    label = METRIC_LABEL[data.primary]

    def evaluate(name, intervention, cfg) -> Trial:
        say(f"Trial {len(trials) + 1}: {name}")
        res = train_and_evaluate(cfg, data, cv_folds=SELECTION_FOLDS, n_jobs=n_jobs, dummy=dummy)
        t = Trial(len(trials), name, intervention, cfg, res)
        trials.append(t)
        return t

    start = replace(base, drop_columns=sorted(set(base.drop_columns) | set(plan.leaky_columns)))
    best = evaluate("Baseline (leaking features removed)" if plan.leaky_columns else "Baseline", None, start)
    best.kept, best.reason = True, "starting point"

    def consider(t: Trial):
        nonlocal best
        if t.cv_score > best.cv_score + MIN_GAIN:
            t.kept, t.reason = True, f"CV {label} {best.cv_score:.3f} → {t.cv_score:.3f}"
            best = t
        else:
            t.reason = f"no CV gain ({t.cv_score:.3f} vs {best.cv_score:.3f})"

    started = time.perf_counter()
    imbalance = [i for i in ("class_weight", "smote") if i in plan.interventions]
    if imbalance:
        anchor = best
        options = []
        if "class_weight" in imbalance:
            options.append(evaluate("Class weighting", "class_weight", replace(anchor.config, class_weight="balanced")))
        if "smote" in imbalance:
            options.append(evaluate("SMOTE oversampling", "smote", replace(anchor.config, resampling="smote")))
        for t in sorted(options, key=lambda t: -t.cv_score):
            consider(t)

    if "target_transform" in plan.interventions:
        consider(evaluate("Log-transform the target", "target_transform", replace(best.config, target_transform="log1p")))

    if "drop_duplicates" in plan.interventions:
        consider(evaluate("Remove duplicate training rows", "drop_duplicates", replace(best.config, dedupe=True)))

    if "feature_selection" in plan.interventions:
        cfg = replace(best.config, drop_columns=sorted(set(best.config.drop_columns) | set(plan.feature_drops)))
        consider(evaluate(f"Feature selection (drop {', '.join(plan.feature_drops)})", "feature_selection", cfg))

    if "hyperparameter_tuning" in plan.interventions:
        say(f"Hyperparameter search ({hpo_iterations} candidates × {SELECTION_FOLDS} folds)")
        params = tune_hyperparameters(best.config, data, hpo_iterations, n_jobs)
        consider(evaluate("Hyperparameter search", "hyperparameter_tuning", replace(best.config, params=params)))

    if "calibration" in plan.interventions:
        consider(evaluate("Sigmoid calibration", "calibration", replace(best.config, calibration="sigmoid")))

    if "threshold_tuning" in plan.interventions and binary:
        say("Choosing a decision threshold on out-of-fold predictions")
        oof = cross_validate_config(best.config, data, SELECTION_FOLDS, n_jobs)
        if oof is not None:
            thr = best_threshold(data.y_train, oof["oof_proba"], data.positive_index)
            if thr != 0.5:
                consider(evaluate(f"Decision threshold {thr:.2f}", "threshold_tuning", replace(best.config, threshold=thr)))
            else:
                plan.notes.append("Threshold tuning: 0.5 was already optimal on out-of-fold predictions.")

    baseline = trials[0]
    return {
        "plan": {"interventions": plan.interventions, "leaky_columns": plan.leaky_columns,
                 "feature_drops": plan.feature_drops, "notes": plan.notes,
                 "labels": {i: INTERVENTIONS[i] for i in plan.interventions}},
        "trials": trials,
        "baseline": baseline,
        "best": best,
        "seconds": round(time.perf_counter() - started, 1),
        "selection": f"{SELECTION_FOLDS}-fold cross-validated {label} on the training split; "
                     "the test set is only used to report results.",
    }


def trial_metrics(metrics: dict) -> dict:
    test = metrics["test"]
    if metrics.get("task") == "regression":
        return {"primary": test["r2"], "r2": test["r2"], "rmse": test["rmse"], "mae": test["mae"], "mape": test.get("mape")}
    return {
        "primary": test["f1_macro"], "f1_macro": test["f1_macro"], "accuracy": test["accuracy"],
        "balanced_accuracy": test["balanced_accuracy"], "roc_auc": test["roc_auc"], "pr_auc": test.get("pr_auc"),
        "ece": test["ece"], "minority_recall": minority_recall(metrics),
    }


def trial_summary(t: Trial, baseline: Trial) -> dict:
    m, b = trial_metrics(t.result.metrics), trial_metrics(baseline.result.metrics)
    delta = {"primary": round(m["primary"] - b["primary"], 4)}
    if "minority_recall" in m:
        delta.update(f1_macro=delta["primary"], minority_recall=round(m["minority_recall"] - b["minority_recall"], 4))
    else:
        delta.update(r2=delta["primary"], rmse=round(m["rmse"] - b["rmse"], 4))
    return {
        "step": t.step, "name": t.name, "intervention": t.intervention,
        "description": t.config.describe(), "config": t.config.to_dict(),
        "kept": t.kept, "reason": t.reason, "cv_primary": round(t.cv_score, 4),
        "metrics": m,
        "delta_vs_baseline": delta,
    }
