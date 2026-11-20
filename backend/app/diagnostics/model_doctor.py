"""Phase 3: the Model Doctor.

Turns a trained model's metrics into findings: over/underfitting, variance,
no-skill models, weak classes, misleading accuracy, poor calibration,
sub-optimal thresholds and suspiciously good performance. When a data-level
finding explains a model symptom (e.g. imbalance -> low minority recall), the
model finding names it as the likely cause.
"""

from app.diagnostics.findings import finding, sort_findings

OVERFIT_WARN, OVERFIT_CRIT = 0.10, 0.20
CV_STD_WARN = 0.05
UNDERFIT_F1 = 0.65
NO_SKILL_MARGIN = 0.05
WEAK_CLASS_F1 = 0.60
CLASS_GAP = 0.25
ACCURACY_GAP = 0.10
ECE_WARN = 0.08
THRESHOLD_GAIN = 0.03
SUSPICIOUS_F1, SUSPICIOUS_AUC = 0.98, 0.995
DOMINANT_SHARE = 0.5


def _has(data_findings, check):
    return next((f for f in data_findings if f["check"] == check), None)


def diagnose_model(metrics: dict, importances: list[dict] | None, data_findings: list[dict]) -> list[dict]:
    out: list[dict] = []
    test, cv, train, dummy = metrics["test"], metrics.get("cv"), metrics["train"], metrics.get("dummy")
    f1 = test["f1_macro"]

    # --- generalisation --------------------------------------------------------
    if cv:
        train_f1, val_f1 = cv["train"]["f1_macro"]["mean"], cv["val"]["f1_macro"]["mean"]
        std = cv["val"]["f1_macro"]["std"]
        source = f"{cv['folds']}-fold CV"
    else:
        train_f1, val_f1, std, source = train["f1_macro"], f1, None, "train vs test"
    gap = round(train_f1 - val_f1, 4)
    if gap >= OVERFIT_WARN:
        out.append(finding(
            "model.overfitting", "model", "critical" if gap >= OVERFIT_CRIT else "warning",
            f"Overfitting: training F1 exceeds validation F1 by {gap:.2f}",
            {"train_f1_macro": train_f1, "validation_f1_macro": val_f1, "gap": gap, "source": source},
            cause="The model memorises training data (too complex, too little data, or noisy/duplicated rows).",
            impact="Performance on new data will be lower than training suggests.",
            recommendation="Regularise (shallower trees, larger min_samples_leaf, stronger C) and tune hyperparameters.",
            interventions=["hyperparameter_tuning"]))
    if std is not None and std >= CV_STD_WARN:
        out.append(finding(
            "model.high_variance", "model", "warning",
            f"Unstable performance across folds (F1 std = {std:.3f})",
            {"cv_f1_macro_std": std, "cv_f1_macro_mean": val_f1},
            cause="Results depend heavily on which rows are in the training set; often small data or rare classes.",
            recommendation="Collect more data, simplify the model, or use repeated cross-validation for decisions."))

    if dummy and test["balanced_accuracy"] <= dummy["balanced_accuracy"] + NO_SKILL_MARGIN:
        out.append(finding(
            "model.no_skill", "model", "critical", "Model is barely better than always predicting the majority class",
            {"balanced_accuracy": test["balanced_accuracy"], "baseline_balanced_accuracy": dummy["balanced_accuracy"]},
            cause="Features carry little signal for the target, or the model collapsed to the majority class.",
            recommendation="Check feature relevance and class balance; try class weighting.",
            interventions=["class_weight"]))
    elif train_f1 < UNDERFIT_F1 and val_f1 < UNDERFIT_F1:
        out.append(finding(
            "model.underfitting", "model", "warning",
            f"Underfitting: low F1 on both training ({train_f1:.2f}) and validation ({val_f1:.2f})",
            {"train_f1_macro": train_f1, "validation_f1_macro": val_f1},
            cause="The model is too simple for the pattern, or the features don't contain enough signal.",
            recommendation="Try a more flexible model, engineer features, or tune hyperparameters.",
            interventions=["hyperparameter_tuning"]))

    # --- class-level performance --------------------------------------------------
    per_class = test["per_class"]
    best_cls = max(per_class, key=lambda c: c["f1"])
    worst_cls = min(per_class, key=lambda c: c["f1"])
    imbalance = _has(data_findings, "data.class_imbalance")
    if worst_cls["f1"] < WEAK_CLASS_F1 and best_cls["f1"] - worst_cls["f1"] >= CLASS_GAP:
        severe = worst_cls["recall"] < 0.5
        cause = (
            f"Class imbalance (ratio {imbalance['evidence']['imbalance_ratio']}): the model sees few "
            f"'{worst_cls['label']}' examples and favours the majority."
            if imbalance else
            "The class overlaps with others in feature space or has too few examples."
        )
        interventions = ["class_weight", "smote", "threshold_tuning"] if imbalance else ["class_weight"]
        out.append(finding(
            "model.weak_class", "model", "critical" if severe else "warning",
            f"Poor performance on class '{worst_cls['label']}' (recall {worst_cls['recall']:.0%}, F1 {worst_cls['f1']:.2f})",
            {"class": worst_cls, "best_class": best_cls, "per_class": per_class},
            cause=cause,
            impact=(f"Most '{worst_cls['label']}' cases are missed (high false-negative rate)." if severe
                    else f"'{worst_cls['label']}' predictions are unreliable."),
            recommendation="Use class-weighted training or SMOTE, and tune the decision threshold.",
            interventions=interventions))

    if test["accuracy"] - test["balanced_accuracy"] >= ACCURACY_GAP:
        out.append(finding(
            "model.misleading_accuracy", "model", "warning",
            f"Accuracy ({test['accuracy']:.0%}) overstates performance; balanced accuracy is {test['balanced_accuracy']:.0%}",
            {"accuracy": test["accuracy"], "balanced_accuracy": test["balanced_accuracy"], "f1_macro": f1},
            cause="With imbalanced classes, predicting the majority class scores well on accuracy.",
            recommendation="Judge models by macro-F1, balanced accuracy or PR-AUC (ModelDoctor ranks by macro-F1)."))

    # --- probabilities and threshold -------------------------------------------
    if test["ece"] >= ECE_WARN:
        out.append(finding(
            "model.poor_calibration", "model", "warning",
            f"Predicted probabilities are poorly calibrated (ECE = {test['ece']:.3f})",
            {"ece": test["ece"], "calibration_curve": test["calibration_curve"]},
            cause="Probabilities don't match observed frequencies (common after resampling/weighting, or with SVM/RF).",
            impact="Risk scores can't be read as probabilities; threshold decisions become unreliable.",
            recommendation="Apply sigmoid or isotonic calibration.",
            interventions=["calibration"]))

    ta = test.get("threshold_analysis")
    if ta and test.get("threshold") is None and ta["best_f1"] - ta["f1_at_0_5"] >= THRESHOLD_GAIN:
        out.append(finding(
            "model.suboptimal_threshold", "model", "warning",
            f"Decision threshold 0.5 is sub-optimal; {ta['best_threshold']:.2f} gives F1 {ta['best_f1']:.2f} vs {ta['f1_at_0_5']:.2f}",
            {"best_threshold": ta["best_threshold"], "best_f1": ta["best_f1"], "f1_at_0_5": ta["f1_at_0_5"],
             "positive_class": metrics.get("positive_label")},
            recommendation="Choose the threshold on validation data to match the cost of false negatives vs false positives.",
            interventions=["threshold_tuning"]))

    # --- too good to be true ------------------------------------------------------
    leak = _has(data_findings, "feature.leakage")
    top = importances[0] if importances else None
    suspicious = f1 >= SUSPICIOUS_F1 or (test.get("roc_auc") or 0) >= SUSPICIOUS_AUC
    if suspicious or (top and top["importance"] >= DOMINANT_SHARE):
        dominant_is_leak = bool(leak and top and top["feature"] in leak["columns"])
        out.append(finding(
            "model.suspicious_performance" if suspicious else "model.dominant_feature",
            "model", "critical" if dominant_is_leak else "warning",
            (f"Suspiciously high performance (macro-F1 {f1:.2f})" if suspicious
             else f"One feature ('{top['feature']}') drives {top['importance']:.0%} of the model"),
            {"f1_macro": f1, "roc_auc": test.get("roc_auc"), "top_feature": top},
            cause=("The dominant feature was flagged for target leakage by the Data Doctor." if dominant_is_leak
                   else "Either the problem is easy or a feature leaks information about the target."),
            impact="Production performance may be far lower if this feature isn't available at prediction time.",
            recommendation="Verify the top feature is known at prediction time; retrain without it if not.",
            interventions=["drop_leaky_features"] if dominant_is_leak else [],
            columns=[top["feature"]] if top else []))

    return sort_findings(out)
