"""Phase 4: error analysis and group performance (fairness) on the held-out test set.

Instead of reading thousands of predictions, find *slices* (feature = value, or
a numeric quartile) where the model errs much more often than average, and
report where false negatives concentrate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from app.diagnostics.findings import finding
from app.ml.features import FeatureSpec

MIN_SLICE = 20
MIN_SLICE_SHARE = 0.03
LIFT_REPORT, LIFT_WARN = 1.3, 1.5
MAX_CATS = 30
UNCERTAIN_CONF = 0.6
CONFIDENT_WRONG = 0.9
FAIR_WARN, FAIR_CRIT = 0.15, 0.30


def _fmt_value(v: float, is_date: bool) -> str:
    if is_date:
        return (pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(v))).strftime("%Y-%m-%d")
    return f"{v:.4g}"


def _slice_labels(s: pd.Series, numeric: bool, is_date: bool = False) -> pd.Series | None:
    if numeric and s.nunique() > 10:
        try:
            bins = pd.qcut(s, q=4, duplicates="drop")
        except ValueError:
            return None
        # Label each quartile by the actual min-max of its values (qcut edges are padded).
        ranges = s.groupby(bins, observed=True).agg(["min", "max"])
        names = {iv: f"{_fmt_value(r['min'], is_date)} – {_fmt_value(r['max'], is_date)}" for iv, r in ranges.iterrows()}
        labels = bins.map(names).astype(object)
        return labels.where(s.notna(), "missing").astype(str)
    if s.nunique() > MAX_CATS:
        return None
    return s.astype(object).where(s.notna(), "missing").astype(str)


def analyse_errors(X: pd.DataFrame, spec: FeatureSpec, y: np.ndarray, pred: np.ndarray, proba: np.ndarray,
                   classes: list[str], positive_index: int | None, row_ids: list) -> dict:
    n = len(y)
    wrong = pred != y
    overall = float(wrong.mean()) if n else 0.0
    fn = (y == positive_index) & (pred != positive_index) if positive_index is not None else None
    n_fn = int(fn.sum()) if fn is not None else 0
    min_n = max(MIN_SLICE, int(MIN_SLICE_SHARE * n))
    numeric = set(spec.numeric_like)

    slices, concentration = [], []
    for col in spec.features:
        labels = _slice_labels(X[col].reset_index(drop=True), col in numeric, col in spec.datetime)
        if labels is None:
            continue
        for value, idx in labels.groupby(labels).groups.items():
            idx = np.asarray(idx)
            if len(idx) < min_n:
                continue
            err = float(wrong[idx].mean())
            entry = {
                "feature": col, "value": str(value), "n": int(len(idx)), "share": round(len(idx) / n, 4),
                "error_rate": round(err, 4), "lift": round(err / overall, 3) if overall else None,
                "excess_errors": round((err - overall) * len(idx), 2),
            }
            if fn is not None and n_fn:
                entry["fn_share"] = round(float(fn[idx].sum()) / n_fn, 4)
                if entry["fn_share"] >= 0.5 and entry["fn_share"] - entry["share"] >= 0.15:
                    concentration.append(entry)
            if overall and err / overall >= LIFT_REPORT:
                slices.append(entry)
    slices.sort(key=lambda e: -e["excess_errors"])
    concentration.sort(key=lambda e: -(e["fn_share"] - e["share"]))

    conf = proba.max(axis=1)
    uncertain = conf < UNCERTAIN_CONF
    confident_wrong = wrong & (conf >= CONFIDENT_WRONG)
    order = np.argsort(-(conf * wrong))
    worst = [
        {"row_id": row_ids[i], "true": classes[int(y[i])], "predicted": classes[int(pred[i])],
         "confidence": round(float(conf[i]), 4)}
        for i in order[: min(20, int(wrong.sum()))]
    ]

    summary = {
        "n_test": n, "n_errors": int(wrong.sum()), "error_rate": round(overall, 4),
        "false_negatives": n_fn if fn is not None else None,
        "false_positives": int(((y != positive_index) & (pred == positive_index)).sum()) if positive_index is not None else None,
        "uncertain_share": round(float(uncertain.mean()), 4) if n else 0,
        "uncertain_error_rate": round(float(wrong[uncertain].mean()), 4) if uncertain.any() else None,
        "confident_errors": int(confident_wrong.sum()),
    }
    statements = [
        f"{e['fn_share']:.0%} of false negatives occur where {e['feature']} = {e['value']} "
        f"({e['share']:.0%} of test rows)." for e in concentration[:3]
    ] + [
        f"The model is wrong {e['error_rate']:.0%} of the time when {e['feature']} = {e['value']} "
        f"vs {overall:.0%} overall ({e['lift']:.1f}× the average)." for e in slices[:3]
    ]
    return {"summary": summary, "slices": slices[:15], "fn_concentration": concentration[:5],
            "statements": statements, "most_confident_errors": worst}


def group_performance(X: pd.DataFrame, spec: FeatureSpec, y: np.ndarray, pred: np.ndarray,
                      classes: list[str], positive_index: int | None) -> dict:
    k = len(classes)
    groups_out, max_gap, worst_col = [], 0.0, None
    for col in spec.categorical:
        s = X[col].reset_index(drop=True).astype(object).where(X[col].reset_index(drop=True).notna(), "missing")
        if not 2 <= s.nunique() <= 10:
            continue
        rows = []
        for value, idx in s.groupby(s).groups.items():
            idx = np.asarray(idx)
            if len(idx) < MIN_SLICE:
                continue
            yy, pp = y[idx], pred[idx]
            row = {"value": str(value), "n": int(len(idx)), "accuracy": round(float((yy == pp).mean()), 4),
                   "f1_macro": round(float(f1_score(yy, pp, average="macro", labels=list(range(k)), zero_division=0)), 4),
                   "balanced_accuracy": None, "positive_recall": None}
            if len(np.unique(yy)) >= 2:
                row["balanced_accuracy"] = round(float(balanced_accuracy_score(yy, pp)), 4)
            if positive_index is not None and (yy == positive_index).any():
                row["positive_recall"] = round(float((pp[yy == positive_index] == positive_index).mean()), 4)
            rows.append(row)
        scored = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
        if len(scored) < 2:
            continue
        gap = round(max(scored) - min(scored), 4)
        groups_out.append({"feature": col, "gap": gap, "groups": rows})
        if gap > max_gap:
            max_gap, worst_col = gap, col
    groups_out.sort(key=lambda g: -g["gap"])
    return {"metric": "balanced_accuracy", "max_gap": round(max_gap, 4) if groups_out else None,
            "worst_feature": worst_col, "features": groups_out}


def prediction_findings(errors: dict, fairness: dict) -> list[dict]:
    out = []
    for e in [s for s in errors["slices"] if (s["lift"] or 0) >= LIFT_WARN][:3]:
        out.append(finding(
            "prediction.error_cluster", "prediction", "warning",
            f"Errors cluster where {e['feature']} = {e['value']} ({e['error_rate']:.0%} vs {errors['summary']['error_rate']:.0%})",
            e, cause="The model hasn't learned this segment well (few examples, different behaviour, or missing signal).",
            recommendation="Inspect this segment; add segment-specific features or more examples.",
            columns=[e["feature"]]))
    for e in errors["fn_concentration"][:2]:
        out.append(finding(
            "prediction.fn_concentration", "prediction", "warning",
            f"{e['fn_share']:.0%} of false negatives occur where {e['feature']} = {e['value']}",
            e, impact="Missed positives are concentrated in one segment.",
            recommendation="Consider a lower threshold or dedicated features for this segment.",
            interventions=["threshold_tuning"], columns=[e["feature"]]))
    s = errors["summary"]
    if s["n_test"] and s["confident_errors"] / s["n_test"] >= 0.02:
        out.append(finding(
            "prediction.confident_errors", "prediction", "warning",
            f"{s['confident_errors']} predictions are wrong with ≥{CONFIDENT_WRONG:.0%} confidence",
            {"confident_errors": s["confident_errors"], "examples": errors["most_confident_errors"][:5]},
            cause="Possible label noise, or the model relies on a shortcut feature.",
            recommendation="Review these rows' labels first."))
    if s["uncertain_share"] >= 0.2:
        out.append(finding(
            "prediction.uncertain", "prediction", "info",
            f"{s['uncertain_share']:.0%} of predictions have confidence below {UNCERTAIN_CONF:.0%}",
            {"uncertain_share": s["uncertain_share"], "uncertain_error_rate": s["uncertain_error_rate"]},
            recommendation="Route low-confidence predictions to human review."))
    if fairness["max_gap"] is not None and fairness["max_gap"] >= FAIR_WARN:
        g = next(f for f in fairness["features"] if f["feature"] == fairness["worst_feature"])
        scored = [r for r in g["groups"] if r["balanced_accuracy"] is not None]
        lo = min(scored, key=lambda r: r["balanced_accuracy"])
        hi = max(scored, key=lambda r: r["balanced_accuracy"])
        out.append(finding(
            "fairness.group_gap", "fairness", "critical" if fairness["max_gap"] >= FAIR_CRIT else "warning",
            f"Performance differs across '{g['feature']}' groups: {lo['value']} {lo['balanced_accuracy']:.2f} "
            f"vs {hi['value']} {hi['balanced_accuracy']:.2f} balanced accuracy",
            {"feature": g["feature"], "gap": g["gap"], "groups": g["groups"]},
            impact="Some groups get systematically worse predictions.",
            recommendation="Check representation of the weaker group; evaluate per-group thresholds or reweighting.",
            columns=[g["feature"]]))
    return out
