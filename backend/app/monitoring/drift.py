"""Phase 8: distribution drift between training data and a production batch.

Per feature:
* numeric     PSI over 10 training-quantile bins + two-sample Kolmogorov-Smirnov test
* categorical PSI over training categories (+ "other" for unseen) + chi-square test
Plus missing-rate shift, unseen-category share, prediction drift (PSI of the
predicted-class mix vs the test set) and, if the batch is labelled, the change
in macro-F1 versus the held-out test set (concept drift).

PSI rule of thumb: < 0.10 stable, 0.10–0.25 moderate shift, > 0.25 major shift.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from app.diagnostics.findings import finding, sort_findings, summarize_counts
from app.ml.features import FeatureSpec

EPS = 1e-4
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
MISSING_SHIFT = 0.10
UNSEEN_WARN = 0.05
PERF_WARN, PERF_CRIT = 0.05, 0.10
TOP_CATS = 30


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _severity(v: float) -> str:
    return "major" if v >= PSI_MAJOR else "moderate" if v >= PSI_MODERATE else "stable"


def _numeric(ref: pd.Series, cur: pd.Series) -> dict:
    r, c = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
    out = {"type": "numeric", "ref_mean": float(np.mean(r)) if len(r) else None,
           "batch_mean": float(np.mean(c)) if len(c) else None}
    if len(r) < 2 or len(c) < 2:
        return {**out, "psi": 0.0, "test": None, "p_value": None}
    edges = np.unique(np.quantile(r, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        edges = np.array([r.min(), r.max() + EPS])
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(r, bins=edges)[0]
    a_counts = np.histogram(c, bins=edges)[0]
    ks = ks_2samp(r, c)
    return {**out, "psi": round(psi(e_counts, a_counts), 4), "test": "ks",
            "statistic": round(float(ks.statistic), 4), "p_value": float(ks.pvalue)}


def _categorical(ref: pd.Series, cur: pd.Series) -> dict:
    r, c = ref.dropna().astype(str), cur.dropna().astype(str)
    cats = list(r.value_counts().index[:TOP_CATS])
    rv = r.where(r.isin(cats), "__other__").value_counts()
    cv = c.where(c.isin(cats), "__other__").value_counts()
    keys = cats + ["__other__"]
    e = np.array([rv.get(k, 0) for k in keys], dtype=float)
    a = np.array([cv.get(k, 0) for k in keys], dtype=float)
    p_value = None
    table = np.vstack([e, a])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] >= 2:
        p_value = float(chi2_contingency(table)[1])
    unseen = float((~c.isin(set(r.unique()))).mean()) if len(c) else 0.0
    shares_r = (e / e.sum()) if e.sum() else e
    shares_c = (a / a.sum()) if a.sum() else a
    moved = sorted(zip(keys, shares_r, shares_c), key=lambda t: -abs(t[2] - t[1]))[:3]
    return {"type": "categorical", "psi": round(psi(e, a), 4), "test": "chi2", "p_value": p_value,
            "unseen_share": round(unseen, 4),
            "largest_shifts": [{"value": k, "ref_share": round(float(x), 4), "batch_share": round(float(y), 4)}
                               for k, x, y in moved]}


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, spec: FeatureSpec) -> list[dict]:
    rows = []
    for col in spec.features:
        info = _numeric(ref[col], cur[col]) if col in spec.numeric_like else _categorical(ref[col], cur[col])
        info.update(
            feature=col, severity=_severity(info["psi"]),
            missing_ref=round(float(ref[col].isna().mean()), 4),
            missing_batch=round(float(cur[col].isna().mean()), 4),
        )
        rows.append(info)
    return sorted(rows, key=lambda r: -r["psi"])


def drift_report(features: list[dict], prediction: dict, performance: dict | None, schema: dict) -> dict:
    findings = []
    for f in [r for r in features if r["severity"] != "stable"][:8]:
        detail = (f"mean {f['ref_mean']:.4g} → {f['batch_mean']:.4g}" if f["type"] == "numeric" and f["ref_mean"] is not None
                  else "; ".join(f"{s['value']}: {s['ref_share']:.0%} → {s['batch_share']:.0%}" for s in f.get("largest_shifts", [])[:2]))
        findings.append(finding(
            "distribution.feature_drift", "distribution", "critical" if f["severity"] == "major" else "warning",
            f"{'Major' if f['severity'] == 'major' else 'Moderate'} drift in '{f['feature']}' (PSI {f['psi']:.2f}; {detail})",
            {k: f[k] for k in ("feature", "psi", "test", "p_value", "missing_ref", "missing_batch") if k in f},
            cause="The production population differs from the training population for this feature.",
            impact="Predictions for shifted regions rely on patterns learned from different data.",
            recommendation="Check the upstream data source; retrain on recent data if the shift is real.",
            columns=[f["feature"]]))
    for f in features:
        if abs(f["missing_batch"] - f["missing_ref"]) >= MISSING_SHIFT:
            findings.append(finding(
                "distribution.missing_shift", "distribution", "warning",
                f"Missing rate of '{f['feature']}' changed {f['missing_ref']:.0%} → {f['missing_batch']:.0%}",
                {"feature": f["feature"], "missing_ref": f["missing_ref"], "missing_batch": f["missing_batch"]},
                cause="A pipeline or collection change upstream.", columns=[f["feature"]]))
        if f.get("unseen_share", 0) >= UNSEEN_WARN:
            findings.append(finding(
                "distribution.unseen_categories", "distribution", "warning",
                f"{f['unseen_share']:.0%} of '{f['feature']}' values were never seen in training",
                {"feature": f["feature"], "unseen_share": f["unseen_share"]},
                impact="Unseen categories are encoded as 'infrequent' and carry no learned signal.",
                columns=[f["feature"]]))
    if prediction["psi"] >= PSI_MODERATE:
        findings.append(finding(
            "distribution.prediction_drift", "distribution", "critical" if prediction["psi"] >= PSI_MAJOR else "warning",
            f"The predicted-class mix shifted (PSI {prediction['psi']:.2f})",
            prediction, impact="The model behaves differently on this batch than on the test set."))
    if performance and performance.get("delta_f1_macro") is not None:
        drop = -performance["delta_f1_macro"]
        if drop >= PERF_WARN:
            findings.append(finding(
                "distribution.concept_drift", "distribution", "critical" if drop >= PERF_CRIT else "warning",
                f"Macro-F1 dropped from {performance['test_f1_macro']:.2f} (test) to {performance['batch_f1_macro']:.2f} on this batch",
                performance, cause="The relationship between features and target has changed, or the population shifted.",
                recommendation="Retrain with recent labelled data and re-run the diagnosis.",
                interventions=["hyperparameter_tuning"]))
    if schema["extra_columns"]:
        findings.append(finding(
            "distribution.schema_change", "distribution", "info",
            f"{len(schema['extra_columns'])} column(s) in the batch are not used by the model",
            schema, columns=schema["extra_columns"]))

    psis = np.array([f["psi"] for f in features]) if features else np.array([0.0])
    score = 100 * (1 - np.clip(psis / (2 * PSI_MAJOR), 0, 1).mean())
    if performance and performance.get("delta_f1_macro") is not None:
        score -= 200 * max(0.0, -performance["delta_f1_macro"])
    score = int(round(max(0.0, min(100.0, score))))
    findings = sort_findings(findings)
    return {
        "score": score,
        "drifted_features": sum(f["severity"] != "stable" for f in features),
        "n_features": len(features),
        "features": features,
        "prediction_drift": prediction,
        "performance": performance,
        "schema": schema,
        "findings": findings,
        "severity_counts": summarize_counts(findings),
        "thresholds": {"psi_moderate": PSI_MODERATE, "psi_major": PSI_MAJOR},
    }


def class_mix_psi(ref_pred: np.ndarray, cur_pred: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    e = np.bincount(ref_pred, minlength=k).astype(float)
    a = np.bincount(cur_pred, minlength=k).astype(float)
    return {"psi": round(psi(e, a), 4),
            "reference": {c: round(float(x / e.sum()), 4) for c, x in zip(classes, e)},
            "batch": {c: round(float(x / a.sum()), 4) for c, x in zip(classes, a)}}
