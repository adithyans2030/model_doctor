"""Phase 2: the Data Doctor.

Runs deterministic data-health and feature-health checks over a dataset and
returns a report with a summary, per-column statistics, findings and a
transparent data-quality score. No model is trained here except tiny
single-feature probes used to detect target leakage.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from app.diagnostics.findings import finding, score_from_findings, sort_findings, summarize_counts
from app.ml.features import build_feature_spec, labelled_rows, prepare_frame

PROBE_SAMPLE = 20_000
CORR_SAMPLE = 50_000

# Thresholds (documented in docs/phase-2-data-doctor.md)
MISSING_WARN, MISSING_CRIT = 5.0, 40.0
DUPLICATE_WARN = 1.0
OUTLIER_IQR_K = 3.0
OUTLIER_WARN_PCT = 0.25
NEAR_CONSTANT_SHARE = 0.99
HIGH_CARDINALITY = 50
IMBALANCE_WARN, IMBALANCE_CRIT = 3.0, 20.0
LEAK_CRIT, LEAK_WARN = 0.95, 0.90
CORR_WARN, CORR_INFO = 0.90, 0.80
CRAMERS_WARN = 0.90
LOW_RELEVANCE_MI = 0.002
SMALL_DATASET = 500


# ----------------------------------------------------------------------------- helpers
def _sample(df: pd.DataFrame, n: int, seed: int = 0, stratify: pd.Series | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df
    if stratify is None:
        return df.sample(n, random_state=seed)
    frac = n / len(df)
    return df.groupby(stratify, group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
    )


def _encode_for_probe(x: pd.Series, is_numeric: bool) -> np.ndarray:
    if is_numeric:
        arr = x.astype(float).to_numpy()
        fill = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
        missing = np.isnan(arr)
        arr = np.where(missing, fill, arr)
        return np.column_stack([arr, missing.astype(float)])
    codes = pd.Series(pd.Categorical(x.fillna("__missing__"))).cat.codes.to_numpy()
    return codes.reshape(-1, 1).astype(float)


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.fillna("__na__"), b.fillna("__na__"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def _column_stats(df: pd.DataFrame, columns: list[dict]) -> list[dict]:
    stats = []
    for col in columns:
        name, kind = col["name"], col["inferred_type"]
        s = df[name]
        entry = {"name": name, "inferred_type": kind, "missing_pct": col["missing_pct"], "n_unique": col["n_unique"]}
        if kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr
                n_out = int(((v < lo) | (v > hi)).sum()) if iqr > 0 and v.nunique() > 10 else 0
                entry.update(
                    mean=float(v.mean()), std=float(v.std(ddof=0)), min=float(v.min()), q1=float(q1),
                    median=float(med), q3=float(q3), max=float(v.max()),
                    skew=float(v.skew()) if len(v) > 2 else 0.0, n_outliers=n_out,
                    outlier_bounds=[float(lo), float(hi)],
                    histogram=_histogram(v),
                )
        elif kind in {"categorical", "boolean"}:
            vc = s.astype(str).where(s.notna()).value_counts().head(10)
            entry["top_values"] = [
                {"value": str(k), "count": int(c), "share": round(c / len(s), 4)} for k, c in vc.items()
            ]
        stats.append(entry)
    return stats


def _histogram(v: pd.Series, bins: int = 20) -> dict:
    # Clip to the 1st-99th percentile so a few outliers don't flatten the chart.
    lo, hi = v.quantile([0.01, 0.99])
    if lo == hi:
        lo, hi = v.min(), v.max()
    if lo == hi:
        return {"edges": [float(lo), float(hi)], "counts": [int(len(v))]}
    counts, edges = np.histogram(v.clip(lo, hi), bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


# ----------------------------------------------------------------------------- checks
def _check_columns(df, columns, target, findings):
    n = len(df)
    for col in columns:
        name, kind = col["name"], col["inferred_type"]
        if name == target:
            continue
        pct = col["missing_pct"]
        if kind == "empty":
            findings.append(finding(
                "data.empty_column", "data", "warning", f"Column '{name}' is completely empty",
                {"column": name}, recommendation="Drop the column.", columns=[name]))
            continue
        if pct > 0:
            sev = "critical" if pct >= MISSING_CRIT else "warning" if pct >= MISSING_WARN else "info"
            findings.append(finding(
                "data.missing_values", "data", sev, f"{pct:.1f}% missing values in '{name}'",
                {"column": name, "missing_pct": pct, "n_missing": col["n_missing"]},
                impact="Models need imputation; heavy missingness can bias results or hide signal.",
                recommendation=(
                    "Consider dropping this column or collecting more data."
                    if sev == "critical" else
                    "Impute (median / most frequent) and keep a missing-indicator feature."
                ),
                columns=[name]))
        if col["n_unique"] == 1:
            findings.append(finding(
                "data.constant_column", "data", "warning", f"Column '{name}' has a single value",
                {"column": name, "value": col["sample_values"][:1]},
                impact="Carries no information and is excluded from modelling.",
                recommendation="Drop the column.", columns=[name]))
            continue
        if kind == "id":
            findings.append(finding(
                "data.id_column", "data", "info", f"'{name}' looks like an identifier",
                {"column": name, "n_unique": col["n_unique"]},
                impact="IDs don't generalise; they are excluded from modelling automatically.",
                recommendation="Keep it for joining/lookup, not as a feature.", columns=[name]))
        elif kind == "text":
            findings.append(finding(
                "data.text_column", "data", "info", f"'{name}' is free text and is not used as a feature",
                {"column": name}, recommendation="Engineer features (length, keywords, embeddings) if it matters.",
                columns=[name]))
        elif kind == "datetime":
            findings.append(finding(
                "data.datetime_column", "data", "info", f"'{name}' is a date and is converted to a number of days",
                {"column": name},
                recommendation="Consider deriving age/recency/seasonality features. Check that dates don't leak the future.",
                columns=[name]))
        elif kind == "categorical":
            if col["n_unique"] > HIGH_CARDINALITY:
                findings.append(finding(
                    "data.high_cardinality", "data", "warning",
                    f"'{name}' has {col['n_unique']} distinct categories",
                    {"column": name, "n_unique": col["n_unique"]},
                    impact="One-hot encoding creates many sparse features; rare categories overfit.",
                    recommendation="Group rare categories or use frequency/target encoding.", columns=[name]))
            top_share = df[name].value_counts(normalize=True, dropna=False).iloc[0]
            if top_share >= NEAR_CONSTANT_SHARE:
                findings.append(finding(
                    "data.near_constant", "data", "info", f"'{name}' is {top_share:.1%} a single value",
                    {"column": name, "top_share": round(float(top_share), 4)},
                    recommendation="Likely little signal; consider dropping.", columns=[name]))
        if col.get("numeric_stored_as_text"):
            s = df[name].dropna()
            bad = s[pd.to_numeric(s, errors="coerce").isna()]
            if len(bad):
                findings.append(finding(
                    "data.invalid_values", "data", "warning",
                    f"{len(bad)} non-numeric values in numeric column '{name}'",
                    {"column": name, "n_invalid": int(len(bad)), "pct": round(100 * len(bad) / n, 2),
                     "examples": [str(v) for v in bad.unique()[:5]]},
                    impact="These values are treated as missing.",
                    recommendation="Fix the source data or map the placeholders to proper missing values.",
                    columns=[name]))


def _check_outliers(stats, findings, n_rows):
    for s in stats:
        n_out = s.get("n_outliers", 0)
        if not n_out:
            continue
        pct = 100 * n_out / n_rows
        sev = "warning" if pct >= OUTLIER_WARN_PCT else "info"
        findings.append(finding(
            "data.outliers", "data", sev, f"{n_out} extreme outliers in '{s['name']}' ({pct:.2f}%)",
            {"column": s["name"], "n_outliers": n_out, "pct": round(pct, 3),
             "bounds": s["outlier_bounds"], "max": s["max"], "median": s["median"]},
            cause=f"Values beyond {OUTLIER_IQR_K}×IQR from the quartiles (data-entry errors, unit mix-ups or genuine extremes).",
            impact="Can distort linear models and scaling; trees are more robust.",
            recommendation="Verify these values; cap (winsorise) or fix them if they are errors.",
            columns=[s["name"]]))


def _check_duplicates(df, target, columns, findings):
    n = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        pct = 100 * n_dup / n
        findings.append(finding(
            "data.duplicate_rows", "data", "warning" if pct >= DUPLICATE_WARN else "info",
            f"{n_dup} duplicate rows ({pct:.1f}%)", {"n_duplicates": n_dup, "pct": round(pct, 2)},
            impact="Duplicates can land in both train and test splits and inflate scores.",
            recommendation="Remove exact duplicates before splitting.", interventions=["drop_duplicates"]))
    if target and target in df.columns:
        skip = {"id", "text", "empty"}
        features = [c["name"] for c in columns if c["name"] != target and c["inferred_type"] not in skip]
        if not features:
            return n_dup
        grouped = df.dropna(subset=[target]).groupby(features, dropna=False)[target].nunique()
        conflicts = int((grouped > 1).sum())
        if conflicts:
            findings.append(finding(
                "data.conflicting_labels", "data", "warning",
                f"{conflicts} groups of identical rows have different labels",
                {"n_groups": conflicts},
                cause="Label noise, or features that don't capture what determines the label.",
                impact="Sets a ceiling on achievable accuracy.",
                recommendation="Review these rows' labels."))
    return n_dup


def _check_target(dataset, findings) -> float | None:
    dist = dataset.class_distribution or []
    if len(dist) < 2:
        return None
    counts = [c["count"] for c in dist]
    ratio = max(counts) / max(1, min(counts))
    minority = min(dist, key=lambda c: c["count"])
    if ratio >= IMBALANCE_WARN:
        findings.append(finding(
            "data.class_imbalance", "data", "critical" if ratio >= IMBALANCE_CRIT else "warning",
            "Significant class imbalance" if ratio >= IMBALANCE_CRIT else "Class imbalance detected",
            {"imbalance_ratio": round(ratio, 2), "minority_class": minority["label"],
             "minority_share": minority["share"], "distribution": dist},
            impact=(f"Models are pulled towards the majority class; recall for '{minority['label']}' "
                    "is likely to suffer and accuracy becomes misleading."),
            recommendation="Evaluate with macro-F1 / balanced accuracy; try class weighting, SMOTE and threshold tuning.",
            interventions=["class_weight", "smote", "threshold_tuning"], columns=[dataset.target_column]))
    return round(ratio, 2)


def _feature_target_checks(df, dataset, spec, settings, findings) -> list[dict]:
    """Leakage probes and relevance (mutual information) per feature."""
    profile = next(c for c in dataset.columns if c["name"] == dataset.target_column)
    frame, y = labelled_rows(df, dataset.target_column, profile, settings)
    X = prepare_frame(frame, spec)
    sample_idx = _sample(pd.DataFrame(index=X.index), PROBE_SAMPLE, stratify=y).index
    X, y = X.loc[sample_idx], y.loc[sample_idx]
    classes = sorted(y.unique())
    y_codes = y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    binary = len(classes) == 2
    min_count = int(pd.Series(y_codes).value_counts().min())
    folds = min(3, min_count)

    results = []
    numeric_set = set(spec.numeric_like)
    encoded = {c: _encode_for_probe(X[c], c in numeric_set) for c in spec.features}

    for c in spec.features:
        entry = {"column": c, "score": None, "metric": "roc_auc" if binary else "balanced_accuracy"}
        if folds >= 2:
            probe = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)
            cv = StratifiedKFold(folds, shuffle=True, random_state=0)
            if binary:
                proba = cross_val_predict(probe, encoded[c], y_codes, cv=cv, method="predict_proba")[:, 1]
                score = roc_auc_score(y_codes, proba)
                score = max(score, 1 - score)
            else:
                pred = cross_val_predict(probe, encoded[c], y_codes, cv=cv)
                score = balanced_accuracy_score(y_codes, pred)
            entry["score"] = round(float(score), 4)
        results.append(entry)

    # Mutual information (discrete for categoricals, kNN estimate for numerics)
    mat = np.column_stack([encoded[c][:, 0] for c in spec.features])
    discrete = [c not in numeric_set for c in spec.features]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(mat, y_codes, discrete_features=discrete, random_state=0)
    for entry, value in zip(results, mi):
        entry["mutual_info"] = round(float(value), 5)

    leaks = [r for r in results if r["score"] is not None and r["score"] >= LEAK_WARN]
    for r in leaks:
        crit = r["score"] >= LEAK_CRIT
        findings.append(finding(
            "feature.leakage", "feature", "critical" if crit else "warning",
            f"Possible target leakage: '{r['column']}' alone predicts the target",
            {"column": r["column"], r["metric"]: r["score"], "threshold": LEAK_CRIT if crit else LEAK_WARN},
            cause=("A single feature predicts the target almost perfectly. It may be recorded after the outcome "
                   "(e.g. a cancellation flag for churn) or be derived from the label."),
            impact="Offline metrics look excellent but the model will fail in production where this value is unavailable.",
            recommendation="Confirm when this value becomes known. If it is only known after the outcome, remove it.",
            interventions=["drop_leaky_features"], columns=[r["column"]]))

    low = [r for r in results if r["mutual_info"] < LOW_RELEVANCE_MI and (r["score"] is None or r["score"] < 0.55)]
    if low:
        findings.append(finding(
            "feature.low_relevance", "feature", "info",
            f"{len(low)} feature(s) show almost no relationship with the target",
            {"columns": [r["column"] for r in low], "mutual_info": {r["column"]: r["mutual_info"] for r in low}},
            impact="Irrelevant features add noise and can slightly hurt generalisation.",
            recommendation="Try removing them (feature selection) and compare.",
            interventions=["feature_selection"], columns=[r["column"] for r in low]))
    return sorted(results, key=lambda r: -(r["score"] or 0))


def _correlation_checks(df, spec, findings) -> list[dict]:
    pairs = []
    sample = _sample(df, CORR_SAMPLE)
    num = pd.DataFrame({c: pd.to_numeric(sample[c], errors="coerce") for c in spec.numeric})
    if num.shape[1] >= 2:
        corr = num.corr(method="pearson")
        for a, b in itertools.combinations(spec.numeric, 2):
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= CORR_INFO:
                pairs.append({"a": a, "b": b, "method": "pearson", "value": round(float(r), 4)})
    cats = [c for c in spec.categorical if 1 < sample[c].nunique() <= HIGH_CARDINALITY]
    for a, b in itertools.combinations(cats, 2):
        v = _cramers_v(sample[a], sample[b])
        if v >= CRAMERS_WARN:
            pairs.append({"a": a, "b": b, "method": "cramers_v", "value": round(v, 4)})

    for p in pairs:
        strong = abs(p["value"]) >= (CORR_WARN if p["method"] == "pearson" else CRAMERS_WARN)
        findings.append(finding(
            "feature.multicollinearity", "feature", "warning" if strong else "info",
            f"'{p['a']}' and '{p['b']}' are highly correlated ({p['method']} = {p['value']:.2f})",
            p, impact="Redundant features make linear-model coefficients unstable and importance harder to interpret.",
            recommendation="Keep one of the pair or combine them.",
            interventions=["feature_selection"] if strong else [], columns=[p["a"], p["b"]]))
    return pairs


# ----------------------------------------------------------------------------- entry point
def run_data_doctor(df: pd.DataFrame, dataset, settings) -> dict:
    findings: list[dict] = []
    columns = dataset.columns
    target = dataset.target_column
    n = len(df)

    _check_columns(df, columns, target, findings)
    stats = _column_stats(df, columns)
    _check_outliers(stats, findings, n)
    n_dup = _check_duplicates(df, target, columns, findings)
    if n < SMALL_DATASET:
        findings.append(finding(
            "data.small_dataset", "data", "info", f"Small dataset ({n} rows)", {"n_rows": n},
            impact="Metrics will have wide confidence intervals; prefer cross-validation and simple models."))

    imbalance = None
    feature_target: list[dict] = []
    correlations: list[dict] = []
    spec = build_feature_spec(columns, target) if target else None
    if target:
        imbalance = _check_target(dataset, findings)
        if spec.features:
            feature_target = _feature_target_checks(df, dataset, spec, settings, findings)
            correlations = _correlation_checks(df, spec, findings)
    else:
        findings.append(finding(
            "data.no_target", "data", "info", "No target column selected",
            recommendation="Select a target to enable imbalance, leakage and relevance checks."))

    findings = sort_findings(findings)
    type_counts: dict[str, int] = {}
    for c in columns:
        type_counts[c["inferred_type"]] = type_counts.get(c["inferred_type"], 0) + 1
    total_cells = n * len(columns)
    summary = {
        "n_rows": n,
        "n_cols": len(columns),
        "n_features_usable": len(spec.features) if spec else None,
        "excluded_columns": spec.excluded if spec else {},
        "missing_cells_pct": round(100 * float(df.isna().sum().sum()) / total_cells, 2) if total_cells else 0,
        "duplicate_rows": n_dup,
        "duplicate_pct": round(100 * n_dup / n, 2) if n else 0,
        "type_counts": type_counts,
        "target": target,
        "task_type": dataset.task_type,
        "class_distribution": dataset.class_distribution,
        "imbalance_ratio": imbalance,
        "severity_counts": summarize_counts(findings),
    }
    return {
        "summary": summary,
        "score": score_from_findings(findings),
        "findings": findings,
        "column_stats": stats,
        "feature_target": feature_target,
        "correlations": correlations,
    }
