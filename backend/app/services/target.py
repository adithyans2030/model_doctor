"""Target column validation and task-type inference."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.errors import InvalidTargetError, UnsupportedTaskError

_UNUSABLE = {
    "empty": "contains no values",
    "id": "looks like an identifier (every value is unique)",
    "text": "looks like free text",
    "datetime": "contains dates/timestamps",
}


@dataclass
class TargetAnalysis:
    task_type: str
    class_distribution: list[dict] | None
    warnings: list[str] = field(default_factory=list)
    summary: dict | None = None  # regression: distribution of the target


def _numeric_values(series: pd.Series) -> pd.Series:
    nonnull = series.dropna()
    values = pd.to_numeric(nonnull, errors="coerce")
    if values.isna().any():
        bad = nonnull[values.isna()].astype(str).unique()[:3]
        raise InvalidTargetError(
            f"Target '{series.name}' is mostly numeric but contains non-numeric "
            f"values such as {', '.join(repr(b) for b in bad)}."
        )
    return values.astype(float)


def is_regression_target(series: pd.Series, profile: dict, settings: Settings) -> bool:
    """Numeric targets with non-integer values or many distinct values are regression."""
    if profile["inferred_type"] != "numeric":
        return False
    values = _numeric_values(series)
    integral = bool(np.all(np.mod(values, 1) == 0))
    return not integral or values.nunique() > settings.max_integer_target_classes


def regression_values(series: pd.Series) -> pd.Series:
    """Non-missing target values as floats (regression)."""
    return _numeric_values(series)


def normalize_labels(series: pd.Series, profile: dict, settings: Settings) -> pd.Series:
    """Normalise target values to string class labels (classification targets only)."""
    inferred = profile["inferred_type"]
    nonnull = series.dropna()

    if inferred == "numeric":
        values = _numeric_values(series)
        if is_regression_target(series, profile, settings):
            raise UnsupportedTaskError(
                f"Target '{series.name}' is continuous; it is a regression target, not class labels."
            )
        return values.astype("int64").astype(str)

    labels = nonnull.astype(str).str.strip()
    if inferred == "boolean":
        labels = labels.str.lower()
    return labels


def analyze_target(series: pd.Series, profile: dict, settings: Settings) -> TargetAnalysis:
    name = series.name
    reason = _UNUSABLE.get(profile["inferred_type"])
    if reason:
        raise InvalidTargetError(f"Column '{name}' can't be a target: it {reason}.")

    n_missing = int(series.isna().sum())
    n_present = len(series) - n_missing
    if n_present < settings.min_rows:
        raise InvalidTargetError(
            f"Target '{name}' has only {n_present} non-missing values; "
            f"at least {settings.min_rows} are needed."
        )

    if is_regression_target(series, profile, settings):
        return _analyze_regression(series, n_missing)

    labels = normalize_labels(series, profile, settings)
    counts = labels.value_counts()
    n_classes = len(counts)
    if n_classes < 2:
        raise InvalidTargetError(f"Target '{name}' has a single value, so there is nothing to predict.")
    if n_classes > settings.max_classes:
        raise InvalidTargetError(
            f"Target '{name}' has {n_classes} classes; the limit is {settings.max_classes}."
        )

    warnings: list[str] = []
    if n_missing:
        warnings.append(
            f"{n_missing} rows ({100 * n_missing / len(series):.1f}%) have no target value "
            "and will be excluded from training."
        )

    lowered = labels.str.lower().str.replace(r"\s+", " ", regex=True)
    if lowered.nunique() < n_classes:
        groups = (
            pd.DataFrame({"label": labels, "key": lowered})
            .drop_duplicates()
            .groupby("key")["label"]
            .apply(list)
        )
        variants = [" / ".join(sorted(v)) for v in groups if len(v) > 1]
        warnings.append(
            "Some labels differ only by case or spacing and are treated as separate classes: "
            + "; ".join(variants[:5])
            + "."
        )

    small = counts[counts < settings.min_class_samples]
    if not small.empty:
        listed = ", ".join(f"'{lbl}' ({cnt})" for lbl, cnt in small.items())
        warnings.append(
            f"Classes with fewer than {settings.min_class_samples} samples: {listed}. "
            "Evaluation on these classes will be unreliable."
        )

    distribution = [
        {"label": str(lbl), "count": int(cnt), "share": round(cnt / n_present, 4)}
        for lbl, cnt in counts.items()
    ]
    task_type = "binary_classification" if n_classes == 2 else "multiclass_classification"
    return TargetAnalysis(task_type, distribution, warnings)


SKEW_WARN = 1.0


def _analyze_regression(series: pd.Series, n_missing: int) -> TargetAnalysis:
    values = regression_values(series)
    name = series.name
    if values.nunique() < 2:
        raise InvalidTargetError(f"Target '{name}' has a single value, so there is nothing to predict.")
    q = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    lo, hi = q[0.01], q[0.99]
    counts, edges = np.histogram(values.clip(lo, hi) if lo < hi else values, bins=20)
    skew = float(values.skew()) if len(values) > 2 else 0.0
    summary = {
        "n": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "min": float(values.min()), "q1": float(q[0.25]), "median": float(q[0.5]), "q3": float(q[0.75]),
        "max": float(values.max()), "skew": round(skew, 3), "n_unique": int(values.nunique()),
        "all_non_negative": bool((values >= 0).all()),
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
    }
    warnings: list[str] = []
    if n_missing:
        warnings.append(
            f"{n_missing} rows ({100 * n_missing / len(series):.1f}%) have no target value "
            "and will be excluded from training."
        )
    if abs(skew) >= SKEW_WARN:
        hint = " A log transform (tested automatically by the optimiser) often helps." if summary["all_non_negative"] else ""
        warnings.append(f"The target is strongly skewed (skewness {skew:.2f}).{hint}")
    return TargetAnalysis("regression", None, warnings, summary)
