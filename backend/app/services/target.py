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
    class_distribution: list[dict]
    warnings: list[str] = field(default_factory=list)


def normalize_labels(series: pd.Series, profile: dict, settings: Settings) -> pd.Series:
    """Normalise target values to string class labels, or reject regression targets."""
    inferred = profile["inferred_type"]
    nonnull = series.dropna()

    if inferred == "numeric":
        values = pd.to_numeric(nonnull, errors="coerce")
        if values.isna().any():
            bad = nonnull[values.isna()].astype(str).unique()[:3]
            raise InvalidTargetError(
                f"Target '{series.name}' is mostly numeric but contains non-numeric "
                f"values such as {', '.join(repr(b) for b in bad)}."
            )
        integral = bool(np.all(np.mod(values, 1) == 0))
        n_unique = values.nunique()
        if not integral or n_unique > settings.max_integer_target_classes:
            raise UnsupportedTaskError(
                f"Target '{series.name}' looks continuous ({n_unique} distinct numeric values), "
                "which is a regression task. ModelDoctor currently supports classification; "
                "regression is on the roadmap."
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
