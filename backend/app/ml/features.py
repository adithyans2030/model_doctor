"""Turning a stored dataset into model-ready features.

``prepare_frame`` cleans raw columns according to the upload-time schema profile;
``FeatureSpec`` records which columns are used and how, so the exact same
preparation can be replayed at prediction / monitoring time.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.services.target import is_regression_target, normalize_labels, regression_values

EXCLUDED_TYPES = {"id", "text", "empty"}


@dataclass
class FeatureSpec:
    target: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)  # includes boolean
    datetime: list[str] = field(default_factory=list)  # converted to numeric (days)
    excluded: dict[str, str] = field(default_factory=dict)  # column -> reason

    @property
    def features(self) -> list[str]:
        return self.numeric + self.datetime + self.categorical

    @property
    def numeric_like(self) -> list[str]:
        return self.numeric + self.datetime

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(**d)


def build_feature_spec(columns: list[dict], target: str, drop: list[str] | tuple = ()) -> FeatureSpec:
    spec = FeatureSpec(target=target)
    drop = set(drop)
    for col in columns:
        name, kind = col["name"], col["inferred_type"]
        if name == target:
            continue
        if name in drop:
            spec.excluded[name] = "dropped by configuration"
        elif kind in EXCLUDED_TYPES:
            spec.excluded[name] = f"{kind} column"
        elif col["n_unique"] <= 1:
            spec.excluded[name] = "constant column"
        elif kind == "numeric":
            spec.numeric.append(name)
        elif kind == "datetime":
            spec.datetime.append(name)
        else:  # categorical, boolean
            spec.categorical.append(name)
    return spec


def prepare_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return only the spec's feature columns, coerced to consistent dtypes."""
    missing = [c for c in spec.features if c not in df.columns]
    if missing:
        raise KeyError(missing)
    out = pd.DataFrame(index=df.index)
    for c in spec.numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in spec.datetime:
        parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        days = (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        out[c] = days.astype(float)
    for c in spec.categorical:
        s = df[c]
        out[c] = s.where(s.isna(), s.astype(str).str.strip()).astype(object)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[spec.features]


def original_feature(transformed_name: str, features: list[str]) -> str:
    """Map a ColumnTransformer output name back to its source column.

    e.g. ``num__age`` -> ``age``, ``num__missingindicator_age`` -> ``age``,
    ``cat__contract_Month-to-month`` -> ``contract``.
    """
    name = transformed_name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        name = name[len("missingindicator_"):]
    if name in features:
        return name
    matches = [f for f in features if name.startswith(f + "_")]
    return max(matches, key=len) if matches else name


def labelled_rows(df: pd.DataFrame, target: str, profile: dict, settings) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows without a target and return (frame, target).

    Classification targets come back as normalised string labels, regression
    targets as floats.
    """
    mask = df[target].notna()
    frame = df.loc[mask]
    if is_regression_target(frame[target], profile, settings):
        values = regression_values(frame[target])
        return frame, values.loc[frame.index]
    labels = normalize_labels(frame[target], profile, settings)
    return frame, labels.loc[frame.index]
