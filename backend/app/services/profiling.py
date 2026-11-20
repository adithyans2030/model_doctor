"""Column type inference and the per-column schema profile.

This is the schema-level profile stored at upload time. Deeper data-health
checks (outliers, leakage, correlation, ...) belong to the Data Doctor (Phase 2).
"""

import re
import warnings

import numpy as np
import pandas as pd

BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
SAMPLE_SIZE = 2000
_DATE_HINT = re.compile(r"[-/:]|\d{8}")


def looks_like_id_name(name: str) -> bool:
    low = name.strip().lower()
    return (
        low in {"id", "uuid", "guid", "index", "key", "pk"}
        or re.fullmatch(r"unnamed: ?\d+", low) is not None
        or re.search(r"[_\- .](id|uuid|guid|key)$", low) is not None
        or re.search(r"[a-z0-9]Id$", name.strip()) is not None  # camelCase customerId
    )


def _sample(s: pd.Series) -> pd.Series:
    return s.sample(SAMPLE_SIZE, random_state=0) if len(s) > SAMPLE_SIZE else s


def _datetime_ratio(strs: pd.Series) -> float:
    sample = _sample(strs)
    if sample.str.contains(_DATE_HINT).mean() < 0.9:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def infer_type(s: pd.Series) -> tuple[str, bool]:
    """Return (inferred_type, numeric_stored_as_text)."""
    nonnull = s.dropna()
    if nonnull.empty:
        return "empty", False

    n_unique = nonnull.nunique()
    unique_ratio = n_unique / len(nonnull)
    id_name = looks_like_id_name(str(s.name))

    if pd.api.types.is_bool_dtype(s):
        return "boolean", False

    if pd.api.types.is_numeric_dtype(s):
        if n_unique == 2 and set(np.unique(nonnull)) <= {0, 1}:
            return "boolean", False
        is_integral = pd.api.types.is_integer_dtype(s) or bool(np.all(np.mod(nonnull, 1) == 0))
        if is_integral and unique_ratio == 1.0 and len(nonnull) > 10:
            if id_name or nonnull.is_monotonic_increasing:
                return "id", False
        return "numeric", False

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", False

    strs = nonnull.astype(str).str.strip()

    if n_unique <= 6:
        lowered = set(strs.str.lower().unique())
        if len(lowered) == 2 and lowered <= BOOL_TOKENS:
            return "boolean", False

    numeric_ratio = pd.to_numeric(_sample(strs), errors="coerce").notna().mean()
    if numeric_ratio >= 0.95:
        return "numeric", True

    if _datetime_ratio(strs) >= 0.95:
        return "datetime", False

    avg_len = strs.str.len().mean()
    has_spaces = strs.str.contains(" ").mean()
    if avg_len >= 30 and has_spaces >= 0.5 and unique_ratio >= 0.5:
        return "text", False

    if len(nonnull) > 10 and (unique_ratio >= 0.99 or (id_name and unique_ratio >= 0.9)):
        return "id", False

    return "categorical", False


def profile_column(s: pd.Series) -> dict:
    n_rows = len(s)
    n_missing = int(s.isna().sum())
    inferred, numeric_as_text = infer_type(s)
    samples = []
    for v in s.dropna().unique()[:5]:
        text = str(v)
        samples.append(text if len(text) <= 50 else text[:47] + "...")
    return {
        "name": str(s.name),
        "pandas_dtype": str(s.dtype),
        "inferred_type": inferred,
        "n_missing": n_missing,
        "missing_pct": round(100 * n_missing / n_rows, 2) if n_rows else 0.0,
        "n_unique": int(s.nunique(dropna=True)),
        "numeric_stored_as_text": numeric_as_text,
        "sample_values": samples,
    }


def profile_columns(df: pd.DataFrame) -> list[dict]:
    return [profile_column(df[c]) for c in df.columns]
