"""Phase 4: model explanations.

* Global: mean |SHAP| per original column (exact TreeSHAP for XGBoost via
  ``pred_contribs``; ``shap`` TreeExplainer / LinearExplainer for Random Forest
  and Logistic Regression) plus model-agnostic permutation importance measured
  on macro-F1 with the model's own decision threshold.
* Local: signed per-feature contributions for one row. Uses SHAP when the model
  supports it, otherwise *occlusion*: replace each feature with a typical value
  (train median / mode) and measure how much the predicted-class probability moves.

SHAP values are computed on the transformed (one-hot) features and summed back
to the original columns, so explanations talk about ``contract``, not
``cat__contract_Month-to-month``.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from app.ml.evaluation import predict_labels
from app.ml.features import FeatureSpec, original_feature
from app.ml.pipeline import underlying_model

SHAP_SAMPLE = 1000
PERM_SAMPLE = 2000
PERM_REPEATS = 3


def _aggregation_matrix(prep, features: list[str]) -> np.ndarray:
    names = prep.get_feature_names_out()
    m = np.zeros((len(names), len(features)))
    index = {f: i for i, f in enumerate(features)}
    for j, n in enumerate(names):
        m[j, index[original_feature(n, features)]] = 1.0
    return m


def _to_nfk(values, n: int, f: int) -> np.ndarray:
    """Normalise SHAP outputs (list per class, (n,f), (n,f,k)) to (n, f, k)."""
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] == n and arr.shape[1] == f:
        return arr
    raise ValueError(f"unexpected SHAP shape {arr.shape}")


def shap_matrix(bundle: dict, X: pd.DataFrame, background: pd.DataFrame | None = None) -> tuple[np.ndarray, str] | None:
    """SHAP values aggregated to original columns: array (n, n_features, k_or_1), method name.

    For binary models with a single output column, values are for class index 1.
    ``background`` sets the baseline for linear SHAP (defaults to the mean of ``X``).
    """
    pipe = bundle["pipeline"]
    algo = bundle["config"]["algorithm"]
    spec = FeatureSpec.from_dict(bundle["spec"])
    prep = pipe.named_steps["prep"]
    est = underlying_model(pipe)
    Xt = np.asarray(prep.transform(X[spec.features]), dtype=float)
    n, f = Xt.shape
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algo == "xgboost":
                import xgboost as xgb

                contribs = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
                raw = contribs[..., :-1]  # drop bias column
                vals = raw[:, :, None] if raw.ndim == 2 else np.transpose(raw, (0, 2, 1))
                method = "TreeSHAP (XGBoost, log-odds)"
            elif algo == "logistic_regression":
                # Exact SHAP for a linear model with independent features:
                # phi_j = w_j * (x_j - baseline_j), per output (class logit).
                if background is not None:
                    base = np.asarray(prep.transform(background[spec.features]), dtype=float).mean(axis=0)
                else:
                    base = Xt.mean(axis=0)
                coef = np.asarray(est.coef_, dtype=float)  # (1, f) binary or (k, f)
                vals = (Xt - base)[:, :, None] * coef.T[None, :, :]
                method = "LinearSHAP (log-odds)"
            elif algo == "random_forest":
                import shap  # optional: needs a numpy-compatible shap (>= 0.48 with numpy 2.x)

                vals = _to_nfk(shap.TreeExplainer(est).shap_values(Xt, check_additivity=False), n, f)
                method = "TreeSHAP (probability)"
            else:
                return None
    except Exception:  # noqa: BLE001 - fall back to model-agnostic methods
        return None
    agg = _aggregation_matrix(prep, spec.features)
    return np.einsum("nfk,fo->nok", vals, agg), method


def _class_values(vals: np.ndarray, class_index: np.ndarray | int, positive_index: int | None) -> np.ndarray:
    """Pick, per row, contributions towards ``class_index``. Single-output binary models
    explain class 1, so the sign is flipped for class 0."""
    n = vals.shape[0]
    idx = np.broadcast_to(np.asarray(class_index), (n,))
    if vals.shape[2] == 1:
        sign = np.where(idx == 1, 1.0, -1.0)
        return vals[:, :, 0] * sign[:, None]
    return vals[np.arange(n), :, idx]


def global_importance(bundle: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    spec = FeatureSpec.from_dict(bundle["spec"])
    pipe, pos = bundle["pipeline"], bundle["positive_index"]
    threshold = bundle["config"].get("threshold")
    rng = np.random.default_rng(0)

    # SHAP
    shap_part = None
    sample = X.iloc[rng.choice(len(X), size=min(len(X), SHAP_SAMPLE), replace=False)] if len(X) else X
    res = shap_matrix(bundle, sample)
    if res is not None:
        vals, method = res
        if pos is not None:
            per = np.abs(_class_values(vals, pos, pos)).mean(axis=0)
        else:
            per = np.abs(vals).mean(axis=(0, 2))
        shap_part = {
            "method": method,
            "explained_class": bundle["classes"][pos] if pos is not None else "all classes (mean)",
            "values": sorted(({"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in zip(spec.features, per)),
                             key=lambda d: -d["mean_abs_shap"]),
        }

    # Permutation importance on macro-F1 (respects the tuned threshold)
    idx = rng.choice(len(X), size=min(len(X), PERM_SAMPLE), replace=False)
    Xp, yp = X.iloc[idx][spec.features].reset_index(drop=True), y[idx]
    k = len(bundle["classes"])

    def score(frame):
        pred = predict_labels(pipe.predict_proba(frame), pos, threshold)
        return f1_score(yp, pred, average="macro", labels=list(range(k)), zero_division=0)

    base = score(Xp)
    perm = []
    for col in spec.features:
        drops = []
        for r in range(PERM_REPEATS):
            shuffled = Xp.copy()
            shuffled[col] = Xp[col].sample(frac=1, random_state=r).to_numpy()
            drops.append(base - score(shuffled))
        perm.append({"feature": col, "importance": round(float(np.mean(drops)), 5),
                     "std": round(float(np.std(drops)), 5)})
    perm.sort(key=lambda d: -d["importance"])
    return {"shap": shap_part, "permutation": {"metric": "f1_macro", "baseline": round(float(base), 4), "values": perm}}


def reference_values(X_train: pd.DataFrame, spec: FeatureSpec) -> dict:
    ref = {}
    for c in spec.features:
        s = X_train[c].dropna()
        if s.empty:
            ref[c] = None
        elif c in spec.categorical:
            ref[c] = s.mode().iloc[0]
        else:
            ref[c] = float(s.median())
    return ref


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def explain_row(bundle: dict, row: pd.DataFrame, reference: dict, raw_row: dict | None = None) -> dict:
    """Explain one prepared row (a single-row DataFrame with the spec's features)."""
    spec = FeatureSpec.from_dict(bundle["spec"])
    pipe, pos, classes = bundle["pipeline"], bundle["positive_index"], bundle["classes"]
    threshold = bundle["config"].get("threshold")
    proba = pipe.predict_proba(row[spec.features])[0]
    pred = int(predict_labels(proba[None, :], pos, threshold)[0])

    background = pd.DataFrame([{c: reference.get(c) for c in spec.features}])
    for c in spec.numeric_like:
        background[c] = pd.to_numeric(background[c], errors="coerce").astype(float)
    res = shap_matrix(bundle, row, background)
    if res is not None:
        vals, method = res
        contrib = _class_values(vals, pred, pos)[0]
        units = "log-odds" if "log-odds" in method else "probability"
    else:
        method, units = "occlusion (replace with typical value)", "probability"
        contrib = np.zeros(len(spec.features))
        for i, c in enumerate(spec.features):
            if reference.get(c) is None:
                continue
            alt = row.copy()
            alt[c] = reference[c]
            contrib[i] = proba[pred] - pipe.predict_proba(alt[spec.features])[0][pred]

    raw_row = raw_row or {}
    items = [
        {"feature": c, "value": _fmt(raw_row.get(c, row.iloc[0][c])), "contribution": round(float(v), 5)}
        for c, v in zip(spec.features, contrib)
    ]
    items.sort(key=lambda d: -abs(d["contribution"]))
    label = classes[pred]
    up = [i for i in items if i["contribution"] > 0][:3]
    down = [i for i in items if i["contribution"] < 0][:2]
    text = f"Predicted '{label}' with {proba[pred]:.0%} confidence."
    if up:
        text += " Mainly driven by " + ", ".join(f"{i['feature']} = {i['value']}" for i in up) + "."
    if down:
        text += " Pulling the other way: " + ", ".join(f"{i['feature']} = {i['value']}" for i in down) + "."
    return {
        "prediction": label,
        "confidence": round(float(proba[pred]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "method": method,
        "units": units,
        "contributions": items,
        "explanation": text,
    }
