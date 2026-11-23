"""Phase 4: model explanations (classification and regression).

* Global: mean |SHAP| per original column (exact TreeSHAP for XGBoost/LightGBM via
  their built-in ``pred_contribs``; exact linear SHAP for Logistic/Ridge; ``shap``
  TreeExplainer for Random Forest when importable) plus model-agnostic
  permutation importance on the primary metric (macro-F1 / R²).
* Local: signed per-feature contributions for one row. SHAP when supported,
  otherwise *occlusion* (replace a feature with a typical value). **LIME** is
  available for any model on request.

SHAP values are computed on the transformed (one-hot) features and summed back
to the original columns, so explanations talk about ``contract``, not
``cat__contract_Month-to-month``.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score

from app.ml.evaluation import predict_labels
from app.ml.features import FeatureSpec, original_feature
from app.ml.pipeline import underlying_model

SHAP_SAMPLE = 1000
PERM_SAMPLE = 2000
PERM_REPEATS = 3
LIME_SAMPLES = 1000
LIME_KERNEL_WIDTH = 0.75


def _task(bundle: dict) -> str:
    return bundle.get("task") or "classification"


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

    Single-output models (binary classifiers, regressors) return one column; for binary
    classifiers it explains class index 1. ``background`` sets the baseline for linear SHAP.
    """
    pipe = bundle["pipeline"]
    if not hasattr(pipe, "named_steps") or "prep" not in pipe.named_steps:
        return None  # uploaded models: model-agnostic methods only
    algo = bundle["config"]["algorithm"]
    regression = _task(bundle) == "regression"
    log_target = bool(bundle["config"].get("target_transform"))
    spec = FeatureSpec.from_dict(bundle["spec"])
    prep = pipe.named_steps["prep"]
    est = underlying_model(pipe)
    Xt = np.asarray(prep.transform(X[spec.features]), dtype=float)
    n, f = Xt.shape
    units = "log target" if log_target else ("prediction" if regression else "log-odds")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if algo == "xgboost":
                import xgboost as xgb

                contribs = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
                raw = contribs[..., :-1]  # drop bias column
                vals = raw[:, :, None] if raw.ndim == 2 else np.transpose(raw, (0, 2, 1))
                method = f"TreeSHAP (XGBoost, {units})"
            elif algo == "lightgbm":
                contribs = est.predict(Xt, pred_contrib=True)
                k = 1 if (regression or len(getattr(est, "classes_", [0, 1])) == 2) else len(est.classes_)
                raw = np.asarray(contribs).reshape(n, k, f + 1)[:, :, :-1]  # drop bias per output
                vals = np.transpose(raw, (0, 2, 1))
                method = f"TreeSHAP (LightGBM, {units})"
            elif algo in ("logistic_regression", "ridge"):
                # Exact SHAP for a linear model with independent features:
                # phi_j = w_j * (x_j - baseline_j), per output.
                if background is not None:
                    base = np.asarray(prep.transform(background[spec.features]), dtype=float).mean(axis=0)
                else:
                    base = Xt.mean(axis=0)
                coef = np.atleast_2d(np.asarray(est.coef_, dtype=float))  # (1, f) or (k, f)
                vals = (Xt - base)[:, :, None] * coef.T[None, :, :]
                method = f"LinearSHAP ({units})"
            elif algo == "random_forest":
                import shap  # optional: needs a numpy-compatible shap (>= 0.48 with numpy 2.x)

                vals = _to_nfk(shap.TreeExplainer(est).shap_values(Xt, check_additivity=False), n, f)
                method = f"TreeSHAP ({'log target' if log_target else ('prediction' if regression else 'probability')})"
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


def _predict(bundle: dict, X: pd.DataFrame):
    """(labels-or-values, probabilities-or-None)."""
    pipe = bundle["pipeline"]
    if _task(bundle) == "regression":
        return np.asarray(pipe.predict(X), dtype=float), None
    proba = pipe.predict_proba(X)
    return predict_labels(proba, bundle["positive_index"], bundle["config"].get("threshold")), proba


def global_importance(bundle: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    spec = FeatureSpec.from_dict(bundle["spec"])
    regression = _task(bundle) == "regression"
    pos = bundle["positive_index"]
    rng = np.random.default_rng(0)

    shap_part = None
    sample = X.iloc[rng.choice(len(X), size=min(len(X), SHAP_SAMPLE), replace=False)] if len(X) else X
    res = shap_matrix(bundle, sample)
    if res is not None:
        vals, method = res
        if regression:
            per, explained = np.abs(vals[:, :, 0]).mean(axis=0), "prediction"
        elif pos is not None:
            per, explained = np.abs(_class_values(vals, pos, pos)).mean(axis=0), bundle["classes"][pos]
        else:
            per, explained = np.abs(vals).mean(axis=(0, 2)), "all classes (mean)"
        shap_part = {
            "method": method,
            "explained_class": explained,
            "values": sorted(({"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in zip(spec.features, per)),
                             key=lambda d: -d["mean_abs_shap"]),
        }

    # Permutation importance on the primary metric (respects the tuned threshold)
    idx = rng.choice(len(X), size=min(len(X), PERM_SAMPLE), replace=False)
    Xp, yp = X.iloc[idx][spec.features].reset_index(drop=True), y[idx]
    k = len(bundle["classes"])

    def score(frame):
        pred, _ = _predict(bundle, frame)
        if regression:
            return r2_score(yp, pred)
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
    return {"shap": shap_part,
            "permutation": {"metric": "r2" if regression else "f1_macro", "baseline": round(float(base), 4),
                            "values": perm}}


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


def _output(bundle: dict, frame: pd.DataFrame, cls: int | None) -> np.ndarray:
    """The quantity being explained: predicted value (regression) or P(class)."""
    if _task(bundle) == "regression":
        return np.asarray(bundle["pipeline"].predict(frame), dtype=float)
    return bundle["pipeline"].predict_proba(frame)[:, cls]


def lime_contributions(bundle: dict, row: pd.DataFrame, background: pd.DataFrame, cls: int | None,
                       n_samples: int = LIME_SAMPLES, seed: int = 0) -> tuple[np.ndarray, float]:
    """LIME for tabular data, implemented directly.

    Perturb the row by replacing random subsets of its features with baseline values
    (``background`` rows: by default the typical training value per feature), weight each
    perturbation by how close it stays to the original (exponential kernel on the share of
    replaced features), and fit a weighted ridge regression of the model output on
    "feature kept" indicators. The coefficients are the local contributions; the weighted
    R² of that fit is reported as *fidelity* (how well a linear story explains the model
    around this row).
    """
    spec = FeatureSpec.from_dict(bundle["spec"])
    feats = spec.features
    rng = np.random.default_rng(seed)
    d = len(feats)
    z = rng.integers(0, 2, size=(n_samples, d))
    z[0] = 1  # include the unperturbed row
    donors = background[feats].iloc[rng.integers(0, len(background), size=n_samples)].reset_index(drop=True)
    perturbed = pd.concat([row[feats]] * n_samples, ignore_index=True)
    for j, c in enumerate(feats):
        replace = z[:, j] == 0
        perturbed.loc[replace, c] = donors.loc[replace, c].to_numpy()
    for c in spec.numeric_like:
        perturbed[c] = pd.to_numeric(perturbed[c], errors="coerce").astype(float)
    y = _output(bundle, perturbed, cls)
    distance = 1 - z.mean(axis=1)  # share of features replaced
    weights = np.exp(-(distance ** 2) / LIME_KERNEL_WIDTH ** 2)
    surrogate = Ridge(alpha=1.0).fit(z, y, sample_weight=weights)
    fidelity = float(surrogate.score(z, y, sample_weight=weights))
    return surrogate.coef_, fidelity


def explain_row(bundle: dict, row: pd.DataFrame, reference: dict, raw_row: dict | None = None,
                method: str = "auto", background: pd.DataFrame | None = None) -> dict:
    """Explain one prepared row (a single-row DataFrame with the spec's features)."""
    spec = FeatureSpec.from_dict(bundle["spec"])
    regression = _task(bundle) == "regression"
    pipe, pos, classes = bundle["pipeline"], bundle["positive_index"], bundle["classes"]
    if regression:
        value = float(pipe.predict(row[spec.features])[0])
        pred, proba = None, None
    else:
        proba = pipe.predict_proba(row[spec.features])[0]
        pred = int(predict_labels(proba[None, :], pos, bundle["config"].get("threshold"))[0])

    bg = pd.DataFrame([{c: reference.get(c) for c in spec.features}])
    for c in spec.numeric_like:
        bg[c] = pd.to_numeric(bg[c], errors="coerce").astype(float)

    fidelity = None
    res = shap_matrix(bundle, row, bg) if method in ("auto", "shap") else None
    if method == "lime":
        if background is None or background.empty:
            background = bg
        contrib, fidelity = lime_contributions(bundle, row, background, pred)
        used, units = "LIME (local weighted linear surrogate)", "prediction" if regression else "probability"
    elif res is not None:
        vals, used = res
        contrib = vals[0, :, 0] if regression else _class_values(vals, pred, pos)[0]
        units = "log target" if "log target" in used else ("prediction" if regression else
                                                           ("log-odds" if "log-odds" in used else "probability"))
    else:
        used, units = "occlusion (replace with typical value)", "prediction" if regression else "probability"
        contrib = np.zeros(len(spec.features))
        base_out = _output(bundle, row[spec.features], pred)[0]
        for i, c in enumerate(spec.features):
            if reference.get(c) is None:
                continue
            alt = row.copy()
            alt[c] = reference[c]
            contrib[i] = base_out - _output(bundle, alt[spec.features], pred)[0]

    raw_row = raw_row or {}
    items = [
        {"feature": c, "value": _fmt(raw_row.get(c, row.iloc[0][c])), "contribution": round(float(v), 5)}
        for c, v in zip(spec.features, contrib)
    ]
    items.sort(key=lambda d: -abs(d["contribution"]))
    up = [i for i in items if i["contribution"] > 0][:3]
    down = [i for i in items if i["contribution"] < 0][:2]
    if regression:
        text = f"Predicted {value:.4g}."
        if up:
            text += " Pushed up mainly by " + ", ".join(f"{i['feature']} = {i['value']}" for i in up) + "."
        if down:
            text += " Pulled down by " + ", ".join(f"{i['feature']} = {i['value']}" for i in down) + "."
        out = {"prediction": round(value, 6), "confidence": None, "probabilities": None}
    else:
        label = classes[pred]
        text = f"Predicted '{label}' with {proba[pred]:.0%} confidence."
        if up:
            text += " Mainly driven by " + ", ".join(f"{i['feature']} = {i['value']}" for i in up) + "."
        if down:
            text += " Pulling the other way: " + ", ".join(f"{i['feature']} = {i['value']}" for i in down) + "."
        out = {"prediction": label, "confidence": round(float(proba[pred]), 4),
               "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)}}
    return {**out, "method": used, "units": units, "fidelity": None if fidelity is None else round(fidelity, 4),
            "contributions": items, "explanation": text}
