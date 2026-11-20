"""Classification metrics, calibration and threshold analysis."""

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def predict_labels(proba: np.ndarray, positive_index: int | None, threshold: float | None) -> np.ndarray:
    if threshold is None or positive_index is None:
        return proba.argmax(axis=1)
    other = 1 - positive_index
    return np.where(proba[:, positive_index] >= threshold, positive_index, other)


def _r(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, positive_index: int | None, bins: int = 10):
    """ECE plus reliability-curve points.

    Binary: calibration of P(positive). Multiclass: confidence (max prob) vs accuracy.
    """
    if positive_index is not None:
        conf = proba[:, positive_index]
        hit = (y == positive_index).astype(float)
    else:
        conf = proba.max(axis=1)
        hit = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if not mask.any():
            continue
        mean_conf, frac = conf[mask].mean(), hit[mask].mean()
        ece += mask.mean() * abs(mean_conf - frac)
        curve.append({"bin": [round(lo, 2), round(hi, 2)], "mean_predicted": _r(mean_conf),
                      "observed": _r(frac), "count": int(mask.sum())})
    return round(float(ece), 4), curve


def threshold_analysis(y: np.ndarray, proba: np.ndarray, positive_index: int) -> dict:
    pos = proba[:, positive_index]
    truth = y == positive_index
    rows = []
    for t in THRESHOLDS:
        pred = pos >= t
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "precision": _r(precision), "recall": _r(recall), "f1": _r(f1)})
    best = max(rows, key=lambda r: (r["f1"], -abs(r["threshold"] - 0.5)))
    at_default = next(r for r in rows if r["threshold"] == 0.5)
    return {"curve": rows, "best_threshold": best["threshold"], "best_f1": best["f1"], "f1_at_0_5": at_default["f1"]}


def best_threshold(y: np.ndarray, proba: np.ndarray, positive_index: int) -> float:
    return threshold_analysis(y, proba, positive_index)["best_threshold"]


def classification_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    positive_index: int | None,
    threshold: float | None = None,
) -> dict:
    k = len(classes)
    labels = list(range(k))
    pred = predict_labels(proba, positive_index, threshold)

    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = [
        {"label": classes[i], "precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]), "support": int(support[i])}
        for i in labels
    ]

    roc = pr = brier = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if positive_index is not None:
                truth = (y == positive_index).astype(int)
                roc = roc_auc_score(truth, proba[:, positive_index])
                pr = average_precision_score(truth, proba[:, positive_index])
                brier = brier_score_loss(truth, proba[:, positive_index])
            else:
                roc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            pass
        ll = log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=labels)

    ece, curve = expected_calibration_error(y, proba, positive_index)
    worst = min(per_class, key=lambda c: (c["f1"], c["support"]))
    metrics = {
        "n": int(len(y)),
        "accuracy": _r(accuracy_score(y, pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y, pred)),
        "f1_macro": _r(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": _r(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        "precision_macro": _r(float(np.mean(p))),
        "recall_macro": _r(float(np.mean(r))),
        "roc_auc": _r(roc),
        "pr_auc": _r(pr),
        "log_loss": _r(ll),
        "brier": _r(brier),
        "ece": ece,
        "calibration_curve": curve,
        "per_class": per_class,
        "worst_class": worst,
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
        "threshold": threshold,
    }
    if positive_index is not None:
        pc = per_class[positive_index]
        metrics["positive_class"] = pc
        metrics["threshold_analysis"] = threshold_analysis(y, proba, positive_index)
    return metrics


def quick_scores(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    labels = list(range(k))
    return {
        "accuracy": _r(accuracy_score(y, pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y, pred)),
        "f1_macro": _r(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
    }
