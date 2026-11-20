"""The Model Health Score: a weighted summary of transparent sub-scores.

It summarises the findings. It does not replace the underlying metrics; every
component reports how it was computed.
"""

WEIGHTS = {"data": 0.25, "performance": 0.30, "robustness": 0.20, "fairness": 0.15, "drift": 0.10}


def _clip(x: float) -> int:
    return int(round(max(0.0, min(100.0, x))))


def compute_health(
    data_score: int | None,
    metrics: dict | None,
    fairness_gap: float | None = None,
    drift_score: int | None = None,
) -> dict:
    comps: dict[str, dict] = {}
    comps["data"] = {
        "value": data_score,
        "explanation": "Data Doctor score: 100 minus penalties for data/feature findings.",
    }

    if metrics:
        test, cv = metrics["test"], metrics.get("cv")
        comps["performance"] = {
            "value": _clip(100 * test["f1_macro"]),
            "explanation": f"Macro-F1 on the held-out test set ({test['f1_macro']:.3f}).",
        }
        if cv:
            gap = max(0.0, cv["train"]["f1_macro"]["mean"] - cv["val"]["f1_macro"]["mean"])
            std = cv["val"]["f1_macro"]["std"]
        else:
            gap = max(0.0, metrics["train"]["f1_macro"] - test["f1_macro"])
            std = 0.0
        ece = test["ece"]
        comps["robustness"] = {
            "value": _clip(100 - 150 * gap - 300 * std - 100 * ece),
            "explanation": (f"100 − 150×overfit gap ({gap:.3f}) − 300×CV std ({std:.3f}) "
                            f"− 100×calibration error ({ece:.3f})."),
        }
    else:
        comps["performance"] = {"value": None, "explanation": "No trained model yet."}
        comps["robustness"] = {"value": None, "explanation": "No trained model yet."}

    comps["fairness"] = (
        {"value": _clip(100 - 150 * fairness_gap),
         "explanation": f"100 − 150×largest balanced-accuracy gap between groups ({fairness_gap:.3f})."}
        if fairness_gap is not None else
        {"value": None, "explanation": "Computed with the model's error analysis (explanations)."}
    )
    comps["drift"] = (
        {"value": drift_score, "explanation": "From the latest drift report for this model."}
        if drift_score is not None else
        {"value": None, "explanation": "Not measured yet: upload a production batch to monitor drift."}
    )

    available = {k: v["value"] for k, v in comps.items() if v["value"] is not None}
    if available:
        total_w = sum(WEIGHTS[k] for k in available)
        overall = _clip(sum(WEIGHTS[k] * v for k, v in available.items()) / total_w)
    else:
        overall = None
    return {
        "overall": overall,
        "components": comps,
        "weights": WEIGHTS,
        "note": "Weights are renormalised over the components that have been measured.",
    }
