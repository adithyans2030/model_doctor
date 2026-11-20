"""The common structure every diagnostic check produces.

A *finding* is a plain dict so it can be stored as JSON, returned by the API,
rendered by the dashboard and handed to the AI assistant without conversion:

    {
      "check": "data.class_imbalance",       # stable id, used for linking and tests
      "category": "data",                     # data | feature | model | prediction | fairness | distribution
      "severity": "critical",                 # critical | warning | info
      "title": "Significant class imbalance",
      "evidence": {...},                      # the actual numbers behind the claim
      "cause": "...", "impact": "...", "recommendation": "...",
      "interventions": ["class_weight", "smote"],   # what the optimiser can try
      "columns": ["churned"],
    }
"""

from typing import Any

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
PENALTY = {"critical": 20, "warning": 8, "info": 2}

# Interventions the optimisation engine (Phase 5) knows how to run.
INTERVENTIONS = {
    "class_weight": "Class-weighted training",
    "smote": "SMOTE oversampling",
    "threshold_tuning": "Decision-threshold tuning",
    "drop_leaky_features": "Remove leaking features",
    "drop_duplicates": "Remove duplicate rows",
    "feature_selection": "Feature selection",
    "hyperparameter_tuning": "Hyperparameter search",
    "calibration": "Probability calibration",
}


def finding(
    check: str,
    category: str,
    severity: str,
    title: str,
    evidence: dict[str, Any] | None = None,
    *,
    cause: str | None = None,
    impact: str | None = None,
    recommendation: str | None = None,
    interventions: list[str] | tuple[str, ...] = (),
    columns: list[str] | tuple[str, ...] = (),
) -> dict:
    assert severity in SEVERITY_ORDER, severity
    unknown = set(interventions) - INTERVENTIONS.keys()
    assert not unknown, unknown
    return {
        "check": check,
        "category": category,
        "severity": severity,
        "title": title,
        "evidence": evidence or {},
        "cause": cause,
        "impact": impact,
        "recommendation": recommendation,
        "interventions": list(interventions),
        "columns": list(columns),
    }


def sort_findings(findings: list[dict]) -> list[dict]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f["severity"]], f["check"]))


def score_from_findings(findings: list[dict], *, max_per_check: int = 25) -> dict:
    """Transparent score: start at 100 and subtract a fixed penalty per finding.

    Penalties for repeated findings of the same check (e.g. many columns with a
    few missing values) are capped so one noisy check can't dominate the score.
    """
    by_check: dict[str, int] = {}
    penalties = []
    for f in findings:
        pts = PENALTY[f["severity"]]
        used = by_check.get(f["check"], 0)
        pts = max(0, min(pts, max_per_check - used))
        by_check[f["check"]] = used + pts
        if pts:
            penalties.append({"check": f["check"], "severity": f["severity"], "points": pts, "title": f["title"]})
    value = max(0, 100 - sum(p["points"] for p in penalties))
    return {"value": value, "penalties": penalties}


def summarize_counts(findings: list[dict]) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f["severity"]] += 1
    return counts
