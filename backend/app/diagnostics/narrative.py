"""Deterministic plain-English summaries built only from computed results."""

from app.diagnostics.findings import INTERVENTIONS, SEVERITY_ORDER


def recommendations(findings: list[dict]) -> list[dict]:
    """Group findings by the intervention that addresses them, most severe first."""
    recs: dict[str, dict] = {}
    for f in findings:
        for key in f["interventions"]:
            rec = recs.setdefault(key, {
                "intervention": key, "label": INTERVENTIONS[key], "severity": f["severity"], "because": [],
            })
            if SEVERITY_ORDER[f["severity"]] < SEVERITY_ORDER[rec["severity"]]:
                rec["severity"] = f["severity"]
            rec["because"].append({"check": f["check"], "title": f["title"]})
    return sorted(recs.values(), key=lambda r: (SEVERITY_ORDER[r["severity"]], -len(r["because"])))


def model_narrative(model_name: str, metrics: dict, findings: list[dict], experiment: dict | None = None) -> list[str]:
    test = metrics["test"]
    lines = [
        f"{model_name} reaches {test['accuracy']:.0%} accuracy and a macro-F1 of {test['f1_macro']:.2f} "
        f"on {test['n']} held-out rows."
    ]
    weak = next((f for f in findings if f["check"] == "model.weak_class"), None)
    if weak:
        c = weak["evidence"]["class"]
        lines.append(f"However, recall for class '{c['label']}' is only {c['recall']:.0%}.")
        if weak.get("cause"):
            lines.append(f"Likely cause: {weak['cause']}")
        if weak.get("impact"):
            lines.append(f"Impact: {weak['impact']}")
    crit = [f for f in findings if f["severity"] == "critical" and f is not weak]
    for f in crit[:2]:
        lines.append(f"Critical: {f['title']}.")
    recs = recommendations(findings)
    if recs:
        lines.append("Recommended next step: " + ", ".join(r["label"].lower() for r in recs[:3]) + ".")
    if experiment and experiment.get("best"):
        b = experiment["best"]
        base_recall = b["metrics"]["minority_recall"] - b["delta_vs_baseline"]["minority_recall"]
        lines.append(
            f"Automated experiment: {b['description']} moved macro-F1 from {experiment['baseline_f1']:.2f} to "
            f"{b['metrics']['f1_macro']:.2f} and minority recall from {base_recall:.0%} to "
            f"{b['metrics']['minority_recall']:.0%}."
        )
        if b.get("version"):
            lines.append(f"Recommendation: promote model v{b['version']} for further validation (requires approval).")
    return lines
