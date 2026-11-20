"""Deterministic fallback assistant: answers common questions straight from the context.

Used when no LLM is configured, when the LLM call fails, or when it declines.
Every sentence is assembled from numbers already in the context.
"""

import re


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _find(ctx, *checks):
    return [f for f in ctx["findings"] if f["check"] in checks and not f.get("resolved")]


def _bullets(items):
    return "\n".join(f"- {i}" for i in items)


def _sources(fs):
    return [f["check"] for f in fs]


def answer_recall(ctx):
    weak = _find(ctx, "model.weak_class")
    imb = _find(ctx, "data.class_imbalance")
    lines, used = [], weak + imb
    if weak:
        c = weak[0]["evidence"]
        lines.append(weak[0]["title"] + ".")
        if weak[0].get("cause"):
            lines.append("Likely cause: " + weak[0]["cause"])
    elif ctx.get("selected_model"):
        m = next((m for m in ctx["models"] if m["id"] == ctx["selected_model"]["id"]), None)
        if m:
            mc = m["minority_class"]
            lines.append(f"Recall for class '{mc['label']}' is {_pct(mc['recall'])}; no weak-class finding was raised.")
    if imb and not weak:
        lines.append(imb[0]["title"] + f" (ratio {imb[0]['evidence'].get('imbalance_ratio')}).")
    stmts = (ctx.get("explanations") or {}).get("error_statements") or []
    if stmts:
        lines.append("Where the misses happen: " + stmts[0])
    if ctx["experiments"] and ctx["experiments"][0].get("improvement"):
        e = ctx["experiments"][0]
        lines.append(f"The latest experiment ({e['best']}) changed minority recall by "
                     f"{e['improvement']['minority_recall']:+.0%}.")
    elif "class_weight" in {r["intervention"] for r in ctx["recommendations"]}:
        lines.append("Try class-weighted training, SMOTE and threshold tuning (run an experiment to measure them).")
    return " ".join(lines) or "No recall problem has been diagnosed for this model.", _sources(used)


def answer_priorities(ctx):
    recs = ctx["recommendations"][:4]
    crit = [f for f in ctx["findings"] if f["severity"] == "critical" and not f.get("resolved")][:3]
    if not recs and not crit:
        return "No outstanding issues need fixing. Keep monitoring for drift.", []
    parts = []
    if crit:
        parts.append("Critical issues first:\n" + _bullets(f"{f['title']} [{f['check']}]" for f in crit))
    if recs:
        parts.append("Recommended fixes, most severe first:\n" + _bullets(
            f"{r['label']} ({r['severity']}; because of {', '.join(r['because'][:3])})" for r in recs))
    return "\n\n".join(parts), _sources(crit) + [c for r in recs for c in r["because"]]


def answer_health(ctx):
    h = ctx["health"]
    comps = ", ".join(f"{k} {v}" if v is not None else f"{k} not measured" for k, v in h["components"].items())
    return f"Model health is {h['overall']}/100 ({comps}).", []


def answer_overfit(ctx):
    f = _find(ctx, "model.overfitting", "model.high_variance", "model.underfitting")
    if not f:
        return "No over- or underfitting was detected for the selected model.", []
    return " ".join(x["title"] + "." for x in f) + " " + (f[0].get("recommendation") or ""), _sources(f)


def answer_leak(ctx):
    f = [x for x in ctx["findings"] if x["check"] in {"feature.leakage", "model.suspicious_performance", "model.dominant_feature"}]
    if not f:
        return "No target leakage was detected.", []
    return " ".join(x["title"] + ("" if not x.get("resolved") else "") + "." for x in f), _sources(f)


def answer_drift(ctx):
    d = ctx.get("drift")
    if not d:
        return "No drift report yet: upload a production batch to the model's monitor endpoint.", []
    top = ", ".join(f"{t['feature']} (PSI {t['psi']:.2f}, {t['severity']})" for t in d["top"][:3])
    text = f"Drift score {d['score']}/100 with {d['drifted_features']} drifted feature(s). Largest shifts: {top}."
    if d.get("performance"):
        p = d["performance"]
        text += f" Macro-F1 on the labelled batch: {p['batch_f1_macro']:.2f} vs {p['test_f1_macro']:.2f} on test."
    return text, _sources(_find(ctx, "distribution.feature_drift", "distribution.concept_drift"))


def answer_experiment(ctx):
    if not ctx["experiments"]:
        return "No experiments have run yet. Start one from the model page to test the recommended fixes.", []
    e = ctx["experiments"][0]
    kept = [t["name"] for t in e["trials"] if t["kept"]]
    return (f"Best configuration: {e['best']}. Kept steps: {', '.join(kept)}. "
            f"Improvement vs baseline: macro-F1 {e['improvement']['f1_macro']:+.3f}, minority recall "
            f"{e['improvement']['minority_recall']:+.0%}. {e['recommendation']}"), []


def answer_features(ctx):
    e = ctx.get("explanations")
    if not e:
        return "Explanations haven't been computed for this model yet (open its Explainability tab).", []
    return (f"Most influential features (SHAP): {', '.join(e['top_features_shap']) or 'n/a'}. "
            f"By permutation importance on macro-F1: {', '.join(e['top_features_permutation'])}."), []


def answer_compare(ctx, a: dict, b: dict):
    rows = []
    for k in ("f1_macro", "balanced_accuracy", "accuracy", "roc_auc"):
        va, vb = a["test"].get(k), b["test"].get(k)
        if va is not None and vb is not None:
            rows.append(f"{k}: v{a['version']} {va:.3f} → v{b['version']} {vb:.3f} ({vb - va:+.3f})")
    ra, rb = a["minority_class"]["recall"], b["minority_class"]["recall"]
    rows.append(f"minority recall: {_pct(ra)} → {_pct(rb)}")
    resolved = sorted(set(a["checks"]) - set(b["checks"]))
    new = sorted(set(b["checks"]) - set(a["checks"]))
    if resolved:
        rows.append("resolved in v%d: %s" % (b["version"], ", ".join(resolved)))
    if new:
        rows.append("new in v%d: %s" % (b["version"], ", ".join(new)))
    better = b if b["test"]["f1_macro"] > a["test"]["f1_macro"] else a
    return f"{a['name']} vs {b['name']}:\n" + _bullets(rows) + f"\nHigher macro-F1: {better['name']}.", []


def answer_summary(ctx):
    top = [f for f in ctx["findings"] if not f.get("resolved")][:3]
    text = " ".join(ctx["narrative"])
    if top:
        text += "\n\nTop findings:\n" + _bullets(f"{f['severity']}: {f['title']}" for f in top)
    return text, _sources(top)


INTENTS = [
    # Patterns match word *prefixes* ("feature" matches "features", "driv" matches "drives").
    (r"\b(recall|precision|false negative|miss|minority)", answer_recall),
    (r"\b(fix|first|priorit|should i|next step|recommend|improve)", answer_priorities),
    (r"\b(drift|shift|production)", answer_drift),
    (r"\b(overfit|underfit|variance|generali[sz])", answer_overfit),
    (r"\b(leak|too good|suspicious)", answer_leak),
    (r"\b(experiment|optimi[sz]|smote|class weight|tuning|trial)", answer_experiment),
    (r"\b(feature|important|driv|shap)", answer_features),
    (r"\b(health|score)", answer_health),
]


def rule_answer(question: str, ctx: dict) -> tuple[str, list[str]]:
    q = question.lower()
    versions = [int(v) for v in re.findall(r"(?:\bv|version\s*|model\s*)(\d+)", q)]
    if ("compare" in q or " vs" in q or "versus" in q) and len(versions) >= 2:
        by_v = {m["version"]: m for m in ctx["models"]}
        a, b = by_v.get(versions[0]), by_v.get(versions[1])
        if a and b:
            return answer_compare(ctx, a, b)
        return f"I couldn't find model versions {versions[0]} and {versions[1]} for this dataset.", []
    for pattern, fn in INTENTS:
        if re.search(pattern, q):
            return fn(ctx)
    return answer_summary(ctx)
