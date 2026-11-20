import numpy as np
import pytest

from app.diagnostics.findings import finding, score_from_findings
from tests.conftest import API, make_df, to_csv_bytes, upload


def _report(client, dataset_id, **params):
    r = client.get(f"{API}/datasets/{dataset_id}/data-report", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _checks(report) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in report["findings"]:
        out.setdefault(f["check"], []).append(f)
    return out


def _upload_with_target(client, project, df, target):
    ds = upload(client, project["id"], to_csv_bytes(df)).json()
    r = client.post(f"{API}/datasets/{ds['id']}/target", json={"column": target})
    assert r.status_code == 200, r.text
    return ds["id"]


def test_detects_every_planted_issue_in_demo_data(client, churn_dataset):
    report = _report(client, churn_dataset["id"])
    checks = _checks(report)

    assert checks["data.class_imbalance"][0]["severity"] in {"warning", "critical"}
    assert checks["data.class_imbalance"][0]["evidence"]["minority_class"] == "1"
    assert {f["columns"][0] for f in checks["data.missing_values"]} >= {"age", "total_charges", "payment_method"}
    assert checks["data.duplicate_rows"][0]["evidence"]["n_duplicates"] > 0
    assert checks["data.outliers"][0]["columns"] == ["monthly_charges"]
    assert checks["data.constant_column"][0]["columns"] == ["country"]
    assert checks["data.id_column"][0]["columns"] == ["customer_id"]
    assert ("tenure_months", "total_charges") in {
        tuple(f["columns"]) for f in checks["feature.multicollinearity"]
    }
    leak = checks["feature.leakage"]
    assert [f["columns"][0] for f in leak] == ["cancellation_request"]
    assert "drop_leaky_features" in leak[0]["interventions"]
    assert "has_partner" in checks["feature.low_relevance"][0]["columns"]


def test_report_structure_and_score(client, churn_dataset):
    report = _report(client, churn_dataset["id"])
    s = report["summary"]
    assert s["n_rows"] == 2020 and s["target"] == "churned"
    assert s["excluded_columns"] == {"customer_id": "id column", "country": "constant column"}
    assert s["n_features_usable"] == 11
    assert s["severity_counts"]["critical"] >= 1

    # Score = 100 - sum(penalties), and findings are sorted most severe first.
    assert report["score"]["value"] == 100 - sum(p["points"] for p in report["score"]["penalties"])
    order = {"critical": 0, "warning": 1, "info": 2}
    sev = [order[f["severity"]] for f in report["findings"]]
    assert sev == sorted(sev)

    stats = {c["name"]: c for c in report["column_stats"]}
    assert stats["monthly_charges"]["n_outliers"] > 0
    assert len(stats["monthly_charges"]["histogram"]["counts"]) == 20
    assert stats["region"]["top_values"][0]["value"] in {"North", "South", "East", "West"}
    assert report["feature_target"][0]["column"] == "cancellation_request"


def test_report_is_cached_and_recomputed_when_target_changes(client, churn_dataset):
    first = _report(client, churn_dataset["id"])
    assert _report(client, churn_dataset["id"])["id"] == first["id"]
    assert _report(client, churn_dataset["id"], refresh=True)["id"] != first["id"]

    client.post(f"{API}/datasets/{churn_dataset['id']}/target", json={"column": "contract"})
    changed = _report(client, churn_dataset["id"])
    assert changed["summary"]["target"] == "contract"
    assert changed["summary"]["task_type"] == "multiclass_classification"


def test_no_target_report(client, project):
    ds = upload(client, project["id"], to_csv_bytes(make_df())).json()
    report = _report(client, ds["id"])
    assert "data.no_target" in _checks(report)
    assert report["feature_target"] == []


def test_column_level_checks(client, project):
    df = make_df(n=200)
    rng = np.random.default_rng(1)
    df["mostly_missing"] = np.where(rng.random(200) < 0.6, np.nan, rng.random(200))
    df["sku"] = [f"SKU-{i % 120}" for i in range(200)]
    df["price_text"] = [str(round(v, 2)) for v in rng.random(200)]
    df.loc[[3, 7], "price_text"] = "unknown"
    df["flag"] = ["x"] * 199 + ["y"]
    report = _report(client, _upload_with_target(client, project, df, "label"))
    checks = _checks(report)

    missing = {f["columns"][0]: f["severity"] for f in checks["data.missing_values"]}
    assert missing["mostly_missing"] == "critical"
    assert checks["data.high_cardinality"][0]["columns"] == ["sku"]
    invalid = checks["data.invalid_values"][0]
    assert invalid["evidence"]["n_invalid"] == 2 and invalid["evidence"]["examples"] == ["unknown"]
    assert checks["data.near_constant"][0]["columns"] == ["flag"]
    assert checks["data.text_column"][0]["columns"] == ["note"]
    assert checks["data.datetime_column"][0]["columns"] == ["signup"]
    assert "data.small_dataset" in checks


def test_conflicting_labels_and_multiclass_imbalance(client, project):
    df = make_df(n=300).drop(columns=["customer_id", "note", "signup", "income", "age"])
    df["label"] = ["a"] * 280 + ["b"] * 14 + ["c"] * 6
    report = _report(client, _upload_with_target(client, project, df, "label"))
    checks = _checks(report)
    assert checks["data.class_imbalance"][0]["severity"] == "critical"  # 280 / 6 > 20
    assert checks["data.conflicting_labels"][0]["evidence"]["n_groups"] > 0


def test_score_caps_repeated_checks():
    many = [finding("data.missing_values", "data", "warning", f"c{i}") for i in range(10)]
    score = score_from_findings(many + [finding("data.class_imbalance", "data", "critical", "x")])
    assert score["value"] == 100 - 25 - 20


def test_finding_rejects_unknown_intervention():
    with pytest.raises(AssertionError):
        finding("x", "data", "info", "t", interventions=["magic"])
