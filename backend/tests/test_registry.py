import pytest

from tests.conftest import API, train


@pytest.fixture
def models(client, churn_dataset):
    job = train(client, churn_dataset["id"], drop_columns=["cancellation_request"], cv_folds=0)
    ids = [r["model_id"] for r in sorted(job["result"]["leaderboard"], key=lambda r: r["name"])]
    return churn_dataset, ids  # [logistic_regression, xgboost]


def _stage(client, model_id, stage, **body):
    return client.post(f"{API}/models/{model_id}/stage", json={"stage": stage, **body})


def test_approval_gate_and_promotion(client, models):
    ds, (lr, xgb) = models
    # Can't skip approval.
    r = _stage(client, lr, "production", note="ship it")
    assert r.status_code == 409 and "Approve it first" in r.json()["error"]["message"]
    # A note is required.
    assert _stage(client, lr, "approved").json()["error"]["code"] == "note_required"
    # Critical findings (weak minority class) block approval unless acknowledged.
    r = _stage(client, lr, "approved", note="looks fine")
    assert r.status_code == 409 and r.json()["error"]["code"] == "unresolved_critical_findings"
    r = _stage(client, lr, "approved", note="accepted for pilot", actor="maya", acknowledge_risks=True)
    assert r.status_code == 200 and r.json()["stage"] == "approved"
    assert _stage(client, lr, "production", note="pilot rollout").json()["stage"] == "production"

    # Promoting another model archives the previous production model.
    _stage(client, xgb, "approved", note="better recall", acknowledge_risks=True)
    _stage(client, xgb, "production", note="replace v1")
    reg = client.get(f"{API}/projects/{ds['project_id']}/registry").json()
    stages = {m["id"]: m["stage"] for m in reg["models"]}
    assert stages == {lr: "archived", xgb: "production"}
    assert reg["production"]["id"] == xgb

    events = client.get(f"{API}/models/{lr}/events").json()
    assert [(e["from_stage"], e["to_stage"]) for e in events] == [
        ("candidate", "approved"), ("approved", "production"), ("production", "archived")]
    assert events[0]["actor"] == "maya" and "acknowledged" in events[0]["note"]
    assert events[2]["note"].startswith("Replaced by")

    # Diagnosis prefers the production model.
    d = client.get(f"{API}/datasets/{ds['id']}/diagnosis").json()
    assert d["model"]["id"] == xgb and d["model"]["stage"] == "production"


def test_invalid_transitions(client, models):
    _, (lr, _) = models
    assert _stage(client, lr, "candidate").status_code == 409  # already candidate
    assert _stage(client, lr, "rejected", note="worse than v2").status_code == 200
    assert _stage(client, lr, "production", note="x").status_code == 409
    assert _stage(client, lr, "candidate").status_code == 200  # reopen
    assert _stage(client, lr, "magic").status_code == 422


def test_compare_models(client, models):
    _, (lr, xgb) = models
    c = client.get(f"{API}/models/compare/{lr}/{xgb}").json()
    assert c["same_test_split"] is True
    metrics = {m["metric"]: m for m in c["metrics"]}
    assert metrics["f1_macro"]["delta"] == pytest.approx(metrics["f1_macro"]["b"] - metrics["f1_macro"]["a"], abs=1e-4)
    assert "recall (1)" in metrics
    assert c["config_diff"]["algorithm"] == {"a": "logistic_regression", "b": "xgboost"}
    assert c["winner"] in {lr, xgb} and "higher macro-F1" in c["summary"][0]


def test_compare_requires_same_dataset(client, project, models, churn_df):
    from tests.conftest import to_csv_bytes, upload

    _, (lr, _) = models
    other = upload(client, project["id"], to_csv_bytes(churn_df)).json()
    client.post(f"{API}/datasets/{other['id']}/target", json={"column": "churned"})
    job = train(client, other["id"], algorithms=["logistic_regression"], cv_folds=0)
    r = client.get(f"{API}/models/compare/{lr}/{job['result']['best_model_id']}")
    assert r.status_code == 400
