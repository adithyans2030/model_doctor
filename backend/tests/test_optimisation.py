import sys
import types

import pytest

from app.diagnostics.findings import finding
from app.ml.optimize import build_plan
from app.ml.pipeline import TrainConfig
from tests.conftest import API, train

LEAK = finding("feature.leakage", "feature", "critical", "leak", interventions=["drop_leaky_features"],
               columns=["flag"])
IMB = finding("data.class_imbalance", "data", "warning", "imb",
              interventions=["class_weight", "smote", "threshold_tuning"])
LOW = finding("feature.low_relevance", "feature", "info", "low", interventions=["feature_selection"],
              columns=["noise"])
CORR = finding("feature.multicollinearity", "feature", "warning", "corr", interventions=["feature_selection"],
               columns=["a", "b"])
CAL = finding("model.poor_calibration", "model", "warning", "cal", interventions=["calibration"])
REPORT = {"feature_target": [{"column": "a", "mutual_info": 0.2}, {"column": "b", "mutual_info": 0.05}]}


def test_plan_orders_interventions_and_finds_columns():
    plan = build_plan([IMB, LEAK, LOW, CORR, CAL], REPORT, TrainConfig("xgboost"), binary=True)
    assert plan.leaky_columns == ["flag"]
    assert plan.feature_drops == ["b", "noise"]  # weaker of the correlated pair (lower MI) + irrelevant
    assert plan.interventions == ["class_weight", "smote", "feature_selection", "calibration", "threshold_tuning"]


def test_plan_respects_task_algorithm_and_restrictions():
    multiclass = build_plan([IMB], REPORT, TrainConfig("xgboost"), binary=False)
    assert "threshold_tuning" not in multiclass.interventions
    svm = build_plan([CAL], REPORT, TrainConfig("svm"), binary=True)
    assert "calibration" not in svm.interventions
    only = build_plan([IMB, LEAK], REPORT, TrainConfig("xgboost"), binary=True, allowed=["class_weight"])
    assert only.interventions == ["class_weight"] and only.leaky_columns == []
    already = build_plan([LEAK], REPORT, TrainConfig("xgboost", drop_columns=["flag"]), binary=True)
    assert already.leaky_columns == []


@pytest.fixture
def leaky_model(client, churn_dataset):
    job = train(client, churn_dataset["id"], algorithms=["logistic_regression"], cv_folds=0)
    return churn_dataset, job["result"]["best_model_id"]


def _experiment(client, model_id, **body):
    r = client.post(f"{API}/models/{model_id}/experiments", json=body)
    assert r.status_code == 202, r.text
    job = client.get(f"{API}/jobs/{r.json()['id']}").json()
    assert job["status"] == "completed", job["error"]
    return job["result"]


def test_experiment_removes_leak_and_fixes_imbalance(client, leaky_model):
    ds, mid = leaky_model
    res = _experiment(client, mid, budget="quick")
    assert res["plan"]["leaky_columns"] == ["cancellation_request"]
    assert "hyperparameter_tuning" not in res["plan"]["interventions"]
    names = [t["name"] for t in res["trials"]]
    assert names[0] == "Baseline (leaking features removed)"
    assert {"Class weighting", "SMOTE oversampling"} <= set(names)
    assert all("cancellation_request" in t["config"]["drop_columns"] for t in res["trials"])
    assert res["trials"][0]["kept"] and sum(t["kept"] for t in res["trials"]) >= 1
    assert "cross-validated" in res["selection"]

    # Best configuration was registered as a new candidate model needing approval.
    best = client.get(f"{API}/models/{res['best_model_id']}").json()
    assert best["source"] == "experiment" and best["stage"] == "candidate"
    assert best["name"].endswith("(optimised)") and best["version"] == 2
    assert res["best"]["model_id"] == best["id"]
    assert res["improvement"]["minority_recall"] > 0  # imbalance handling helps the minority class
    assert "leaking feature" in res["recommendation"] or "Promote" in res["recommendation"]

    d = client.get(f"{API}/datasets/{ds['id']}/diagnosis", params={"model_id": best["id"]}).json()
    assert d["latest_experiment"]["best_model_id"] == best["id"]
    assert any(line.startswith("Automated experiment") for line in d["narrative"])

    listed = client.get(f"{API}/models/{mid}/experiments").json()
    assert len(listed) == 1 and listed[0]["kind"] == "experiment"


def test_standard_budget_runs_hyperparameter_search(client, leaky_model):
    _, mid = leaky_model
    res = _experiment(client, mid, interventions=["drop_leaky_features", "hyperparameter_tuning"])
    assert res["plan"]["interventions"] == ["hyperparameter_tuning"]
    assert [t["intervention"] for t in res["trials"]] == [None, "hyperparameter_tuning"]
    assert res["trials"][1]["config"]["params"]  # tuned params were stored


def test_restricted_experiment_and_validation(client, leaky_model):
    _, mid = leaky_model
    res = _experiment(client, mid, budget="quick", interventions=["class_weight"])
    assert [t["intervention"] for t in res["trials"]] == [None, "class_weight"]
    assert res["plan"]["leaky_columns"] == []  # leak removal wasn't allowed

    bad = client.post(f"{API}/models/{mid}/experiments", json={"interventions": ["magic"]})
    assert bad.status_code == 422
    assert {i["key"] for i in client.get(f"{API}/interventions").json()} >= {"smote", "class_weight"}


def test_mlflow_tracking_is_optional_and_logs_runs(client, churn_dataset, settings, monkeypatch):
    calls = []

    class Run:
        def __init__(self, name):
            self.info = types.SimpleNamespace(run_id=f"run-{name}")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = types.SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=lambda run_name=None, nested=False: (calls.append(("run", run_name)), Run(run_name))[1],
        set_tags=lambda tags: calls.append(("tags", tags)),
        set_tag=lambda k, v: None,
        log_params=lambda p: calls.append(("params", p)),
        log_metrics=lambda m: calls.append(("metrics", m)),
        log_dict=lambda d, name: calls.append(("dict", name)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    settings.mlflow_tracking_uri = "http://mlflow:5000"
    train(client, churn_dataset["id"], algorithms=["logistic_regression"], cv_folds=0)
    assert ("uri", "http://mlflow:5000") in calls
    assert any(c[0] == "metrics" and "f1_macro" in c[1] for c in calls)
    assert ("dict", "diagnosis.json") in calls
