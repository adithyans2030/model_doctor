"""Regression support end to end, on the house-prices demo (skewed target + a planted leak)."""

import numpy as np
import pytest

from app.ml.algorithms import ALGORITHMS
from tests.conftest import API, to_csv_bytes, train, upload


@pytest.fixture(scope="session")
def house_df():
    from scripts.make_demo_data import make_house_prices

    return make_house_prices(rows=800, seed=11)


@pytest.fixture
def houses(client, project, house_df):
    ds = upload(client, project["id"], to_csv_bytes(house_df), filename="houses.csv").json()
    r = client.post(f"{API}/datasets/{ds['id']}/target", json={"column": "price"})
    assert r.status_code == 200, r.text
    return r.json()


def _checks(findings):
    return {f["check"]: f for f in findings}


def test_target_and_data_doctor(client, houses):
    ds = houses["dataset"]
    assert ds["task_type"] == "regression" and ds["target_summary"]["skew"] > 1
    report = client.get(f"{API}/datasets/{ds['id']}/data-report").json()
    checks = _checks(report["findings"])
    assert "data.class_imbalance" not in checks
    assert checks["data.skewed_target"]["interventions"] == ["target_transform"]
    leak = checks["feature.leakage"]
    assert leak["columns"] == ["assessed_value"] and leak["severity"] == "critical"
    assert report["feature_target"][0]["metric"] == "r2"


def test_algorithms_endpoint_and_validation(client, houses):
    algos = {a["key"]: a for a in client.get(f"{API}/algorithms", params={"task": "regression"}).json()}
    assert {"ridge", "random_forest", "xgboost", "lightgbm", "svm"} <= {k for k, a in algos.items() if a["default"]}
    assert "logistic_regression" not in algos
    r = client.post(f"{API}/datasets/{houses['dataset']['id']}/train", json={"algorithms": ["logistic_regression"]})
    assert r.status_code == 400 and "Not available for regression" in r.json()["error"]["message"]


@pytest.fixture
def trained_houses(client, houses):
    job = train(client, houses["dataset"]["id"], algorithms=["ridge", "lightgbm"], cv_folds=3)
    return houses["dataset"], job


def test_training_metrics_and_diagnosis(client, trained_houses):
    ds, job = trained_houses
    res = job["result"]
    assert res["task"] == "regression" and res["primary_metric"] == "r2"
    assert [r["r2"] for r in res["leaderboard"]] == sorted((r["r2"] for r in res["leaderboard"]), reverse=True)
    best = client.get(f"{API}/models/{res['best_model_id']}").json()
    t = best["metrics"]["test"]
    assert {"r2", "rmse", "mae", "residual_histogram", "points"} <= set(t)
    assert best["scores"]["primary"] == t["r2"] and best["scores"]["worst_class"] is None
    # The leak makes the fit too good; the doctor ties it to the leakage finding.
    sus = _checks(best["findings"]).get("model.suspicious_performance") or _checks(best["findings"])["model.dominant_feature"]
    assert sus["severity"] == "critical" and sus["interventions"] == ["drop_leaky_features"]

    d = client.get(f"{API}/datasets/{ds['id']}/diagnosis").json()
    assert "R²" in d["health"]["components"]["performance"]["explanation"]
    assert d["narrative"][0].startswith(best["name"]) and "variance" in d["narrative"][0]


def test_explanations_shap_lime_and_errors(client, trained_houses):
    ds, job = trained_houses
    ids = {r["algorithm"]: r["model_id"] for r in job["result"]["leaderboard"]}
    lgb = client.get(f"{API}/models/{ids['lightgbm']}/explanations").json()
    assert lgb["global_importance"]["shap"]["method"].startswith("TreeSHAP (LightGBM")
    assert lgb["global_importance"]["permutation"]["metric"] == "r2"
    assert lgb["global_importance"]["permutation"]["values"][0]["feature"] == "assessed_value"
    ea = lgb["error_analysis"]
    assert {"mae", "large_errors"} <= set(ea["summary"]) and "largest_errors" in ea
    assert lgb["fairness"]["metric"] == "relative_mae_gap"
    assert lgb["samples"][0]["abs_error"] >= 0 and lgb["samples"][0]["correct"] is None

    ridge = ids["ridge"]
    client.get(f"{API}/models/{ridge}/explanations")
    row = client.get(f"{API}/models/{ridge}/explanations").json()["samples"][0]["row_id"]
    shap = client.post(f"{API}/models/{ridge}/explain", json={"row_id": row}).json()
    lime = client.post(f"{API}/models/{ridge}/explain", json={"row_id": row, "method": "lime"}).json()
    assert shap["method"].startswith("LinearSHAP") and isinstance(shap["actual"], float)
    assert lime["method"].startswith("LIME") and lime["fidelity"] > 0.99
    # On a linear model, LIME with the same baseline recovers the exact linear SHAP values.
    s = {c["feature"]: c["contribution"] for c in shap["contributions"]}
    lv = {c["feature"]: c["contribution"] for c in lime["contributions"]}
    top = max(s, key=lambda k: abs(s[k]))
    assert lv[top] == pytest.approx(s[top], rel=0.05)
    assert shap["explanation"].startswith("Predicted ")


def test_lime_for_classification_models(client, churn_dataset):
    job = train(client, churn_dataset["id"], algorithms=["xgboost"], cv_folds=0)
    mid = job["result"]["best_model_id"]
    row = client.get(f"{API}/models/{mid}/explanations").json()["samples"][0]["row_id"]
    r = client.post(f"{API}/models/{mid}/explain", json={"row_id": row, "method": "lime"}).json()
    assert r["method"].startswith("LIME") and 0 <= r["fidelity"] <= 1 and r["units"] == "probability"
    assert len(r["contributions"]) == 11


def test_optimiser_regression_plan(client, trained_houses):
    _, job = trained_houses
    best = job["result"]["best_model_id"]
    r = client.post(f"{API}/models/{best}/experiments", json={"budget": "quick"})
    res = client.get(f"{API}/jobs/{r.json()['id']}").json()["result"]
    assert res["primary_metric"] == "r2" and res["task"] == "regression"
    assert res["plan"]["leaky_columns"] == ["assessed_value"]
    assert res["plan"]["interventions"][0] == "target_transform"
    assert "class_weight" not in res["plan"]["interventions"]
    tt = next(t for t in res["trials"] if t["intervention"] == "target_transform")
    assert tt["config"]["target_transform"] == "log1p" and "rmse" in tt["metrics"]
    assert "R²" in res["selection"]


def test_predict_compare_and_drift(client, trained_houses, house_df):
    ds, job = trained_houses
    a, b = [r["model_id"] for r in job["result"]["leaderboard"]]
    rec = {k: (None if isinstance(v, float) and np.isnan(v) else v)
           for k, v in house_df.drop(columns=["price"]).iloc[0].to_dict().items()}
    p = client.post(f"{API}/models/{a}/predict", json={"records": [rec]}).json()["predictions"][0]
    assert isinstance(p["prediction"], float) and p["probabilities"] is None

    c = client.get(f"{API}/models/compare/{a}/{b}").json()
    rows = {m["metric"]: m for m in c["metrics"]}
    assert {"r2", "rmse", "mae"} <= set(rows) and not any(k.startswith("recall") for k in rows)
    if rows["rmse"]["delta"]:
        assert rows["rmse"]["better"] == ("b" if rows["rmse"]["delta"] < 0 else "a")  # lower is better

    shifted = house_df.copy()
    shifted["sqft"] = shifted["sqft"] * 1.8
    shifted["price"] = shifted["price"] * 0.5
    r = client.post(f"{API}/models/{a}/monitor", files={"file": ("b.csv", to_csv_bytes(shifted), "text/csv")})
    rep = r.json()["report"]
    assert rep["prediction_drift"]["kind"] == "values"
    assert rep["performance"]["metric"] == "r2" and rep["performance"]["delta_primary"] < -0.1
    assert "distribution.concept_drift" in {f["check"] for f in rep["findings"]}


def test_assistant_answers_regression_questions(client, trained_houses):
    ds, _ = trained_houses
    a = client.post(f"{API}/assistant/ask", json={"dataset_id": ds["id"], "question": "Why are the errors large?"}).json()
    assert a["mode"] == "rules" and "R²" in a["answer"]


@pytest.mark.skipif(not ALGORITHMS["pytorch_mlp"].available, reason="torch not installed")
@pytest.mark.parametrize("target", ["price", "churned"])
def test_pytorch_mlp_trains_for_both_tasks(client, project, house_df, churn_df, target):
    df = house_df if target == "price" else churn_df
    ds = upload(client, project["id"], to_csv_bytes(df)).json()
    client.post(f"{API}/datasets/{ds['id']}/target", json={"column": target})
    job = train(client, ds["id"], algorithms=["pytorch_mlp"], cv_folds=0)
    m = client.get(f"{API}/models/{job['result']['best_model_id']}").json()
    assert m["algorithm"] == "pytorch_mlp" and m["scores"]["primary"] > 0.3
