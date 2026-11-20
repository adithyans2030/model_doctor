import numpy as np
import pandas as pd
import pytest

from app.ml.error_analysis import analyse_errors, group_performance, prediction_findings
from app.ml.explain import _class_values, shap_matrix
from app.ml.features import FeatureSpec
from app.ml.pipeline import TrainConfig
from app.ml.training import fit_config
from tests.conftest import API, train


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "g": rng.choice(["x", "y", "z"], n).astype(object),
    })
    y = ((X["a"] + 0.5 * (X["g"] == "x")) > 0.3).astype(int).to_numpy()
    spec = FeatureSpec(target="t", numeric=["a", "b"], categorical=["g"])
    return X, y, spec


def _bundle(algo):
    X, y, spec = _toy()
    pipe, spec_cfg = fit_config(TrainConfig(algorithm=algo), X, y, spec, n_jobs=1)
    return {"pipeline": pipe, "config": TrainConfig(algorithm=algo).to_dict(), "spec": spec_cfg.to_dict(),
            "classes": ["0", "1"], "positive_index": 1}, X


@pytest.mark.parametrize("algo", ["logistic_regression", "xgboost"])
def test_shap_values_are_additive(algo):
    """Contributions sum to the model's log-odds relative to the baseline."""
    bundle, X = _bundle(algo)
    vals, method = shap_matrix(bundle, X.head(20), background=X)
    assert vals.shape == (20, 3, 1) and "log-odds" in method
    p = bundle["pipeline"].predict_proba(X.head(20))[:, 1]
    logit = np.log(p / (1 - p))
    total = vals[:, :, 0].sum(axis=1)
    # Differences between rows are exact for both (the baseline/bias cancels out).
    np.testing.assert_allclose(np.diff(total), np.diff(logit), atol=1e-3)


def test_class_values_flip_sign_for_single_output():
    vals = np.array([[[1.0], [-2.0]]])
    np.testing.assert_array_equal(_class_values(vals, 1, 1), [[1.0, -2.0]])
    np.testing.assert_array_equal(_class_values(vals, 0, 1), [[-1.0, 2.0]])


def test_error_analysis_finds_cluster_and_group_gap():
    n = 400
    X = pd.DataFrame({"segment": ["A"] * 300 + ["B"] * 100, "v": np.random.default_rng(0).normal(size=n)})
    spec = FeatureSpec(target="t", numeric=["v"], categorical=["segment"])
    y = np.array([0, 1] * 200)
    pred = y.copy()
    pred[300:][y[300:] == 1] = 0  # in segment B every positive is missed
    proba = np.column_stack([1 - pred * 0.8 - 0.1, pred * 0.8 + 0.1])
    errors = analyse_errors(X, spec, y, pred, proba, ["0", "1"], 1, list(range(n)))
    assert errors["summary"]["false_negatives"] == 50
    assert errors["fn_concentration"][0]["value"] == "B" and errors["fn_concentration"][0]["fn_share"] == 1.0
    assert any("100% of false negatives occur where segment = B" in s for s in errors["statements"])

    fairness = group_performance(X, spec, y, pred, ["0", "1"], 1)
    assert fairness["worst_feature"] == "segment" and fairness["max_gap"] == 0.5
    checks = {f["check"]: f for f in prediction_findings(errors, fairness)}
    assert checks["fairness.group_gap"]["severity"] == "critical"
    assert checks["prediction.fn_concentration"]["columns"] == ["segment"]
    assert "prediction.error_cluster" in checks


# ----------------------------------------------------------------------------- API
@pytest.fixture
def two_models(client, churn_dataset):
    job = train(client, churn_dataset["id"], drop_columns=["cancellation_request"], cv_folds=0)
    ids = {r["algorithm"]: r["model_id"] for r in job["result"]["leaderboard"]}
    return churn_dataset, ids


def test_explanations_endpoint(client, two_models):
    _, ids = two_models
    e = client.get(f"{API}/models/{ids['xgboost']}/explanations").json()
    g = e["global_importance"]
    assert g["shap"]["method"].startswith("TreeSHAP") and g["shap"]["explained_class"] == "1"
    assert len(g["permutation"]["values"]) == 10
    assert {"contract", "tenure_months"} & {v["feature"] for v in g["shap"]["values"][:3]}
    ea = e["error_analysis"]
    assert ea["summary"]["n_test"] == 404 and ea["summary"]["false_negatives"] > 0
    assert ea["statements"]
    assert e["fairness"]["metric"] == "balanced_accuracy"
    assert len(e["samples"]) == 200

    lr = client.get(f"{API}/models/{ids['logistic_regression']}/explanations").json()
    assert lr["global_importance"]["shap"]["method"] == "LinearSHAP (log-odds)"


def test_explanations_are_cached_until_refresh(client, two_models):
    _, ids = two_models
    first = client.get(f"{API}/models/{ids['xgboost']}/explanations").json()["computed_at"]
    assert client.get(f"{API}/models/{ids['xgboost']}/explanations").json()["computed_at"] == first
    assert client.get(f"{API}/models/{ids['xgboost']}/explanations", params={"refresh": True}).json()["computed_at"] != first


def test_explain_row_and_record(client, two_models, churn_df):
    _, ids = two_models
    mid = ids["xgboost"]
    sample = client.get(f"{API}/models/{mid}/explanations").json()["samples"][0]
    r = client.post(f"{API}/models/{mid}/explain", json={"row_id": sample["row_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["actual"] == sample["true"] and body["prediction"] == sample["predicted"]
    assert body["in_test_sample"] is True
    assert len(body["contributions"]) == 10 and body["explanation"].startswith("Predicted")
    mags = [abs(c["contribution"]) for c in body["contributions"]]
    assert mags == sorted(mags, reverse=True)

    record = {k: (None if isinstance(v, float) and np.isnan(v) else v)
              for k, v in churn_df.drop(columns=["churned"]).iloc[0].to_dict().items()}
    r = client.post(f"{API}/models/{ids['logistic_regression']}/explain", json={"record": record})
    assert r.status_code == 200 and r.json()["actual"] is None and r.json()["units"] == "log-odds"


def test_explain_validation(client, two_models):
    _, ids = two_models
    mid = ids["xgboost"]
    assert client.post(f"{API}/models/{mid}/explain", json={}).status_code == 422
    assert client.post(f"{API}/models/{mid}/explain", json={"row_id": 1, "record": {}}).status_code == 422
    assert client.post(f"{API}/models/{mid}/explain", json={"row_id": 999999}).status_code == 404
    r = client.post(f"{API}/models/{mid}/explain", json={"record": {"age": 1}})
    assert r.status_code == 400


def test_diagnosis_includes_fairness_after_explanations(client, two_models):
    ds, ids = two_models
    before = client.get(f"{API}/datasets/{ds['id']}/diagnosis", params={"model_id": ids["xgboost"]}).json()
    assert before["health"]["components"]["fairness"]["value"] is None
    client.get(f"{API}/models/{ids['xgboost']}/explanations")
    after = client.get(f"{API}/datasets/{ds['id']}/diagnosis", params={"model_id": ids["xgboost"]}).json()
    assert after["health"]["components"]["fairness"]["value"] is not None
    assert any(f["category"] == "prediction" for f in after["findings"])
