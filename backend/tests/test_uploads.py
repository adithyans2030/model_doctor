"""Bring-your-own-model: safe skops uploads, evaluation and explanation."""

import pickle

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import build_feature_spec, labelled_rows, prepare_frame
from tests.conftest import API, to_csv_bytes, train, upload

skops_io = pytest.importorskip("skops.io")


class Evil:
    """Stands in for arbitrary code an attacker would hide in a model file."""

    def __init__(self):
        self.payload = "rm -rf /"


def _user_pipeline(estimator, spec):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), spec.numeric_like),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), spec.categorical),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def _fit_user_model(dataset, df, settings, estimator, *, drop=()):
    """What a user would do: train their own sklearn pipeline on the dataset's columns."""
    profile = next(c for c in dataset["columns"] if c["name"] == dataset["target_column"])
    frame, y = labelled_rows(df, dataset["target_column"], profile, settings)
    spec = build_feature_spec(dataset["columns"], dataset["target_column"], drop)
    X = prepare_frame(frame, spec)
    y = y.astype(int) if not np.issubdtype(y.dtype, np.floating) else y
    return _user_pipeline(estimator, spec).fit(X, y)


def _skops_bytes(obj) -> bytes:
    return skops_io.dumps(obj)


def _upload_model(client, dataset_id, content: bytes, filename="model.skops", name=None):
    data = {"name": name} if name else {}
    return client.post(f"{API}/datasets/{dataset_id}/models/upload", data=data,
                       files={"file": (filename, content, "application/octet-stream")})


def test_upload_classifier_is_evaluated_and_explained(client, churn_dataset, churn_df, settings):
    est = _fit_user_model(churn_dataset, churn_df, settings, LogisticRegression(max_iter=2000))
    r = _upload_model(client, churn_dataset["id"], _skops_bytes(est), name="My logistic model")
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["source"] == "uploaded" and m["algorithm"] == "uploaded" and m["name"] == "My logistic model"
    assert m["metrics"]["external"]["estimator"] == "Pipeline" and m["metrics"]["cv"] is None
    assert 0 < m["scores"]["f1_macro"] <= 1
    checks = {f["check"] for f in m["findings"]}
    assert "model.external_evaluation" in checks
    assert not list((settings.data_dir / "incoming").glob("*"))  # raw upload not kept

    e = client.get(f"{API}/models/{m['id']}/explanations").json()
    assert e["global_importance"]["shap"] is None  # own preprocessing: model-agnostic methods
    assert e["global_importance"]["permutation"]["values"][0]["feature"] == "cancellation_request"
    row = e["samples"][0]["row_id"]
    for method in ("auto", "lime"):
        x = client.post(f"{API}/models/{m['id']}/explain", json={"row_id": row, "method": method}).json()
        assert x["prediction"] in {"0", "1"} and x["contributions"]
    assert client.post(f"{API}/models/{m['id']}/predict", json={"records": [
        {k: (None if isinstance(v, float) and np.isnan(v) else v)
         for k, v in churn_df.drop(columns=["churned"]).iloc[0].to_dict().items()}]}).status_code == 200

    # Experiments retrain models, so they are refused for uploads; comparison works.
    assert client.post(f"{API}/models/{m['id']}/experiments", json={}).status_code == 400
    job = train(client, churn_dataset["id"], algorithms=["logistic_regression"], cv_folds=0)
    c = client.get(f"{API}/models/compare/{job['result']['best_model_id']}/{m['id']}").json()
    assert c["same_test_split"] is True


def test_upload_regressor(client, project, settings):
    from scripts.make_demo_data import make_house_prices

    df = make_house_prices(rows=600, seed=5)
    ds = upload(client, project["id"], to_csv_bytes(df)).json()
    ds = client.post(f"{API}/datasets/{ds['id']}/target", json={"column": "price"}).json()["dataset"]
    est = _fit_user_model(ds, df, settings, Ridge())
    m = _upload_model(client, ds["id"], _skops_bytes(est)).json()
    assert m["task"] == "regression" and m["scores"]["r2"] > 0.5 and m["name"].startswith("Uploaded Pipeline v")


def test_unsafe_and_wrong_files_are_refused(client, churn_dataset, churn_df, settings):
    # A skops file carrying a non-allowlisted type is refused before anything is loaded.
    evil = _skops_bytes({"model": Evil()})
    r = _upload_model(client, churn_dataset["id"], evil)
    assert r.status_code == 422 and "outside the allowed ML libraries" in r.json()["error"]["message"]
    assert "Evil" in r.json()["error"]["message"]

    # Pickle/joblib are never accepted, whatever their content.
    pkl = pickle.dumps({"a": 1})
    r = _upload_model(client, churn_dataset["id"], pkl, filename="model.pkl")
    assert r.status_code == 422 and "Pickle" in r.json()["error"]["message"]
    r = _upload_model(client, churn_dataset["id"], pkl, filename="model.skops")  # renamed pickle
    assert r.status_code == 422 and "Not a valid skops file" in r.json()["error"]["message"]

    # A model trained on different columns fails cleanly.
    est = _fit_user_model(churn_dataset, churn_df, settings, LogisticRegression(max_iter=500), drop=["contract", "region"])
    r = _upload_model(client, churn_dataset["id"], _skops_bytes(est))
    assert r.status_code == 201  # extra columns are fine: the pipeline selects what it was trained on
    narrow = _user_pipeline(LogisticRegression(), build_feature_spec(churn_dataset["columns"], "churned"))
    r = _upload_model(client, churn_dataset["id"], _skops_bytes(narrow))  # never fitted
    assert r.status_code == 422 and "failed to predict" in r.json()["error"]["message"] or "classes" in r.text
