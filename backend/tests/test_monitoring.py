import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import RateLimiter, parse_api_keys
from app.db.models import Base
from app.main import create_app
from app.monitoring.drift import psi
from tests.conftest import API, to_csv_bytes, train


# ----------------------------------------------------------------------------- drift maths
def test_psi_properties():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0, abs=1e-9)
    assert psi([50, 50], [90, 10]) > 0.25  # major shift
    assert 0 < psi([50, 50], [55, 45]) < 0.1  # small shift


# ----------------------------------------------------------------------------- drift API
@pytest.fixture
def model(client, churn_dataset):
    job = train(client, churn_dataset["id"], algorithms=["logistic_regression"],
                drop_columns=["cancellation_request"], cv_folds=0)
    return churn_dataset, job["result"]["best_model_id"]


def _monitor(client, model_id, df, name="batch.csv"):
    return client.post(f"{API}/models/{model_id}/monitor", files={"file": (name, to_csv_bytes(df), "text/csv")})


def test_stable_batch_has_little_drift(client, model):
    from scripts.make_demo_data import make_churn

    _, mid = model
    r = _monitor(client, mid, make_churn(rows=800, seed=99).drop(columns=["churned"]))
    assert r.status_code == 201, r.text
    rep = r.json()["report"]
    assert rep["score"] >= 85
    assert rep["performance"] is None and rep["schema"]["has_labels"] is False
    assert not [f for f in rep["findings"] if f["severity"] == "critical"]


def test_shifted_batch_is_flagged(client, model, settings):
    from scripts.make_demo_data import make_churn

    ds, mid = model
    batch = make_churn(rows=800, seed=5)
    batch["monthly_charges"] = batch["monthly_charges"] * 1.6
    batch["contract"] = np.where(np.random.default_rng(0).random(len(batch)) < 0.9, "Two year", batch["contract"])
    batch["age"] = np.nan  # pipeline broke upstream
    batch["new_col"] = 1
    batch["churned"] = 1 - batch["churned"]  # labels flipped -> performance collapses
    rep = _monitor(client, mid, batch).json()["report"]

    by_feature = {f["feature"]: f for f in rep["features"]}
    assert by_feature["monthly_charges"]["severity"] == "major" and by_feature["monthly_charges"]["test"] == "ks"
    assert by_feature["contract"]["severity"] == "major" and by_feature["contract"]["test"] == "chi2"
    checks = {f["check"] for f in rep["findings"]}
    assert {"distribution.feature_drift", "distribution.missing_shift", "distribution.concept_drift",
            "distribution.schema_change"} <= checks
    assert rep["performance"]["delta_f1_macro"] < -0.1
    assert rep["schema"]["extra_columns"] == ["new_col"]
    assert rep["score"] < 60
    assert not (settings.data_dir / "batches").exists() or not any((settings.data_dir / "batches").iterdir())

    history = client.get(f"{API}/models/{mid}/drift").json()
    assert len(history) == 1 and history[0]["report"]["batch"]["filename"] == "batch.csv"

    d = client.get(f"{API}/datasets/{ds['id']}/diagnosis", params={"model_id": mid}).json()
    assert d["health"]["components"]["drift"]["value"] == rep["score"]
    assert any(f["category"] == "distribution" for f in d["findings"])


def test_batch_missing_features_is_rejected(client, model, churn_df):
    _, mid = model
    r = _monitor(client, mid, churn_df.drop(columns=["contract", "age"]).head(100))
    assert r.status_code == 400 and r.json()["error"]["code"] == "schema_mismatch"
    assert "contract" in r.json()["error"]["message"]


# ----------------------------------------------------------------------------- metrics
def test_prometheus_metrics(client, model):
    _, mid = model
    client.get(f"{API}/models/{mid}")
    text = client.get("/metrics").text
    assert "# TYPE modeldoctor_http_requests_total counter" in text
    assert 'route="/api/v1/models/{model_id}"' in text  # route templates, not raw ids
    assert "modeldoctor_http_request_duration_seconds_bucket" in text
    assert 'modeldoctor_models{stage="candidate"} 1' in text
    assert 'modeldoctor_jobs{kind="training",status="completed"} 1' in text


# ----------------------------------------------------------------------------- security
@pytest.fixture
def secured(tmp_path):
    app = create_app(Settings(_env_file=None, data_dir=tmp_path / "sec", api_keys=["alice:s3cret", "ci:ci-key"],
                              rate_limit_per_minute=8, n_jobs=1))
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        yield c
    app.state.engine.dispose()


def test_api_key_auth(secured):
    assert secured.get(f"{API}/health").status_code == 200  # public
    r = secured.get(f"{API}/projects")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert secured.get(f"{API}/projects", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get(f"{API}/projects", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert secured.get(f"{API}/projects", headers={"Authorization": "Bearer ci-key"}).status_code == 200
    assert secured.get("/metrics").status_code == 200  # scrape stays public by default


def test_audit_log_records_authenticated_actor(secured):
    h = {"X-API-Key": "s3cret"}
    secured.post(f"{API}/projects", json={"name": "audited"}, headers=h)
    secured.post(f"{API}/projects", json={"name": ""}, headers=h)  # failed attempts are audited too
    logs = secured.get(f"{API}/audit-logs", headers=h).json()
    assert [(l["actor"], l["method"], l["status_code"]) for l in logs][:2] == [
        ("alice", "POST", 422), ("alice", "POST", 201)]
    assert all(l["path"] == "/api/v1/projects" for l in logs)


def test_rate_limit_and_security_headers(secured):
    h = {"X-API-Key": "ci-key"}
    codes = [secured.get(f"{API}/projects", headers=h).status_code for _ in range(10)]
    assert codes[:8] == [200] * 8 and codes[8] == 429
    r = secured.get(f"{API}/projects", headers=h)
    assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
    # A different actor has its own budget.
    assert secured.get(f"{API}/projects", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"


def test_registry_uses_authenticated_actor(tmp_path, churn_df):
    app = create_app(Settings(_env_file=None, data_dir=tmp_path / "reg", api_keys=["alice:k"], n_jobs=1))
    Base.metadata.create_all(app.state.engine)
    h = {"X-API-Key": "k"}
    with TestClient(app) as c:
        p = c.post(f"{API}/projects", json={"name": "p"}, headers=h).json()
        ds = c.post(f"{API}/projects/{p['id']}/datasets", headers=h,
                    files={"file": ("d.csv", to_csv_bytes(churn_df), "text/csv")}).json()
        c.post(f"{API}/datasets/{ds['id']}/target", json={"column": "churned"}, headers=h)
        job = c.post(f"{API}/datasets/{ds['id']}/train", headers=h,
                     json={"algorithms": ["logistic_regression"], "cv_folds": 0}).json()
        mid = c.get(f"{API}/jobs/{job['id']}", headers=h).json()["result"]["best_model_id"]
        c.post(f"{API}/models/{mid}/stage", headers=h,
               json={"stage": "approved", "note": "ok", "actor": "mallory", "acknowledge_risks": True})
        events = c.get(f"{API}/models/{mid}/events", headers=h).json()
    app.state.engine.dispose()
    assert events[0]["actor"] == "alice"  # the self-declared actor is ignored


def test_helpers():
    assert parse_api_keys(["a:1", "2", ":3"]) == {"1": "a", "2": "api-key-2", "3": "api-key-3"}
    rl = RateLimiter(2)
    assert rl.check("x") is None and rl.check("x") is None and rl.check("x") > 0
    assert RateLimiter(0).check("x") is None
