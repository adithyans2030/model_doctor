import numpy as np
import pytest

from app.diagnostics.health import compute_health
from app.diagnostics.model_doctor import diagnose_model
from app.ml.evaluation import classification_metrics, expected_calibration_error, threshold_analysis
from tests.conftest import API, make_df, to_csv_bytes, train, upload


# ----------------------------------------------------------------------------- metrics
def test_classification_metrics_binary():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6], [0.45, 0.55]])
    m = classification_metrics(y, proba, ["no", "yes"], positive_index=1)
    assert m["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert m["confusion_matrix"]["matrix"] == [[3, 1], [0, 2]]
    assert m["positive_class"]["recall"] == 1.0
    assert m["per_class"][0]["support"] == 4
    assert m["worst_class"]["label"] == "yes"
    # Threshold 0.65 keeps the false positive (0.7) and loses both true positives.
    strict = classification_metrics(y, proba, ["no", "yes"], 1, threshold=0.65)
    assert strict["confusion_matrix"]["matrix"] == [[3, 1], [2, 0]]


def test_threshold_analysis_and_ece():
    y = np.array([0] * 90 + [1] * 10)
    p1 = np.r_[np.full(90, 0.2), np.full(10, 0.35)]
    proba = np.column_stack([1 - p1, p1])
    ta = threshold_analysis(y, proba, 1)
    assert ta["f1_at_0_5"] == 0.0
    assert 0.2 < ta["best_threshold"] <= 0.35 and ta["best_f1"] == 1.0

    perfect = np.array([[0.0, 1.0]] * 10 + [[1.0, 0.0]] * 10)
    yy = np.array([1] * 10 + [0] * 10)
    assert expected_calibration_error(yy, perfect, 1)[0] == 0.0


# ----------------------------------------------------------------------------- doctor rules
def _metrics(**over):
    per_class = over.pop("per_class", [
        {"label": "0", "precision": 0.92, "recall": 0.98, "f1": 0.95, "support": 900},
        {"label": "1", "precision": 0.6, "recall": 0.2, "f1": 0.3, "support": 100},
    ])
    test = {
        "accuracy": 0.9, "balanced_accuracy": 0.59, "f1_macro": 0.62, "roc_auc": 0.8, "ece": 0.02,
        "per_class": per_class, "worst_class": per_class[1], "calibration_curve": [], "threshold": None,
        "threshold_analysis": {"best_threshold": 0.25, "best_f1": 0.45, "f1_at_0_5": 0.3},
        **over.pop("test", {}),
    }
    base = {
        "test": test, "train": {"f1_macro": 0.66}, "positive_label": "1",
        "cv": {"folds": 5, "train": {"f1_macro": {"mean": 0.9, "std": 0.01}},
               "val": {"f1_macro": {"mean": 0.6, "std": 0.02}}},
        "dummy": {"balanced_accuracy": 0.5},
    }
    base.update(over)
    return base


IMBALANCE = {"check": "data.class_imbalance", "columns": ["y"], "evidence": {"imbalance_ratio": 9.0}}


def test_doctor_flags_imbalance_symptoms():
    findings = diagnose_model(_metrics(), [{"feature": "a", "importance": 0.3}], [IMBALANCE])
    checks = {f["check"]: f for f in findings}
    weak = checks["model.weak_class"]
    assert weak["severity"] == "critical" and "Class imbalance" in weak["cause"]
    assert set(weak["interventions"]) == {"class_weight", "smote", "threshold_tuning"}
    assert checks["model.overfitting"]["severity"] == "critical"
    assert checks["model.misleading_accuracy"]["severity"] == "warning"
    assert checks["model.suboptimal_threshold"]["evidence"]["best_threshold"] == 0.25
    assert "model.no_skill" not in checks


def test_doctor_no_skill_underfit_calibration_and_leak():
    no_skill = _metrics(test={"balanced_accuracy": 0.52})
    assert "model.no_skill" in {f["check"] for f in diagnose_model(no_skill, None, [])}

    under = _metrics(cv=None, train={"f1_macro": 0.5}, test={"f1_macro": 0.48, "balanced_accuracy": 0.7})
    assert "model.underfitting" in {f["check"] for f in diagnose_model(under, None, [])}

    calib = _metrics(test={"ece": 0.15})
    assert "calibration" in {i for f in diagnose_model(calib, None, []) for i in f["interventions"]}

    leak = {"check": "feature.leakage", "columns": ["flag"], "evidence": {}}
    great = _metrics(test={"f1_macro": 0.99, "roc_auc": 0.999})
    f = next(f for f in diagnose_model(great, [{"feature": "flag", "importance": 0.8}], [leak])
             if f["check"] == "model.suspicious_performance")
    assert f["severity"] == "critical" and f["interventions"] == ["drop_leaky_features"]


def test_health_score_renormalises_weights():
    h = compute_health(80, _metrics())
    comps = {k: v["value"] for k, v in h["components"].items()}
    assert comps["performance"] == 62 and comps["fairness"] is None and comps["drift"] is None
    expected = (0.25 * 80 + 0.30 * 62 + 0.20 * comps["robustness"]) / 0.75
    assert h["overall"] == round(expected)
    assert compute_health(None, None)["overall"] is None
    assert compute_health(90, None)["overall"] == 90


# ----------------------------------------------------------------------------- API
def test_training_job_and_leaderboard(client, churn_dataset, settings):
    job = train(client, churn_dataset["id"])
    res = job["result"]
    assert len(res["model_ids"]) == 2 and res["split"]["n_test"] == 404
    assert res["task"] == "classification" and res["primary_metric"] == "f1_macro"
    f1s = [r["f1_macro"] for r in res["leaderboard"]]
    assert all(r["primary"] == r["f1_macro"] for r in res["leaderboard"])
    assert f1s == sorted(f1s, reverse=True) and res["best_model_id"] == res["leaderboard"][0]["model_id"]
    assert res["dummy"]["balanced_accuracy"] == 0.5
    assert len(list(settings.models_dir.iterdir())) == 2

    models = client.get(f"{API}/datasets/{churn_dataset['id']}/models").json()
    assert [m["version"] for m in models] == [1, 2]
    assert all(m["stage"] == "candidate" for m in models)

    detail = client.get(f"{API}/models/{res['best_model_id']}").json()
    assert detail["metrics"]["cv"]["folds"] == 3
    assert detail["metrics"]["positive_label"] == "1"
    spec = detail["feature_spec"]
    assert "cancellation_request" in spec["numeric"] + spec["categorical"]
    # With the leak included the model looks too good, and the doctor says why.
    checks = {f["check"] for f in detail["findings"]}
    assert checks & {"model.dominant_feature", "model.suspicious_performance"}
    assert detail["importances"][0]["feature"] == "cancellation_request"


def test_leak_free_model_shows_imbalance_problem(client, trained):
    best = trained["best"]
    spec = best["feature_spec"]
    assert "cancellation_request" not in spec["numeric"] + spec["categorical"]
    assert spec["excluded"]["cancellation_request"] == "dropped by configuration"
    checks = {f["check"]: f for f in best["findings"]}
    assert "model.weak_class" in checks
    assert "Class imbalance" in checks["model.weak_class"]["cause"]
    assert best["metrics"]["test"]["accuracy"] > best["metrics"]["test"]["balanced_accuracy"]


def test_diagnosis_combines_everything(client, trained):
    d = client.get(f"{API}/datasets/{trained['dataset']['id']}/diagnosis").json()
    assert d["model"]["id"] == trained["best"]["id"]
    comps = d["health"]["components"]
    assert comps["data"]["value"] is not None and comps["performance"]["value"] is not None
    assert d["health"]["overall"] is not None
    leak = next(f for f in d["findings"] if f["check"] == "feature.leakage")
    assert leak["resolved"] is True and leak["severity"] == "info"  # excluded from this model
    labels = [r["intervention"] for r in d["recommendations"]]
    assert "class_weight" in labels and "drop_leaky_features" not in labels
    assert any("recall for class '1'" in line for line in d["narrative"])


def test_diagnosis_without_models(client, churn_dataset):
    d = client.get(f"{API}/datasets/{churn_dataset['id']}/diagnosis").json()
    assert d["model"] is None
    assert d["health"]["overall"] == d["health"]["components"]["data"]["value"]


def _records(df, n=3):
    records = df.head(n).to_dict(orient="records")
    for r in records:  # JSON can't carry NaN
        for k, v in list(r.items()):
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return records


def test_predict(client, trained, churn_df):
    records = _records(churn_df.drop(columns=["churned"]))
    res = client.post(f"{API}/models/{trained['best']['id']}/predict", json={"records": records})
    assert res.status_code == 200, res.text
    preds = res.json()["predictions"]
    assert len(preds) == 3 and all(p["prediction"] in {"0", "1"} for p in preds)
    assert all(abs(sum(p["probabilities"].values()) - 1) < 1e-3 for p in preds)

    bad = client.post(f"{API}/models/{trained['best']['id']}/predict", json={"records": [{"age": 3}]})
    assert bad.status_code == 400 and "missing feature columns" in bad.json()["error"]["message"]


def test_training_validation_errors(client, churn_dataset):
    r = client.post(f"{API}/datasets/{churn_dataset['id']}/train", json={"drop_columns": ["nope"]})
    assert r.status_code == 400
    r = client.post(f"{API}/datasets/{churn_dataset['id']}/train", json={"algorithms": ["magic"]})
    assert r.status_code == 422
    r = client.post(f"{API}/datasets/{churn_dataset['id']}/train", json={"cv_folds": 1})
    assert r.status_code == 422
    assert client.get(f"{API}/jobs/nope").status_code == 404


def test_training_without_target_fails_cleanly(client, project):
    ds = upload(client, project["id"], to_csv_bytes(make_df())).json()
    r = client.post(f"{API}/datasets/{ds['id']}/train", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_target"


def test_multiclass_training(client, churn_dataset):
    client.post(f"{API}/datasets/{churn_dataset['id']}/target", json={"column": "contract"})
    job = train(client, churn_dataset["id"], algorithms=["logistic_regression", "svm"], cv_folds=0)
    best = client.get(f"{API}/models/{job['result']['best_model_id']}").json()
    assert best["metrics"]["positive_label"] is None and best["metrics"]["cv"] is None
    assert len(best["metrics"]["test"]["per_class"]) == 3
    assert "threshold_analysis" not in best["metrics"]["test"]


def test_deleting_dataset_removes_model_files(client, trained, settings):
    assert any(settings.models_dir.iterdir())
    client.delete(f"{API}/datasets/{trained['dataset']['id']}")
    assert not any(settings.models_dir.iterdir())


def test_interrupted_jobs_are_recovered(app):
    from app.db.models import Dataset, Job, Project
    from app.services.jobs import recover_interrupted

    with app.state.session_factory() as db:
        p = Project(name="p")
        db.add(p)
        db.flush()
        d = Dataset(project_id=p.id, original_filename="x.csv", stored_filename="x.csv", size_bytes=1,
                    sha256="0", encoding="utf-8", delimiter=",", n_rows=1, n_cols=1, columns=[])
        db.add(d)
        db.flush()
        db.add(Job(dataset_id=d.id, kind="training", status="running", params={}))
        db.commit()
    assert recover_interrupted(app.state.session_factory) == 1
