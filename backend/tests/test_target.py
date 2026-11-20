import numpy as np
import pytest

from tests.conftest import API, make_df, to_csv_bytes, upload


def _set_target(client, project, df, column):
    ds = upload(client, project["id"], to_csv_bytes(df)).json()
    return client.post(f"{API}/datasets/{ds['id']}/target", json={"column": column})


def test_binary_target(client, project):
    df = make_df()
    r = _set_target(client, project, df, "label")
    assert r.status_code == 200, r.text
    body = r.json()
    ds = body["dataset"]
    assert ds["target_column"] == "label"
    assert ds["task_type"] == "binary_classification"
    assert ds["n_classes"] == 2
    counts = {c["label"]: c["count"] for c in ds["class_distribution"]}
    assert counts == {"0": int((df.label == 0).sum()), "1": int((df.label == 1).sum())}
    assert ds["class_distribution"][0]["count"] >= ds["class_distribution"][1]["count"]
    assert body["warnings"] == []


def test_multiclass_string_target(client, project):
    r = _set_target(client, project, make_df(), "plan")
    ds = r.json()["dataset"]
    assert ds["task_type"] == "multiclass_classification"
    assert sorted(c["label"] for c in ds["class_distribution"]) == ["basic", "enterprise", "pro"]
    assert sum(c["share"] for c in ds["class_distribution"]) == pytest.approx(1.0, abs=1e-3)


def test_integer_target_with_few_values_is_classification(client, project):
    df = make_df()
    df["rating"] = np.arange(len(df)) % 5 + 1
    r = _set_target(client, project, df, "rating")
    assert r.status_code == 200
    assert r.json()["dataset"]["n_classes"] == 5


def test_boolean_strings_normalised(client, project):
    df = make_df()
    df["active"] = ["Yes" if i % 3 else "yes" for i in range(len(df))]
    df.loc[::4, "active"] = "No"
    r = _set_target(client, project, df, "active")
    assert r.status_code == 200, r.text
    labels = {c["label"] for c in r.json()["dataset"]["class_distribution"]}
    assert labels == {"yes", "no"}


def test_continuous_target_is_unsupported(client, project):
    r = _set_target(client, project, make_df(), "income")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_task"
    assert "regression" in r.json()["error"]["message"]


@pytest.mark.parametrize("column", ["customer_id", "note", "signup"])
def test_unusable_column_types(client, project, column):
    r = _set_target(client, project, make_df(), column)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_target"


def test_constant_and_missing_column(client, project):
    df = make_df()
    df["const"] = "same"
    r = _set_target(client, project, df, "const")
    assert r.status_code == 422
    assert "single value" in r.json()["error"]["message"]

    r = _set_target(client, project, make_df(), "nope")
    assert r.status_code == 422
    assert "does not exist" in r.json()["error"]["message"]


def test_too_few_labelled_rows(client, project):
    df = make_df()
    df.loc[20:, "label"] = None
    r = _set_target(client, project, df, "label")
    assert r.status_code == 422
    assert "non-missing" in r.json()["error"]["message"]


def test_warnings(client, project):
    df = make_df()
    df["segment"] = ["A"] * 100 + ["a"] * 15 + ["B"] * 5
    df.loc[0:4, "segment"] = None
    r = _set_target(client, project, df, "segment")
    assert r.status_code == 200, r.text
    warnings = " | ".join(r.json()["warnings"])
    assert "5 rows" in warnings and "no target value" in warnings
    assert "differ only by case" in warnings and "A / a" in warnings
    assert "'B' (5)" in warnings


def test_resetting_target_replaces_previous(client, project):
    ds = upload(client, project["id"], to_csv_bytes(make_df())).json()
    client.post(f"{API}/datasets/{ds['id']}/target", json={"column": "label"})
    r = client.post(f"{API}/datasets/{ds['id']}/target", json={"column": "plan"})
    got = client.get(f"{API}/datasets/{ds['id']}").json()
    assert got["target_column"] == "plan"
    assert got["n_classes"] == 3
