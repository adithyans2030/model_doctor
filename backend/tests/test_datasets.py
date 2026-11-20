import pandas as pd
import pytest

from tests.conftest import API, make_df, to_csv_bytes, upload


def _types(body: dict) -> dict:
    return {c["name"]: c["inferred_type"] for c in body["columns"]}


def test_upload_profiles_columns(client, project):
    r = upload(client, project["id"], to_csv_bytes(make_df()), filename="customers.csv")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["original_filename"] == "customers.csv"
    assert (body["n_rows"], body["n_cols"]) == (120, 8)
    assert body["delimiter"] == ","
    assert body["target_column"] is None
    assert _types(body) == {
        "customer_id": "id",
        "age": "numeric",
        "income": "numeric",
        "plan": "categorical",
        "active": "boolean",
        "signup": "datetime",
        "note": "text",
        "label": "boolean",
    }


def test_missing_values_and_numeric_as_text(client, project):
    df = make_df()
    df.loc[:9, "income"] = None
    df["amount"] = df["income"].map(lambda v: "" if pd.isna(v) else f"{v}")
    df.loc[50, "amount"] = "unknown"  # not one of pandas' default NA tokens
    body = upload(client, project["id"], to_csv_bytes(df)).json()
    cols = {c["name"]: c for c in body["columns"]}
    assert cols["income"]["n_missing"] == 10
    assert cols["income"]["missing_pct"] == pytest.approx(8.33)
    assert cols["amount"]["inferred_type"] == "numeric"
    assert cols["amount"]["numeric_stored_as_text"] is True


def test_semicolon_latin1_and_bom(client, project):
    df = make_df()
    df["city"] = ["Zürich" if i % 2 else "Málaga" for i in range(len(df))]

    r = upload(client, project["id"], to_csv_bytes(df, sep=";", encoding="latin-1"))
    assert r.status_code == 201, r.text
    assert r.json()["delimiter"] == ";"
    assert r.json()["n_cols"] == 9

    r = upload(client, project["id"], b"\xef\xbb\xbf" + to_csv_bytes(df))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["encoding"] == "utf-8-sig"
    assert body["columns"][0]["name"] == "customer_id"  # BOM not glued to the first name


def test_whitespace_padded_headers_round_trip(client, project):
    df = make_df().rename(columns={"label": " label "})
    body = upload(client, project["id"], to_csv_bytes(df)).json()
    assert "label" in [c["name"] for c in body["columns"]]
    r = client.post(f"{API}/datasets/{body['id']}/target", json={"column": "label"})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("data.xlsx", b"a,b\n1,2\n", 422, "invalid_file"),
        ("data", b"a,b\n1,2\n", 422, "invalid_file"),
        ("data.csv", b"", 422, "invalid_file"),
        ("data.csv", b"PK\x03\x04rest-of-zip", 422, "invalid_file"),
        ("data.csv", b"\x80\x04\x95pickle", 422, "invalid_file"),
        ("data.csv", b"a,b\n1,\x002\n", 422, "invalid_file"),
        ("data.csv", b"a,b\n" + b"1,2\n" * 10, 422, "invalid_file"),  # too few rows
        ("data.csv", b"only\n" + b"1\n" * 50, 422, "invalid_file"),  # one column
        ("data.csv", b"a,a,b\n" + b"1,2,3\n" * 50, 422, "invalid_file"),  # duplicate names
        ("data.csv", b"a,b\n" + b"1,2\n" * 40 + b"1,2,3,4\n", 422, "invalid_file"),  # malformed
    ],
)
def test_rejects_bad_uploads(client, project, settings, filename, content, status, code):
    r = upload(client, project["id"], content, filename=filename)
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    assert list(settings.uploads_dir.iterdir()) == []  # nothing left behind


def test_rejects_too_many_columns_and_rows(client, project, settings):
    wide = ",".join(f"c{i}" for i in range(51)).encode() + b"\n"
    wide += b",".join(b"1" for _ in range(51)) * 1 + b"\n"
    r = upload(client, project["id"], wide * 1)
    assert r.status_code == 422
    assert "51 columns" in r.json()["error"]["message"]

    tall = b"a,b\n" + b"1,2\n" * 5001
    r = upload(client, project["id"], tall)
    assert r.status_code == 422
    assert "more than 5,000 rows" in r.json()["error"]["message"]


def test_rejects_oversized_file(client, project, settings):
    big = b"a,b\n" + b"12345,67890\n" * 100_000  # ~1.2 MB, limit is 1 MB
    r = upload(client, project["id"], big)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"
    assert list(settings.uploads_dir.iterdir()) == []


def test_path_traversal_filename_is_display_only(client, project, settings):
    r = upload(client, project["id"], to_csv_bytes(make_df()), filename="../../evil.csv")
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.csv"
    stored = list(settings.uploads_dir.iterdir())
    assert len(stored) == 1 and stored[0].name != "evil.csv"


def test_upload_to_missing_project(client):
    r = upload(client, "nope", to_csv_bytes(make_df()))
    assert r.status_code == 404


def test_get_list_preview_delete(client, project, settings):
    ds = upload(client, project["id"], to_csv_bytes(make_df())).json()

    assert client.get(f"{API}/datasets/{ds['id']}").json()["sha256"] == ds["sha256"]
    listed = client.get(f"{API}/projects/{project['id']}/datasets").json()
    assert [d["id"] for d in listed] == [ds["id"]]

    prev = client.get(f"{API}/datasets/{ds['id']}/preview", params={"rows": 5}).json()
    assert prev["columns"][0] == "customer_id"
    assert prev["n_rows_shown"] == 5 and prev["total_rows"] == 120
    assert prev["rows"][0][0] == "C0000"

    assert client.get(f"{API}/datasets/{ds['id']}/preview", params={"rows": 500}).status_code == 422

    assert client.delete(f"{API}/datasets/{ds['id']}").status_code == 204
    assert client.get(f"{API}/datasets/{ds['id']}").status_code == 404
    assert list(settings.uploads_dir.iterdir()) == []


def test_preview_serialises_missing_as_null(client, project):
    df = make_df()
    df.loc[0, "income"] = None
    ds = upload(client, project["id"], to_csv_bytes(df)).json()
    prev = client.get(f"{API}/datasets/{ds['id']}/preview", params={"rows": 1}).json()
    assert prev["rows"][0][prev["columns"].index("income")] is None
