from tests.conftest import API, make_df, to_csv_bytes, upload


def test_health(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0", "database": "ok"}


def test_create_list_get_project(client):
    r = client.post(f"{API}/projects", json={"name": "  Churn  ", "description": "telco"})
    assert r.status_code == 201
    created = r.json()
    assert created["name"] == "Churn"
    assert created["dataset_count"] == 0

    listed = client.get(f"{API}/projects").json()
    assert [p["id"] for p in listed] == [created["id"]]

    got = client.get(f"{API}/projects/{created['id']}").json()
    assert got["description"] == "telco"


def test_project_validation_and_not_found(client):
    r = client.post(f"{API}/projects", json={"name": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"

    r = client.get(f"{API}/projects/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_dataset_count_and_delete_removes_files(client, project, settings):
    upload(client, project["id"], to_csv_bytes(make_df()))
    upload(client, project["id"], to_csv_bytes(make_df(seed=1)))

    listed = client.get(f"{API}/projects").json()
    assert listed[0]["dataset_count"] == 2
    assert len(list(settings.uploads_dir.iterdir())) == 2

    assert client.delete(f"{API}/projects/{project['id']}").status_code == 204
    assert client.get(f"{API}/projects/{project['id']}").status_code == 404
    assert list(settings.uploads_dir.iterdir()) == []
