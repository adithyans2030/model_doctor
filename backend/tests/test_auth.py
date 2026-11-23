"""User accounts: registration, login, roles and enforcement."""

import pytest
from fastapi.testclient import TestClient

from app.core.auth import hash_password, verify_password
from app.core.config import Settings
from app.db.models import Base
from app.main import create_app
from tests.conftest import API, to_csv_bytes


@pytest.fixture
def secure_client(tmp_path):
    app = create_app(Settings(_env_file=None, data_dir=tmp_path / "auth", auth_required=True, n_jobs=1,
                              assistant_mode="rules", rate_limit_per_minute=0))
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        yield c
    app.state.engine.dispose()


def _register(c, email, name="X", password="correct-horse-1"):
    return c.post(f"{API}/auth/register", json={"email": email, "name": name, "password": password})


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def test_password_hashing():
    h = hash_password("s3cret-pass")
    assert h.startswith("$argon2id$") and "s3cret-pass" not in h
    assert verify_password("s3cret-pass", h) and not verify_password("wrong", h)
    assert not verify_password("x", "not-a-hash")


def test_first_user_is_admin_and_login_flow(secure_client):
    c = secure_client
    assert c.get(f"{API}/auth/status").json() == {"auth_required": True, "open_registration": True, "has_users": False}
    assert c.get(f"{API}/projects").status_code == 401  # sign-in required

    r = _register(c, "Admin@Example.com", "Ada")
    assert r.status_code == 201
    admin = r.json()
    assert admin["user"]["role"] == "admin" and admin["user"]["email"] == "admin@example.com"
    assert admin["token_type"] == "bearer" and admin["expires_in"] > 0

    second = _register(c, "bob@example.com", "Bob").json()
    assert second["user"]["role"] == "member"
    assert _register(c, "bob@example.com").json()["error"]["code"] == "email_taken"
    assert _register(c, "not-an-email").status_code == 422
    assert c.post(f"{API}/auth/register", json={"email": "c@d.io", "name": "C", "password": "short"}).status_code == 422

    bad = c.post(f"{API}/auth/login", json={"email": "bob@example.com", "password": "nope"})
    assert bad.status_code == 401 and bad.json()["error"]["message"] == "Incorrect email or password."
    unknown = c.post(f"{API}/auth/login", json={"email": "ghost@example.com", "password": "nope"})
    assert unknown.json()["error"]["message"] == bad.json()["error"]["message"]  # no account enumeration

    tok = c.post(f"{API}/auth/login", json={"email": "BOB@example.com", "password": "correct-horse-1"}).json()
    me = c.get(f"{API}/auth/me", headers=_h(tok["access_token"])).json()
    assert me["name"] == "Bob" and me["last_login_at"]
    assert c.get(f"{API}/projects", headers=_h(tok["access_token"])).status_code == 200
    assert c.get(f"{API}/projects", headers=_h("garbage.token.value")).status_code == 401


def test_roles_viewer_member_admin(secure_client, churn_df):
    c = secure_client
    admin = _register(c, "admin@x.io").json()
    member = _register(c, "member@x.io").json()
    viewer = _register(c, "viewer@x.io").json()
    ah, mh, vh = _h(admin["access_token"]), _h(member["access_token"]), _h(viewer["access_token"])

    # Only admins manage users.
    assert c.get(f"{API}/auth/users", headers=mh).status_code == 403
    users = c.get(f"{API}/auth/users", headers=ah).json()
    assert [u["role"] for u in users] == ["admin", "member", "member"]
    r = c.patch(f"{API}/auth/users/{viewer['user']['id']}", json={"role": "viewer"}, headers=ah)
    assert r.json()["role"] == "viewer"
    assert c.patch(f"{API}/auth/users/{admin['user']['id']}", json={"role": "member"}, headers=ah).status_code == 400

    # Viewers are read-only (the role change applies to their existing token immediately).
    assert c.post(f"{API}/projects", json={"name": "v"}, headers=vh).status_code == 403
    p = c.post(f"{API}/projects", json={"name": "team"}, headers=mh).json()
    assert c.get(f"{API}/projects", headers=vh).status_code == 200

    ds = c.post(f"{API}/projects/{p['id']}/datasets", headers=mh,
                files={"file": ("c.csv", to_csv_bytes(churn_df), "text/csv")}).json()
    c.post(f"{API}/datasets/{ds['id']}/target", json={"column": "churned"}, headers=mh)
    job = c.post(f"{API}/datasets/{ds['id']}/train", headers=mh,
                 json={"algorithms": ["logistic_regression"], "cv_folds": 0}).json()
    mid = c.get(f"{API}/jobs/{job['id']}", headers=mh).json()["result"]["best_model_id"]

    # Viewers can still ask the assistant.
    a = c.post(f"{API}/assistant/ask", json={"dataset_id": ds["id"], "question": "summary"}, headers=vh)
    assert a.status_code == 200

    # Members can't approve; admins can, and the event records the signed-in user.
    body = {"stage": "approved", "note": "ok", "acknowledge_risks": True}
    assert c.post(f"{API}/models/{mid}/stage", json=body, headers=mh).status_code == 403
    assert c.post(f"{API}/models/{mid}/stage", json=body, headers=ah).json()["stage"] == "approved"
    events = c.get(f"{API}/models/{mid}/events", headers=ah).json()
    assert events[0]["actor"] == "admin@x.io"

    # Deactivated accounts lose access at once, even with a valid token.
    c.patch(f"{API}/auth/users/{member['user']['id']}", json={"is_active": False}, headers=ah)
    assert c.get(f"{API}/projects", headers=mh).status_code == 401
    assert c.post(f"{API}/auth/login", json={"email": "member@x.io", "password": "correct-horse-1"}).status_code == 403


def test_closed_registration_needs_admin(tmp_path):
    app = create_app(Settings(_env_file=None, data_dir=tmp_path / "closed", auth_required=True,
                              open_registration=False, rate_limit_per_minute=0))
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        admin = _register(c, "root@x.io").json()  # first account is always allowed
        assert _register(c, "eve@x.io").status_code == 401
        r = c.post(f"{API}/auth/register", headers=_h(admin["access_token"]),
                   json={"email": "eve@x.io", "name": "Eve", "password": "correct-horse-1"})
        assert r.status_code == 201 and r.json()["user"]["role"] == "member"
    app.state.engine.dispose()


def test_open_mode_still_works_without_login(client):
    """Default local setup: no sign-in required, but accounts can still be created."""
    assert client.get(f"{API}/projects").status_code == 200
    r = _register(client, "solo@x.io")
    assert r.status_code == 201 and r.json()["user"]["role"] == "admin"
