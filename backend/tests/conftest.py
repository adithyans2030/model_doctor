import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.models import Base
from app.main import create_app

API = "/api/v1"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        max_upload_mb=1,
        max_rows=5000,
        max_cols=50,
        min_rows=30,
    )


@pytest.fixture
def app(settings):
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    yield app
    app.state.engine.dispose()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def project(client) -> dict:
    r = client.post(f"{API}/projects", json={"name": "Test project"})
    assert r.status_code == 201
    return r.json()


def make_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "age": rng.integers(18, 80, n),
            "income": rng.normal(50_000, 12_000, n).round(2),
            "plan": rng.choice(["basic", "pro", "enterprise"], n),
            "active": rng.choice(["yes", "no"], n),
            "signup": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "note": [f"customer left a fairly long comment number {i} about the service" for i in range(n)],
            "label": (rng.random(n) < 0.2).astype(int),
        }
    )


def to_csv_bytes(df: pd.DataFrame, **kwargs) -> bytes:
    encoding = kwargs.pop("encoding", "utf-8")
    buf = io.StringIO()
    df.to_csv(buf, index=False, **kwargs)
    return buf.getvalue().encode(encoding)


def upload(client, project_id: str, content: bytes, filename: str = "data.csv"):
    return client.post(
        f"{API}/projects/{project_id}/datasets",
        files={"file": (filename, content, "text/csv")},
    )
