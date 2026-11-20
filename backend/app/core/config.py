"""Application settings, loaded from environment variables prefixed with ``MD_``."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MD_", env_file=BACKEND_DIR / ".env", extra="ignore"
    )

    app_name: str = "ModelDoctor"
    version: str = "0.1.0"

    data_dir: Path = BACKEND_DIR / "data"
    # Leave unset to use a SQLite file inside data_dir. Set a Postgres URL in Phase 7.
    database_url: str | None = None

    # Upload and dataset limits
    max_upload_mb: int = 50
    max_rows: int = 1_000_000
    max_cols: int = 500
    min_rows: int = 30

    # Target column rules
    max_classes: int = 50
    max_integer_target_classes: int = 20
    min_class_samples: int = 10

    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'modeldoctor.db').as_posix()}"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
