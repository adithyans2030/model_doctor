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

    # Training
    max_train_rows: int = 50_000  # larger labelled sets are stratified-sampled
    cv_folds: int = 5
    test_size: float = 0.2
    svm_max_rows: int = 10_000  # RBF SVM is O(n^2); skipped above this
    hpo_iterations: int = 12
    n_jobs: int = -1

    # MLOps (Phase 7): optional MLflow mirror of the built-in registry
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "ModelDoctor"

    # Security (Phase 8). Empty api_keys = open API (local development).
    api_keys: list[str] = []  # entries "name:key"
    rate_limit_per_minute: int = 300  # per actor / client IP; 0 disables
    metrics_require_auth: bool = False

    # AI assistant (Phase 9).
    # auto = local Ollama if running, else Claude if ANTHROPIC_API_KEY is set, else rule-based answers.
    assistant_mode: str = "auto"  # auto | ollama | anthropic | rules
    # 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and adds ~2 s per call.
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:latest"
    ollama_num_ctx: int = 4096  # local models get a compact (~1k token) context
    ollama_max_tokens: int = 400  # cap on generated tokens per answer
    ollama_temperature: float = 0.2
    ollama_timeout: float = 300.0  # max seconds between streamed chunks (first load can be slow)
    llm_model: str = "claude-opus-5-5"  # used only by the anthropic provider
    llm_effort: str = "medium"  # low | medium | high | xhigh | max

    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

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
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
