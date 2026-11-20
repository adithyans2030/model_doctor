"""Pydantic request and response models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

InferredType = Literal[
    "numeric", "categorical", "boolean", "datetime", "text", "id", "empty"
]
TaskType = Literal["binary_classification", "multiclass_classification"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime
    dataset_count: int = 0


class ColumnProfile(BaseModel):
    name: str
    pandas_dtype: str
    inferred_type: InferredType
    n_missing: int
    missing_pct: float
    n_unique: int
    numeric_stored_as_text: bool = False
    sample_values: list[str]


class ClassCount(BaseModel):
    label: str
    count: int
    share: float


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    original_filename: str
    n_rows: int
    n_cols: int
    target_column: str | None
    task_type: TaskType | None
    created_at: datetime


class DatasetOut(DatasetSummary):
    size_bytes: int
    sha256: str
    encoding: str
    delimiter: str
    columns: list[ColumnProfile]
    n_classes: int | None
    class_distribution: list[ClassCount] | None


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    n_rows_shown: int
    total_rows: int


class TargetRequest(BaseModel):
    column: str = Field(min_length=1, max_length=255)


class TargetResponse(BaseModel):
    dataset: DatasetOut
    warnings: list[str]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "error"]


# ----------------------------------------------------------------------------- Phase 2+
Severity = Literal["critical", "warning", "info"]


class Finding(BaseModel):
    check: str
    category: str
    severity: Severity
    title: str
    evidence: dict[str, Any] = {}
    cause: str | None = None
    impact: str | None = None
    recommendation: str | None = None
    interventions: list[str] = []
    columns: list[str] = []
    resolved: bool = False


class Penalty(BaseModel):
    check: str
    severity: Severity
    points: int
    title: str


class Score(BaseModel):
    value: int
    penalties: list[Penalty]


class DataReportOut(BaseModel):
    id: str
    dataset_id: str
    created_at: datetime
    summary: dict[str, Any]
    score: Score
    findings: list[Finding]
    column_stats: list[dict[str, Any]]
    feature_target: list[dict[str, Any]]
    correlations: list[dict[str, Any]]


# ----------------------------------------------------------------------------- Phase 3
AlgorithmKey = Literal["logistic_regression", "random_forest", "xgboost", "svm"]
JobStatus = Literal["queued", "running", "completed", "failed"]
Stage = Literal["candidate", "approved", "production", "rejected", "archived"]


class TrainRequest(BaseModel):
    algorithms: list[AlgorithmKey] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost", "svm"], min_length=1
    )
    cv_folds: int = Field(5, ge=0, le=10, description="0 disables cross-validation")
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    drop_columns: list[str] = []
    seed: int = 42

    @field_validator("algorithms")
    @classmethod
    def _unique(cls, v):
        return list(dict.fromkeys(v))

    @field_validator("cv_folds")
    @classmethod
    def _folds(cls, v):
        if v == 1:
            raise ValueError("cv_folds must be 0 (disabled) or at least 2")
        return v


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    kind: str
    status: JobStatus
    params: dict[str, Any]
    result: dict[str, Any] | None
    progress: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    project_id: str
    dataset_id: str
    job_id: str | None
    name: str
    algorithm: str
    version: int
    source: str
    stage: Stage
    created_at: datetime
    scores: dict[str, Any]
    n_findings: int


class ModelOut(ModelSummary):
    config: dict[str, Any]
    feature_spec: dict[str, Any]
    metrics: dict[str, Any]
    findings: list[Finding]
    importances: list[dict[str, Any]] | None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    predictions: list[Prediction]


class DiagnosisOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    model: dict[str, Any] | None
    health: dict[str, Any]
    severity_counts: dict[str, int]
    findings: list[Finding]
    recommendations: list[dict[str, Any]]
    narrative: list[str]
    latest_experiment: dict[str, Any] | None
