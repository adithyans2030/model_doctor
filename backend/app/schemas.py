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
