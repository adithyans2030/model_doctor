from typing import Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, model_validator

from app.api.deps import DbDep, SettingsDep
from app.schemas import Finding
from app.services import explanations as svc
from app.services.training import get_model

router = APIRouter(tags=["explainability"])


class ExplanationsOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    computed_at: str
    global_importance: dict[str, Any]
    error_analysis: dict[str, Any]
    fairness: dict[str, Any]
    findings: list[Finding]
    samples: list[dict[str, Any]]


class ExplainRequest(BaseModel):
    row_id: int | None = None
    record: dict[str, Any] | None = None
    method: Literal["auto", "shap", "lime"] = "auto"  # auto = SHAP if supported, else occlusion

    @model_validator(mode="after")
    def _one(self):
        if (self.row_id is None) == (self.record is None):
            raise ValueError("Provide exactly one of row_id or record")
        return self


class Contribution(BaseModel):
    feature: str
    value: str
    contribution: float


class ExplainOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    row_id: int | None
    in_test_sample: bool
    actual: str | float | None = None
    prediction: str | float
    confidence: float | None
    probabilities: dict[str, float] | None
    fidelity: float | None = None
    method: str
    units: str
    contributions: list[Contribution]
    explanation: str


@router.get("/models/{model_id}/explanations", response_model=ExplanationsOut)
def explanations(model_id: str, db: DbDep, settings: SettingsDep, refresh: bool = Query(False)):
    """Global importance (SHAP + permutation), error analysis and group performance on the test set."""
    model = get_model(db, model_id)
    e = svc.compute_explanations(db, settings, model, refresh=refresh)
    return {"model_id": model.id, "computed_at": e["computed_at"], "global_importance": e["global"],
            "error_analysis": e["error_analysis"], "fairness": e["fairness"], "findings": e["findings"],
            "samples": e["samples"]}


@router.post("/models/{model_id}/explain", response_model=ExplainOut)
def explain(model_id: str, body: ExplainRequest, db: DbDep, settings: SettingsDep):
    """Explain one prediction: a dataset row (`row_id`) or a new `record`."""
    model = get_model(db, model_id)
    return svc.explain_prediction(db, settings, model, row_id=body.row_id, record=body.record, method=body.method)
