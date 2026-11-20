import json
from typing import Literal

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import DbDep, SettingsDep
from app.assistant.context import build_context
from app.assistant.engine import (
    AssistantUnavailable, answer, answer_stream, choose_local_model, loaded_local_models, ollama_models,
    resolve_provider, warm_up,
)
from app.services.datasets import get_dataset

router = APIRouter(tags=["assistant"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    model_id: str | None = None
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    llm_model: str | None = Field(default=None, max_length=200, description="Local Ollama model to use")


class AskResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    mode: Literal["ollama", "anthropic", "rules"]
    model: str | None
    sources: list[str]
    note: str | None
    context_model_id: str | None


class ProvidersOut(BaseModel):
    provider: Literal["ollama", "anthropic", "rules"]
    ollama_url: str
    ollama_available: bool
    default_model: str
    local_models: list[dict]
    loaded_models: list[str]


class WarmupRequest(BaseModel):
    llm_model: str | None = Field(default=None, max_length=200)


class WarmupOut(BaseModel):
    status: Literal["loaded", "loading", "unavailable"]
    model: str | None
    detail: str | None = None


@router.get("/assistant/providers", response_model=ProvidersOut)
def providers(settings: SettingsDep):
    """Which LLM provider the assistant will use, the local models installed in Ollama,
    and which of them are already loaded in memory (ready to answer quickly)."""
    models = ollama_models(settings, fresh=True)
    return {
        "provider": resolve_provider(settings),
        "ollama_url": settings.ollama_url,
        "ollama_available": models is not None,
        "default_model": settings.ollama_model,
        "local_models": models or [],
        "loaded_models": loaded_local_models(settings) if models is not None else [],
    }


@router.post("/assistant/warmup", response_model=WarmupOut)
def warmup(body: WarmupRequest, background: BackgroundTasks, settings: SettingsDep):
    """Start loading a local model into memory so the first answer begins sooner.

    Returns immediately; loading continues in the background. Poll `/assistant/providers`
    (`loaded_models`) to see when it's ready.
    """
    if resolve_provider(settings) != "ollama":
        return {"status": "unavailable", "model": None, "detail": "The local (Ollama) provider isn't active."}
    try:
        chosen = choose_local_model(settings, body.llm_model)
    except AssistantUnavailable as exc:
        return {"status": "unavailable", "model": None, "detail": str(exc)}
    if chosen in loaded_local_models(settings):
        return {"status": "loaded", "model": chosen}
    background.add_task(warm_up, settings, chosen)
    return {"status": "loading", "model": chosen}


@router.post("/assistant/ask/stream")
def ask_stream(body: AskRequest, db: DbDep, settings: SettingsDep):
    """Like `/assistant/ask`, but streams the answer as NDJSON events (`meta`, `delta`..., `done`).

    Local models are slow on CPU; streaming shows the answer as it is generated.
    """
    dataset = get_dataset(db, body.dataset_id)
    ctx = build_context(db, settings, dataset, body.model_id)  # built before streaming starts
    context_model_id = (ctx.get("selected_model") or {}).get("id")
    history = [t.model_dump() for t in body.history]

    def events():
        for event in answer_stream(settings, body.question, ctx, history, body.llm_model):
            if event["type"] == "meta":
                event["context_model_id"] = context_model_id
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/assistant/ask", response_model=AskResponse)
def ask(body: AskRequest, db: DbDep, settings: SettingsDep):
    """Ask a question about a dataset's diagnosis. Answers are grounded in ModelDoctor's own results."""
    dataset = get_dataset(db, body.dataset_id)
    ctx = build_context(db, settings, dataset, body.model_id)
    result = answer(settings, body.question, ctx, [t.model_dump() for t in body.history], body.llm_model)
    return {**result, "context_model_id": (ctx.get("selected_model") or {}).get("id")}
