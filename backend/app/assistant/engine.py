"""Phase 9: the AI assistant.

Architecture rule: **the LLM interprets ModelDoctor's structured results; it
never produces the diagnosis.** The context is built from stored analysis
(``context.py``), the system prompt forbids claims that aren't in it, and every
answer reports which finding ids it cited.

Providers (``MD_ASSISTANT_MODE``):
* ``auto``      - local Ollama if it is running, else Claude if an Anthropic key is set, else rules
* ``ollama``    - a local model served by Ollama (default model ``MD_OLLAMA_MODEL``)
* ``anthropic`` - Claude via the Anthropic API (needs ``ANTHROPIC_API_KEY``)
* ``rules``     - deterministic answers only (no network)

Any provider failure falls back to the rule engine, with a note explaining why.
"""

import json
import logging
import os
import re
import time
from collections.abc import Iterator

import httpx

from app.assistant.context import compact_context
from app.assistant.rules import rule_answer
from app.core.config import Settings

log = logging.getLogger("modeldoctor.assistant")

SYSTEM_PROMPT = """You are ModelDoctor's assistant. You help machine-learning engineers understand the
diagnosis of their dataset and models.

You will receive a JSON document called CONTEXT. It contains the only facts you may use: computed
metrics, findings (each with a stable "check" id, severity, evidence, cause and recommendation),
recommendations, models and their versions, experiment results, explanations and drift reports.

Rules:
- Answer only from CONTEXT. Never invent metrics, features, findings or results. If CONTEXT does not
  contain what is needed, say what is missing and which ModelDoctor action would produce it
  (train models, compute explanations, run an experiment, upload a monitoring batch).
- Quote the actual numbers from CONTEXT when they support the answer.
- When a statement relies on a finding, cite its check id in square brackets, e.g. [model.weak_class].
- Findings marked "resolved" do not affect the selected model; say so if relevant.
- Prioritise by severity (critical, then warning, then info) when asked what to do.
- Be concise and practical: a short direct answer first, then supporting detail as a brief list.
- Plain text or simple Markdown lists; no tables, no headings."""

LOCAL_SUFFIX = "\n- Keep the answer under 150 words."


class AssistantUnavailable(Exception):
    pass


# ----------------------------------------------------------------------------- Ollama (local)
_OLLAMA_CACHE: dict[str, tuple[float, list[dict] | None]] = {}
_CACHE_SECONDS = 30


def _http() -> httpx.Client:
    """Separated so tests can inject a mock transport."""
    return httpx.Client()


def ollama_models(settings: Settings, *, fresh: bool = False) -> list[dict] | None:
    """Installed local models, or None if Ollama isn't reachable. Cached briefly."""
    url = settings.ollama_url.rstrip("/")
    cached = _OLLAMA_CACHE.get(url)
    if cached and not fresh and time.monotonic() - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        with _http() as client:
            r = client.get(f"{url}/api/tags", timeout=2.0)
        r.raise_for_status()
        models = [
            {
                "name": m["name"],
                "size_gb": round(m.get("size", 0) / 1e9, 1),
                "parameters": (m.get("details") or {}).get("parameter_size"),
                "family": (m.get("details") or {}).get("family"),
                "context_length": (m.get("details") or {}).get("context_length"),
            }
            for m in r.json().get("models", [])
        ]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        models = None
    _OLLAMA_CACHE[url] = (time.monotonic(), models)
    return models


def _choose_local_model(settings: Settings, model: str | None) -> str:
    installed = ollama_models(settings)
    if installed is None:
        raise AssistantUnavailable(f"Ollama is not reachable at {settings.ollama_url}")
    names = [m["name"] for m in installed]
    chosen = model or settings.ollama_model
    if chosen not in names and f"{chosen}:latest" in names:
        chosen = f"{chosen}:latest"
    if chosen not in names:
        # Accept the bare model name when exactly one installed tag matches (e.g. "gemma2" -> "gemma2:2b").
        tagged = [n for n in names if n.split(":", 1)[0] == chosen]
        if len(tagged) == 1:
            chosen = tagged[0]
    if chosen not in names:
        raise AssistantUnavailable(
            f"model '{chosen}' is not installed in Ollama (installed: {', '.join(names) or 'none'})")
    return chosen


def _local_messages(question: str, ctx: dict, history: list[dict]) -> list[dict]:
    # Local CPU inference is slow and prompt length dominates latency, so local models get a
    # question-focused slice of the context (~1k tokens) and a word limit.
    small = compact_context(ctx, question)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + LOCAL_SUFFIX}]
    messages += [{"role": h["role"], "content": h["content"]} for h in history[-6:]]
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{json.dumps(small, default=str, separators=(',', ':'))}\n\nQUESTION:\n{question}",
    })
    return messages


def _ollama_stream(settings: Settings, model: str, messages: list[dict]) -> Iterator[str]:
    """Yield text chunks from Ollama's streaming chat API (one JSON object per line)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Low temperature keeps a small model close to the facts it was given.
        "options": {"temperature": settings.ollama_temperature, "num_ctx": settings.ollama_num_ctx,
                    "num_predict": settings.ollama_max_tokens},
        "keep_alive": "15m",
    }
    # The read timeout applies between chunks, so slow-but-progressing generation never times
    # out; loading a model into memory the first time can take over a minute.
    timeout = httpx.Timeout(settings.ollama_timeout, connect=5.0)
    try:
        with _http() as client, client.stream(
            "POST", f"{settings.ollama_url.rstrip('/')}/api/chat", json=payload, timeout=timeout
        ) as r:
            if r.status_code >= 400:
                r.read()
                raise AssistantUnavailable(f"Ollama error {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise AssistantUnavailable(f"Ollama error: {event['error']}")
                chunk = (event.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk
                if event.get("done"):
                    return
    except httpx.TimeoutException as exc:
        raise AssistantUnavailable(f"the local model sent nothing for {settings.ollama_timeout:.0f}s") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise AssistantUnavailable(f"Ollama error: {exc}") from exc


def _ask_ollama(settings: Settings, question: str, ctx: dict, history: list[dict],
                model: str | None) -> tuple[str, str]:
    chosen = _choose_local_model(settings, model)
    text = "".join(_ollama_stream(settings, chosen, _local_messages(question, ctx, history))).strip()
    if not text:
        raise AssistantUnavailable("the local model returned no text")
    return text, chosen


# ----------------------------------------------------------------------------- Anthropic (optional)
def make_client():
    """Separated so tests can substitute a fake client."""
    import anthropic

    return anthropic.Anthropic()


def _anthropic_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _ask_claude(settings: Settings, question: str, ctx: dict, history: list[dict]) -> tuple[str, str]:
    import anthropic

    client = make_client()
    messages = [{"role": h["role"], "content": h["content"]} for h in history[-10:]]
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{json.dumps(ctx, default=str, separators=(',', ':'))}\n\nQUESTION:\n{question}",
    })
    try:
        response = client.beta.messages.create(
            model=settings.llm_model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": settings.llm_effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines, the API retries on a fallback model
            messages=messages,
        )
    except anthropic.APIConnectionError as exc:
        raise AssistantUnavailable(f"could not reach the Anthropic API ({exc.__class__.__name__})") from exc
    except anthropic.APIStatusError as exc:
        raise AssistantUnavailable(f"Anthropic API error {exc.status_code}") from exc

    if response.stop_reason == "refusal":
        raise AssistantUnavailable("the model declined this request")
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise AssistantUnavailable("the model returned no text")
    return text, getattr(response, "model", settings.llm_model)


# ----------------------------------------------------------------------------- routing
def resolve_provider(settings: Settings) -> str:
    mode = settings.assistant_mode
    if mode == "llm":  # backwards-compatible alias
        mode = "anthropic"
    if mode != "auto":
        return mode
    if ollama_models(settings) is not None:
        return "ollama"
    if _anthropic_configured():
        return "anthropic"
    return "rules"


def _cited(text: str, ctx: dict) -> list[str]:
    known = {f["check"] for f in ctx["findings"]}
    return sorted({c for c in re.findall(r"\[([a-z_]+\.[a-z_]+)\]", text) if c in known})


def _rules(question: str, ctx: dict, note: str | None) -> dict:
    text, sources = rule_answer(question, ctx)
    return {"answer": text, "mode": "rules", "model": None, "sources": sorted(set(sources)), "note": note}


def answer(settings: Settings, question: str, ctx: dict, history: list[dict] | None = None,
           model: str | None = None) -> dict:
    history = history or []
    provider = resolve_provider(settings)
    note = None
    if provider in {"ollama", "anthropic"}:
        try:
            if provider == "ollama":
                text, used = _ask_ollama(settings, question, ctx, history, model)
            else:
                text, used = _ask_claude(settings, question, ctx, history)
            return {"answer": text, "mode": provider, "model": used, "sources": _cited(text, ctx), "note": None}
        except AssistantUnavailable as exc:
            log.warning("assistant (%s) falling back to rules: %s", provider, exc)
            note = f"AI unavailable ({exc}); answered from the diagnosis rules instead."
        except ImportError:
            note = "The anthropic package is not installed; answered from the diagnosis rules instead."
    return _rules(question, ctx, note)


def answer_stream(settings: Settings, question: str, ctx: dict, history: list[dict] | None = None,
                  model: str | None = None) -> Iterator[dict]:
    """Streaming variant of ``answer``. Yields events:

        {"type": "meta", "mode": ..., "model": ...}
        {"type": "delta", "text": "..."}            (one or more)
        {"type": "done", "sources": [...], "note": ...}

    Only Ollama streams token by token; other providers send their answer as one delta.
    If the local model fails before producing any text, the rule engine answers instead.
    """
    history = history or []
    if resolve_provider(settings) != "ollama":
        result = answer(settings, question, ctx, history, model)
        yield {"type": "meta", "mode": result["mode"], "model": result["model"]}
        yield {"type": "delta", "text": result["answer"]}
        yield {"type": "done", "sources": result["sources"], "note": result["note"]}
        return

    try:
        chosen = _choose_local_model(settings, model)
        stream = _ollama_stream(settings, chosen, _local_messages(question, ctx, history))
        first = next(stream, None)  # surfaces connection/model errors before committing to "ollama"
        if first is None:
            raise AssistantUnavailable("the local model returned no text")
    except AssistantUnavailable as exc:
        log.warning("assistant (ollama) falling back to rules: %s", exc)
        r = _rules(question, ctx, f"AI unavailable ({exc}); answered from the diagnosis rules instead.")
        yield {"type": "meta", "mode": "rules", "model": None}
        yield {"type": "delta", "text": r["answer"]}
        yield {"type": "done", "sources": r["sources"], "note": r["note"]}
        return

    yield {"type": "meta", "mode": "ollama", "model": chosen}
    parts = [first]
    yield {"type": "delta", "text": first}
    note = None
    try:
        for chunk in stream:
            parts.append(chunk)
            yield {"type": "delta", "text": chunk}
    except AssistantUnavailable as exc:
        note = f"The local model stopped early ({exc})."
    yield {"type": "done", "sources": _cited("".join(parts), ctx), "note": note}
