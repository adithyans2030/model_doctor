import json
import types

import httpx

import pytest

import app.assistant.engine as engine
from tests.conftest import API, train


@pytest.fixture
def diagnosed(client, churn_dataset):
    """Leaky baseline -> explanations -> experiment, i.e. a full diagnosis to talk about."""
    job = train(client, churn_dataset["id"], algorithms=["logistic_regression"], cv_folds=0)
    base = job["result"]["best_model_id"]
    r = client.post(f"{API}/models/{base}/experiments", json={"budget": "quick"})
    exp = client.get(f"{API}/jobs/{r.json()['id']}").json()["result"]
    best = exp["best_model_id"]
    client.get(f"{API}/models/{best}/explanations")
    return churn_dataset, base, best


def ask(client, dataset_id, question, **extra):
    r = client.post(f"{API}/assistant/ask", json={"dataset_id": dataset_id, "question": question, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_rules_answers_are_grounded(client, diagnosed):
    ds, base, best = diagnosed
    recall = ask(client, ds["id"], "Why is my recall so low?", model_id=best)
    assert recall["mode"] == "rules" and recall["context_model_id"] == best
    assert "recall" in recall["answer"].lower()

    first = ask(client, ds["id"], "What should I fix first?")
    assert "Recommended fixes" in first["answer"] or "Critical" in first["answer"]

    exp = ask(client, ds["id"], "What did the experiment find?", model_id=best)
    assert "Best configuration" in exp["answer"]

    feats = ask(client, ds["id"], "Which features drive the model?", model_id=best)
    assert "SHAP" in feats["answer"]

    health = ask(client, ds["id"], "What is the health score?")
    assert "/100" in health["answer"]

    drift = ask(client, ds["id"], "Is there drift?", model_id=best)
    assert "No drift report yet" in drift["answer"]


def test_compare_versions(client, diagnosed):
    ds, _, _ = diagnosed
    a = ask(client, ds["id"], "Compare model version 1 and version 2")
    assert "v1" in a["answer"] and "v2" in a["answer"] and "minority recall" in a["answer"]
    missing = ask(client, ds["id"], "compare v1 vs v9")
    assert "couldn't find" in missing["answer"]


def test_validation(client, churn_dataset):
    assert client.post(f"{API}/assistant/ask", json={"dataset_id": churn_dataset["id"], "question": ""}).status_code == 422
    assert client.post(f"{API}/assistant/ask", json={"dataset_id": "nope", "question": "hi"}).status_code == 404
    # Works before any model exists.
    assert "No model trained yet" in ask(client, churn_dataset["id"], "summary please")["answer"]


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response, self.error = response, error
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _response(text, stop="end_turn"):
    return types.SimpleNamespace(stop_reason=stop, model="claude-opus-5-5",
                                 content=[types.SimpleNamespace(type="text", text=text)])


def test_llm_mode_sends_grounded_context(client, diagnosed, settings, monkeypatch):
    ds, _, best = diagnosed
    fake = FakeClient(_response("Recall is low because of imbalance [model.weak_class] [made.up]."))
    monkeypatch.setattr(engine, "make_client", lambda: fake)
    settings.assistant_mode = "llm"
    out = ask(client, ds["id"], "Why is recall low?", model_id=best,
              history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert out["mode"] == "anthropic" and out["model"] == "claude-opus-5-5"
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in call["betas"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Answer only from CONTEXT" in call["system"][0]["text"]
    assert len(call["messages"]) == 3 and call["messages"][-1]["content"].startswith("CONTEXT:")
    assert '"findings"' in call["messages"][-1]["content"]
    # Only citations that exist in the context are reported.
    assert "made.up" not in out["sources"]


def test_llm_refusal_and_errors_fall_back_to_rules(client, diagnosed, settings, monkeypatch):
    import anthropic

    ds, _, best = diagnosed
    settings.assistant_mode = "llm"
    monkeypatch.setattr(engine, "make_client", lambda: FakeClient(_response("", stop="refusal")))
    out = ask(client, ds["id"], "What should I fix first?")
    assert out["mode"] == "rules" and "declined" in out["note"]

    err = anthropic.APIConnectionError(request=types.SimpleNamespace(method="POST", url="x"))
    monkeypatch.setattr(engine, "make_client", lambda: FakeClient(error=err))
    out = ask(client, ds["id"], "What should I fix first?")
    assert out["mode"] == "rules" and "could not reach" in out["note"]


# ----------------------------------------------------------------------------- local models (Ollama)
TAGS = {"models": [
    {"name": "llama3.1:latest", "size": 4_920_000_000,
     "details": {"parameter_size": "8.0B", "family": "llama", "context_length": 131072}},
    {"name": "gemma2:2b", "size": 1_600_000_000,
     "details": {"parameter_size": "2.6B", "family": "gemma2", "context_length": 8192}},
]}


class FakeOllama:
    """Stands in for the Ollama HTTP API via httpx.MockTransport."""

    def __init__(self, chunks=("Recall is low because of ", "class imbalance [model.weak_class] [made.up]."),
                 up=True, chat_status=200):
        self.chunks, self.up, self.chat_status = chunks, up, chat_status
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        body = json.loads(request.content)
        self.requests.append(body)
        if self.chat_status != 200:
            return httpx.Response(self.chat_status, text="model failed to load")
        lines = [json.dumps({"message": {"role": "assistant", "content": c}, "done": False}) for c in self.chunks]
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
        return httpx.Response(200, text="\n".join(lines) + "\n")

    def install(self, monkeypatch):
        engine._OLLAMA_CACHE.clear()
        monkeypatch.setattr(engine, "_http", lambda: httpx.Client(transport=httpx.MockTransport(self.handler)))
        return self


@pytest.fixture(autouse=True)
def _clear_ollama_cache():
    engine._OLLAMA_CACHE.clear()
    yield
    engine._OLLAMA_CACHE.clear()


def test_providers_lists_local_models(client, settings, monkeypatch):
    FakeOllama().install(monkeypatch)
    settings.assistant_mode = "auto"
    p = client.get(f"{API}/assistant/providers").json()
    assert p["provider"] == "ollama" and p["ollama_available"] is True
    assert [m["name"] for m in p["local_models"]] == ["llama3.1:latest", "gemma2:2b"]
    assert p["local_models"][0]["parameters"] == "8.0B"

    FakeOllama(up=False).install(monkeypatch)
    p = client.get(f"{API}/assistant/providers").json()
    assert p["ollama_available"] is False and p["local_models"] == []


def test_ollama_answer_uses_compact_grounded_context(client, diagnosed, settings, monkeypatch):
    ds, _, best = diagnosed
    fake = FakeOllama().install(monkeypatch)
    settings.assistant_mode = "ollama"
    out = ask(client, ds["id"], "Why is recall low?", model_id=best, llm_model="gemma2")
    assert out["mode"] == "ollama", out["note"]
    assert out["mode"] == "ollama" and out["model"] == "gemma2:2b"  # ":latest"-style name resolution
    assert out["answer"].startswith("Recall is low because") and out["sources"] == ["model.weak_class"]
    req = fake.requests[0]
    assert req["model"] == "gemma2:2b" and req["stream"] is True
    assert req["options"]["temperature"] == settings.ollama_temperature
    assert req["options"]["num_predict"] == settings.ollama_max_tokens
    system, user = req["messages"][0], req["messages"][-1]
    assert system["role"] == "system" and "Answer only from CONTEXT" in system["content"]
    ctx = json.loads(user["content"].split("CONTEXT:\n", 1)[1].split("\n\nQUESTION:")[0])
    assert len(ctx["findings"]) <= 8 and "health" in ctx
    # The question is about recall, so recall-related findings come first.
    assert ctx["findings"][0]["check"] in {"model.weak_class", "data.class_imbalance", "model.misleading_accuracy",
                                           "model.suboptimal_threshold", "prediction.fn_concentration"}
    assert len(user["content"]) < 8000  # compact: a fraction of the full context


def test_ollama_failures_fall_back_to_rules(client, diagnosed, settings, monkeypatch):
    ds, _, _ = diagnosed
    settings.assistant_mode = "ollama"
    FakeOllama().install(monkeypatch)
    out = ask(client, ds["id"], "What should I fix first?", llm_model="mistral")
    assert out["mode"] == "rules" and "not installed" in out["note"]

    FakeOllama(chat_status=500).install(monkeypatch)
    out = ask(client, ds["id"], "What should I fix first?")
    assert out["mode"] == "rules" and "Ollama error 500" in out["note"]

    FakeOllama(up=False).install(monkeypatch)
    out = ask(client, ds["id"], "What should I fix first?")
    assert out["mode"] == "rules" and "not reachable" in out["note"]


def _stream(client, dataset_id, question, **extra):
    with client.stream("POST", f"{API}/assistant/ask/stream",
                       json={"dataset_id": dataset_id, "question": question, **extra}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in r.iter_lines() if line]


def test_stream_endpoint(client, diagnosed, settings, monkeypatch):
    ds, _, best = diagnosed
    FakeOllama().install(monkeypatch)
    settings.assistant_mode = "auto"
    events = _stream(client, ds["id"], "Why is recall low?", model_id=best)
    assert [e["type"] for e in events] == ["meta", "delta", "delta", "done"]
    assert events[0]["mode"] == "ollama" and events[0]["model"] == "llama3.1:latest"
    assert events[0]["context_model_id"] == best
    assert "".join(e["text"] for e in events if e["type"] == "delta").startswith("Recall is low")
    assert events[-1]["sources"] == ["model.weak_class"]

    # Ollama down -> the rule engine answers in the same event format.
    FakeOllama(up=False).install(monkeypatch)
    engine._OLLAMA_CACHE.clear()
    settings.assistant_mode = "ollama"
    events = _stream(client, ds["id"], "What should I fix first?")
    assert events[0]["mode"] == "rules" and "not reachable" in events[-1]["note"]


def test_provider_resolution(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    settings.assistant_mode = "auto"
    FakeOllama(up=False).install(monkeypatch)
    assert engine.resolve_provider(settings) == "rules"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert engine.resolve_provider(settings) == "anthropic"
    engine._OLLAMA_CACHE.clear()
    FakeOllama().install(monkeypatch)
    assert engine.resolve_provider(settings) == "ollama"  # local model wins
    settings.assistant_mode = "rules"
    assert engine.resolve_provider(settings) == "rules"
