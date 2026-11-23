# Phase 9: AI assistant (local LLMs via Ollama)

**Goal:** let users ask "Why is my recall so low?", "What should I fix
first?", "Compare version 1 and version 2" and get answers grounded in
ModelDoctor's actual analysis, **using the LLMs installed on your own machine**.
No API key is needed and no data leaves the machine.

## The architecture rule

> **The LLM interprets ModelDoctor's structured results. It never produces
> the diagnosis.**

```
stored analysis ──► context builder ──► CONTEXT (JSON) ──► local LLM (Ollama) ──► streamed answer + cited check ids
 (findings, metrics,     (question-focused,     │
  experiments, drift,     ~1k tokens)           ├──► Claude API (optional, if you set a key)
  explanations)                                 └──► rule engine (Ollama down / model missing / error)
```

## Providers

| `MD_ASSISTANT_MODE` | Behaviour |
|---|---|
| `auto` (default) | **Ollama** if it is running, else Claude if `ANTHROPIC_API_KEY` is set, else rules |
| `ollama` | always the local model; falls back to rules if Ollama is down, the model isn't installed, or it errors |
| `anthropic` | Claude (`claude-opus-5-5`); optional, kept for anyone with an API key |
| `rules` | deterministic answers only (the tests use this) |

### Models found on this machine (Ollama 0.40.16)

| Model | Size | Context | Measured with the compact context |
|---|---|---|---|
| `llama3.1:latest` **(default)** | 8B, Q4_K_M | 131k | about 75 s per answer when warm (CPU), 158 s on a cold start; best quality |
| `luttapi:latest` | 8B (llama3.1-based, custom) | 131k | similar to llama3.1 |
| `gemma2:2b` | 2.6B | 8k | first word in about 20 s, full answer about 22 s; faster but shallower |

Choose the model in the Assistant panel; the choice is remembered. You can
also set `MD_OLLAMA_MODEL` on the server or pass `llm_model` in the API
request. Short names are accepted: `llama3.1` resolves to
`llama3.1:latest`, and `gemma2` to `gemma2:2b` when that is the only
installed `gemma2` tag.

## Making local models practical on a CPU

The first attempt sent the full context (about 3,500 tokens) and timed out
after 180 s. Three changes fixed it:

1. **Compact, question-focused context** (`compact_context()`), about 960
   tokens:
   - it keeps up to 8 findings: those related to the question's topic first
     (recall, drift, overfitting, leakage, fairness, data), then the most
     severe others
   - evidence is reduced to a few headline numbers
   - model versions are included only when the question compares versions
2. **Bounded generation:** "under 150 words" in the prompt,
   `num_predict=400`, `num_ctx=4096`, `temperature=0.2` (keeps a small model
   close to the facts it was given).
3. **Streaming:** `POST /assistant/ask/stream` returns NDJSON events as the
   model generates. The 300 s timeout applies *between chunks*, so slow but
   steady generation never times out. `keep_alive: 15m` keeps the model
   loaded between questions, so only the first question pays the loading
   time.

### Warm-up (why the first answer used to take a minute)

Ollama unloads an idle model after a while, and loading an 8B model on a CPU
takes up to a minute. When the Assistant panel opens, the dashboard calls
`POST /assistant/warmup` for the selected model (a prompt-less Ollama
`generate` call that loads the model). It then polls
`/assistant/providers` (`loaded_models`) and shows "loading model into
memory…" until the model is ready. Measured on this machine with
`llama3.1`: the first word after **1.9 s** with the model warm, versus
**50 s** cold. The chat also shows a running seconds counter while it waits.

### Never a silent fallback

Before each answer the backend checks that Ollama is reachable. While Ollama
is loading a model it can be slow to respond, so the check allows 5 s, and
an "unreachable" result is cached for only 5 s (a successful model list is
cached for 30 s). Whenever the rules answer instead of a model, the
response's `note` says why (for example "Ollama was not reachable at …"),
and the UI shows it under the answer.

## What the model receives

- **System prompt:**
  - answer only from CONTEXT, quoting its numbers
  - cite check ids in brackets (`[model.weak_class]`)
  - if something is missing, say what, and which ModelDoctor action would
    produce it
  - respect "resolved" findings and prioritise by severity
- **User turn:** `CONTEXT: {...}` followed by `QUESTION: ...`, plus up to 6
  earlier turns.
- **Citations are verified:** only check ids that exist in the context are
  returned as `sources`, so an invented id never shows up as a source.

## The rule engine (always available)

If Ollama isn't running, the answer comes from deterministic rules over the
same context. They cover recall/false negatives, what to fix first,
comparing versions, drift, overfitting, leakage, experiments, features and
the health score. The response's `note` explains why the fallback was used.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/v1/assistant/providers` | active provider, whether Ollama is reachable, the default model, installed local models (name, parameters, size, context) |
| POST | `/api/v1/assistant/ask/stream` | NDJSON: `{"type":"meta","mode","model","context_model_id"}`, then `{"type":"delta","text"}`…, then `{"type":"done","sources","note"}` |
| POST | `/api/v1/assistant/ask` | the same, as one JSON response |
| POST | `/api/v1/assistant/warmup` | starts loading a local model into memory in the background; returns `loaded`, `loading` or `unavailable` |

Request body: `{dataset_id, model_id?, question, history?, llm_model?}`.

## Configuration

| Setting | Default |
|---|---|
| `MD_ASSISTANT_MODE` | `auto` |
| `MD_OLLAMA_URL` | `http://127.0.0.1:11434` (Docker: `http://host.docker.internal:11434`). Use 127.0.0.1 rather than `localhost`: on Windows, `localhost` tries IPv6 first and added about 2 s to every call |
| `MD_OLLAMA_MODEL` | `llama3.1:latest` |
| `MD_OLLAMA_NUM_CTX` / `MD_OLLAMA_MAX_TOKENS` | `4096` / `400` |
| `MD_OLLAMA_TEMPERATURE` | `0.2` |
| `MD_OLLAMA_TIMEOUT` | `300` s between streamed chunks |

In `docker compose` the API container reaches the Ollama running on your
machine through `host.docker.internal` (mapped with `extra_hosts`).

## Tests (`tests/test_assistant.py`)

No test needs Ollama or the internet. The Ollama HTTP API is replaced with
`httpx.MockTransport` (`FakeOllama`) and Claude with a fake client.

- Rules answers after a full pipeline, version comparison, validation.
- **Providers endpoint:** lists the local models, and reports Ollama as
  unavailable when it can't be reached.
- **Ollama request shape:**
  - the model name is resolved (`gemma2` → `gemma2:2b`)
  - streaming is on, with temperature and the token cap
  - the grounding system prompt is sent
  - the context is compact, with recall-related findings first for a recall
    question
  - invented citations are filtered out
- **Fallbacks:**
  - model not installed
  - Ollama returns a 500
  - Ollama unreachable
- **Stream endpoint:** the event sequence and NDJSON content type, and the
  rules fallback in the same event format.
- **Provider resolution:** local model, then Claude key, then rules; the
  explicit `rules` mode.
- **Claude path (optional):** request shape (fallbacks, cached system
  prompt), refusal and connection-error fallback.

**Verified against the real Ollama on this machine:**
- `/assistant/providers` lists all 3 models.
- A streamed answer from `gemma2:2b` arrived through the real API with CORS,
  correctly citing `[model.weak_class]`.
- `llama3.1` answered correctly with the compact context: it identified the
  leak and recommended removing it first.

## Limitations

- On CPU-only hardware, 8B models take about a minute per answer. A GPU, or
  a smaller model such as `gemma2:2b`, makes this much faster.
- Small local models follow the grounding rules less strictly than large
  hosted models. Verified citations, the low temperature and the compact
  context reduce this, but answers should still be read against the
  findings they cite.
- Conversations aren't stored; the browser sends the history.
