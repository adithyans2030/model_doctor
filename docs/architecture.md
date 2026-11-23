# Architecture

## Components

```
            ┌──────────────────────────────┐
            │  Dashboard (React+Vite, :3000)│
            └──────────────┬───────────────┘
                           │ REST (JSON, X-API-Key)
            ┌──────────────▼───────────────┐
            │  FastAPI  /api/v1  (:8000)    │  GatewayMiddleware: auth · rate limit ·
            │                               │  audit log · metrics · security headers
            └──┬────────┬────────┬────────┬─┘
               │        │        │        │
   ┌───────────▼┐ ┌─────▼─────┐ ┌▼───────┐ ┌▼──────────────┐
   │ Ingestion  │ │ Data      │ │ Model  │ │ Monitoring     │
   │ & profiling│ │ Doctor    │ │ Engine │ │ (drift)        │
   └─────┬──────┘ └─────┬─────┘ └───┬────┘ └──────┬────────┘
         └──────────────┴───────────┼─────────────┘
                                    ▼
                     ┌───────────────────────────┐
                     │  Findings (one format)     │──► Health score, narrative,
                     │  check · severity ·        │    recommendations
                     │  evidence · cause ·        │
                     │  interventions             │
                     └───────┬───────────┬────────┘
                             │           │
              ┌──────────────▼──┐   ┌────▼──────────────────┐
              │ Explainability  │   │ Optimisation engine    │
              │ SHAP · slices   │   │ interventions → trials │
              └──────────────┬──┘   └────┬──────────────────┘
                             │           ▼
                             │   ┌────────────────────────┐
                             │   │ Registry (human gate)  │──► MLflow (optional mirror)
                             │   └────────────┬───────────┘
                             ▼                ▼
                ┌──────────────────────────────────────────┐
                │ SQLite / Postgres: projects, datasets,    │
                │ reports, jobs, models, model_events,      │
                │ audit_logs     +  files: uploads/, models/│
                └──────────────────────────────────────────┘
                             ▲
              ┌──────────────┴──────────┐
              │ AI assistant (Ollama)    │ reads the structured results only
              └──────────────────────────┘

  Prometheus scrapes /metrics ──► Grafana dashboard
```

## Data flow for one dataset

```
upload CSV → validate & profile → choose target → Data Doctor (findings, score)
  → train baselines (job) → Model Doctor (findings, health)
  → explanations (SHAP, error slices, fairness)
  → experiments (job: plan from findings, greedy trials chosen by CV)
  → candidate model → human approval → production
  → monitor batches (drift, concept drift) → health updates
  → ask the assistant at any step
```

## Key design decisions

1. **Deterministic diagnosis, LLM for language.** Every claim comes from
   computed statistics in the shared *finding* format. The assistant only
   interprets those, and its citations are checked against the context.
2. **Findings drive fixes.** Each finding names `interventions` from a fixed
   catalogue. The optimiser reads them, which links "what's wrong" directly to
   "what we tried" and "what worked".
3. **Honest evaluation.**
   - The primary metric is macro-F1, so a model can't score well by
     predicting the majority class.
   - The split is fixed and reproducible.
   - Experiments are accepted or rejected by cross-validation on training
     data; the test set is only for reporting.
   - Thresholds are tuned on out-of-fold predictions.
4. **Exact reproducibility.** Phases 4, 5 and 8 rebuild a model's exact
   train/test split from the stored seed, test size and target. No row
   indices need to be stored.
5. **Transparent scores.** Data quality = 100 − listed penalties. Each health
   component shows its formula and inputs, and weights are renormalised over
   the components that have been measured.
6. **Human in the loop.**
   - Nothing reaches production without approval.
   - Critical findings must be explicitly acknowledged.
   - Every transition is audited with the authenticated actor.
7. **Safe by default.**
   - No uploaded code or model files; only bundles ModelDoctor wrote itself
     are ever loaded.
   - Uploads are content-sniffed and size-capped, stored under UUID names.
   - Production batches aren't retained.
   - Containers run as non-root users; secrets come only from the
     environment.
8. **Seams for scale.**
   - `run_job(session_factory, settings, job_id, worker)` can move to
     Celery or RQ unchanged.
   - Moving from SQLite to Postgres is a URL change.
   - DB-derived metrics stay correct with several workers.

## Repository layout

```
backend/
  app/
    api/routes/      one router per area (projects, datasets, reports, models, explain,
                     experiments, registry, monitoring, assistant, health)
    assistant/       context builder (full + compact), Ollama / Claude engine, rule engine
    core/            config, errors, security gateway
    db/              SQLAlchemy models, engine
    diagnostics/     findings, data_doctor, model_doctor, health, narrative
    ml/              features, algorithms, pipeline, training, evaluation, explain,
                     error_analysis, optimize, artifacts
    monitoring/      drift, metrics
    services/        use cases (datasets, reports, training, explanations, experiments,
                     registry, monitoring, jobs, tracking, diagnosis)
  alembic/           migrations 0001–0005
  scripts/           make_demo_data.py, run_pipeline.py
  tests/             128 pytest tests (+ 8 Vitest, 3 Playwright)
frontend/            React + Vite + Tailwind dashboard
deploy/              Prometheus + Grafana provisioning
docs/                this documentation
sample_data/         demo dataset with planted problems
dvc.yaml, params.yaml, docker-compose.yml, .github/workflows/ci.yml
```
