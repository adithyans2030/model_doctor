# Phase 7: MLOps (registry, human approval, MLflow, DVC, Docker)

**Goal:** make results reproducible and deployable, and put a person between
"the optimiser found something better" and "it's in production".

## What was built

| Piece | File |
|---|---|
| Registry with an approval gate, events, comparison | `backend/app/services/registry.py`, `api/routes/registry.py`, table `model_events` |
| MLflow mirror (optional) | `backend/app/services/tracking.py` |
| Headless pipeline runner | `backend/scripts/run_pipeline.py` |
| DVC pipeline | `dvc.yaml`, `params.yaml` |
| Containers | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| Migrations | `0004_registry_audit` (also contains Phase 8's `audit_logs`) |

## Registry and human-in-the-loop

```
candidate ──approve──► approved ──promote──► production ──► archived
    │                      │
    └──reject──► rejected  └──reject──► rejected        (rejected/archived can be reopened)
```

Rules (`TRANSITIONS` in `registry.py`):

1. **No skipping approval.** `candidate → production` returns `409
   invalid_transition`: "Approve it first."
2. **A note is required** to approve, reject or promote (`note_required`).
3. **Critical findings block approval.** If the model (or its explanation
   findings) has unresolved critical findings, approval fails with `409
   unresolved_critical_findings` listing them. The reviewer must send
   `acknowledge_risks: true`, and the acknowledgement is recorded in the
   event note.
4. **One production model per project.** Promoting archives the previous
   one, with the event "Replaced by …".
5. **Every transition is an event** (`model_events`: from, to, actor, note,
   time). The actor is the **authenticated API-key name** when auth is on
   (Phase 8). A self-declared `actor` field is ignored in that case, so
   nobody can approve under someone else's name.

The diagnosis endpoint prefers the production model when one exists.

### Comparing versions

`GET /models/compare/{a}/{b}` compares two models trained on the same
dataset. It returns per-metric deltas (and which side is better; for ECE and
log-loss lower is better), minority recall, the config diff, findings
resolved or introduced, the winner, and a plain-English summary. It warns if
the models weren't evaluated on the same test split. The Phase 9 assistant's
"compare version 4 and version 5" uses the same data.

## MLflow

ModelDoctor's own registry (in the database) is the source of truth. MLflow
is an optional mirror for experiment browsing. With `MD_MLFLOW_TRACKING_URI`
set (Compose sets `http://mlflow:5000`):

- each trained model → an MLflow run with params, test metrics and
  `diagnosis.json`
- each optimisation → a parent run with one nested run per trial (with a
  `kept` tag)

Tracking failures are logged and never fail a job. The production image
installs `mlflow-skinny` (client only); the Compose `mlflow` service is the
server.

## Reproducibility: headless pipeline and DVC

`scripts/run_pipeline.py` runs the **entire product** in-process against a
throwaway database: upload → target → Data Doctor → baselines → explanations
→ experiments → diagnosis. It writes `metrics.json`, `diagnosis.json`,
`data_report.json` and `experiment.json`. It goes through the real API code,
so it doubles as an end-to-end test (CI runs it).

```bash
python backend/scripts/run_pipeline.py --csv sample_data/churn_demo.csv --target churned --out reports
```

Result on the demo data (quick budget, 12 s):

```json
{ "data_quality_score": 34, "baseline_f1_macro": 0.9496, "leak_free_baseline_f1_macro": 0.5207,
  "optimised_f1_macro": 0.6376, "optimised_minority_recall": 0.5102, "health_score": 54 }
```

`dvc.yaml` chains `make_demo_data → diagnose`, with parameters in
`params.yaml` (demo rows/seed, target, budget). `dvc repro` regenerates
everything and `dvc metrics diff` compares runs (for example after changing
a threshold in the doctor).

## Containers

- **API image:** `python:3.12-slim` plus `libgomp1` (XGBoost), runs as a
  non-root user (uid 10001), data in a `/data` volume, with a healthcheck. On
  start it runs `alembic upgrade head`, then uvicorn.
- **Web image:** a two-stage build: Node 22 builds the Vite app, and nginx
  serves the static files, falling back to `index.html` for client-side
  routes, with security headers and long caching for hashed assets.
- **docker-compose.yml:** Postgres 16 (with healthcheck), the MLflow server,
  the API (wired to Postgres and MLflow), the dashboard, Prometheus and
  Grafana. Secrets come from `.env` (`.env.example` provided).

```bash
cp .env.example .env
docker compose up --build
# dashboard :3000 · API :8000/docs · MLflow :5000 · Prometheus :9090 · Grafana :3001
```

## Postgres

The models use only portable types (String UUID keys, JSON, timezone-aware
DateTime), and every migration was generated with Alembic autogenerate and
checked with `alembic check` (no drift between models and migrations). CI
applies the migrations to a real Postgres 16 service and downgrades them
again.

## Tests (`tests/test_registry.py`, plus the MLflow test in `test_optimisation.py`)

- The approval gate: no skipping, notes required, critical findings blocking
  until acknowledged, promotion archiving the previous model, the event
  trail, and the diagnosis preferring production.
- Invalid transitions and reopening.
- Comparison deltas and config diff; cross-dataset comparison rejected.
- The MLflow integration, using a fake module.

## What was not verified locally

Docker Desktop wasn't running in this environment, so the images weren't
built here. `docker compose config` validates the Compose file, and CI
builds both images. DVC isn't installed locally; the stages call the same
script that was run directly.
