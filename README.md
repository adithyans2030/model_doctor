# 🩺 ModelDoctor

**An intelligent system for diagnosing, explaining and optimising machine-learning models.**

ModelDoctor examines a dataset and the models trained on it, finds out *why* a model
underperforms or behaves unreliably, explains the root causes, recommends fixes and
runs experiments to check which fix actually works.

> Diagnoses come from deterministic ML and statistical analysis. The LLM assistant
> (Phase 9) only explains those structured results; it never invents a diagnosis.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: FastAPI, project structure, dataset ingestion | ✅ Done |
| 2 | Data Doctor: missing values, duplicates, outliers, imbalance, leakage | ⏳ Next |
| 3 | Model Doctor: baselines, CV, over/underfitting, per-class metrics | |
| 4 | Explainability: SHAP, error analysis | |
| 5 | Auto-optimisation: SMOTE, class weights, HPO, feature selection | |
| 6 | Dashboard: Next.js + TypeScript | |
| 7 | MLOps: MLflow, model registry, DVC, Docker, Postgres | |
| 8 | Production: drift monitoring, Prometheus/Grafana, CI/CD | |
| 9 | AI assistant: LLM over structured diagnostics | |

## Quick start

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

alembic upgrade head                                # creates data/modeldoctor.db
uvicorn app.main:create_app --factory --reload      # http://localhost:8000/docs
```

Run the tests with `pytest`.

### Try it with the demo dataset

`sample_data/churn_demo.csv` is a synthetic churn dataset with **planted problems**
(class imbalance, missing values, duplicates, outliers, a constant column, a leaky
feature, a correlated pair, and a region with worse outcomes). Later phases are
tested against this known ground truth. Regenerate it with
`python scripts/make_demo_data.py`.

```bash
curl -X POST localhost:8000/api/v1/projects -H "content-type: application/json" -d "{\"name\":\"Churn\"}"
curl -F "file=@../sample_data/churn_demo.csv" localhost:8000/api/v1/projects/<project_id>/datasets
curl -X POST localhost:8000/api/v1/datasets/<dataset_id>/target -H "content-type: application/json" -d "{\"column\":\"churned\"}"
```

## API (v1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness + database check |
| POST / GET | `/api/v1/projects` | Create / list projects |
| GET / DELETE | `/api/v1/projects/{id}` | Get / delete a project (and its files) |
| POST | `/api/v1/projects/{id}/datasets` | Upload a CSV and get its schema profile |
| GET | `/api/v1/projects/{id}/datasets` | List a project's datasets |
| GET / DELETE | `/api/v1/datasets/{id}` | Dataset details / delete |
| GET | `/api/v1/datasets/{id}/preview?rows=20` | First rows (max 100) |
| POST | `/api/v1/datasets/{id}/target` | Choose the target, infer the task type, check usability |

Errors always have the shape `{"error": {"code": "...", "message": "..."}}`.

### What happens on upload

1. **File checks.** Only `.csv` is accepted. The content is sniffed, so Excel, ZIP,
   PDF, Parquet, pickle and other binary files are rejected even if renamed. The
   upload is streamed to disk with a size cap and stored under a random UUID name,
   so the user's filename is used for display only.
2. **Format detection.** Encoding (UTF-8, UTF-8 with BOM, or a detected fallback
   such as Latin-1) and delimiter (`,` `;` tab `|`).
3. **Structural checks.** At least 2 columns, no duplicate headers, and row and
   column counts within the configured limits.
4. **Schema profile.** Each column gets an inferred type (`numeric`, `categorical`,
   `boolean`, `datetime`, `text`, `id`, `empty`), plus missing counts, cardinality
   and samples. Numbers stored as text are flagged.

### What happens when a target is set

The column is rejected if it is an ID, free text, a date, empty, constant, has
too few labelled rows, or has too many classes. A continuous numeric target is
reported as regression, which isn't supported yet. Otherwise the endpoint
returns the task type (binary or multiclass), the class distribution and
warnings: missing target values, labels that differ only by case, and classes
too small to evaluate reliably.

## Architecture (backend)

```
backend/
  app/
    main.py              app factory (create_app), CORS, routers
    core/config.py       settings from MD_* env vars / .env
    core/errors.py       domain errors → consistent JSON errors
    db/                  SQLAlchemy models (Project, Dataset), engine
    schemas.py           Pydantic request/response models
    storage.py           streamed, size-capped, UUID-named file storage
    services/
      ingestion.py       validation, encoding/delimiter detection, safe parsing
      profiling.py       column type inference and schema profile
      target.py          target validation and task-type inference
      datasets.py        use cases (ingest, preview, set target, delete)
    api/routes/          health, projects, datasets
  alembic/               database migrations
  scripts/               demo data generator
  tests/                 pytest suite
sample_data/             demo CSV with planted issues
```

## Security notes

- No uploaded content is ever executed. Model files (pickle/joblib) are not
  accepted, because unpickling runs arbitrary code. ModelDoctor trains its own models.
- File type, size, row and column limits are all configurable (`backend/.env.example`).
- Stored filenames are random, which rules out path traversal through upload names.
- Planned: authentication and per-user dataset isolation, rate limiting and audit
  logs (Phase 8).

## Configuration

Copy `backend/.env.example` to `backend/.env`. Every setting has a default.
`MD_DATABASE_URL` switches from SQLite to Postgres.
