# 🩺 ModelDoctor

**An intelligent system for diagnosing, explaining and optimising machine-learning models.**

ModelDoctor examines a dataset and the models trained on it, works out *why* a model
underperforms, explains the root causes, recommends fixes, **runs the experiments to
check which fix actually works**, and hands the winner to a human for approval.

> Diagnoses come from deterministic ML and statistical analysis. The AI assistant (your local
> Ollama models by default) only explains those structured results; it never invents a diagnosis.

## What it finds (demo dataset)

`sample_data/churn_demo.csv` has 9 planted problems. ModelDoctor:

1. **Data Doctor** flags all 9: a *leaking* `cancellation_request` column
   (single-feature AUC 0.95), class imbalance (9.3:1), a constant column,
   missing values, outliers, duplicates, an ID column, irrelevant features and
   a correlated pair.
2. **Model Doctor** shows that the "98% accurate" baseline relies on that
   leak. Without it the model has **90% accuracy but only 12% churn recall**.
   It traces the cause to class imbalance and also flags overfitting and a
   sub-optimal threshold.
3. **Explainability** finds that 79% of missed churners are on month-to-month
   contracts (54% of test rows), along with the features that drive
   predictions (TreeSHAP) and the per-group gaps.
4. **Optimisation** removes the leak, then tests class weighting, SMOTE,
   deduplication, feature selection, hyperparameter search and the threshold.
   It keeps a step only if **cross-validation** improves. Result: churn recall
   **12% → 50%**, macro-F1 0.57 → 0.62.
5. **Registry:** the optimised model is a *candidate* until a person approves
   it, explicitly acknowledging any critical findings.
6. **Monitoring** checks production batches for feature, prediction and
   concept drift.
7. **The assistant** answers questions like "Why is my recall so low?" or
   "Compare version 1 and 2" from the stored results.

## Quick start (local)

```bash
# Backend
cd backend
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:create_app --factory --reload       # http://localhost:8000/docs
pytest                                                # 107 tests

# Dashboard (second terminal)
cd frontend
npm install
npm run dev                                           # http://localhost:3000
```

Run the whole pipeline headlessly and write JSON reports:

```bash
cd backend
python scripts/run_pipeline.py --csv ../sample_data/churn_demo.csv --target churned --out ../reports
```

## Full stack (Docker)

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API + OpenAPI docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard "ModelDoctor") | http://localhost:3001 |

The assistant uses the Ollama models installed on your machine (default `llama3.1`; pick another in the UI). Without Ollama running,
it uses rule-based answers built from the same results. A Claude API key is optional.

## Tech stack

| Area | Tools |
|---|---|
| API | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE), SciPy |
| Explainability | TreeSHAP (XGBoost), exact linear SHAP, `shap` (RF), permutation importance, occlusion |
| MLOps | model registry with approval gate, MLflow, DVC, Docker, Postgres |
| Monitoring | PSI / KS / chi-square drift, Prometheus metrics, Grafana |
| Security | API keys, rate limiting, audit log, upload sniffing, non-root containers |
| AI | Local LLMs via Ollama (llama3.1, gemma2), streamed; optional Claude; grounded on structured results with verified citations |
| Frontend | React 19, Vite 8, Tailwind CSS 4, React Router 7, TypeScript, hand-built SVG charts |
| CI | GitHub Actions: pytest, migration check (SQLite and Postgres), e2e pipeline, typecheck/build, Docker builds |

## Documentation

Start with [docs/README.md](docs/README.md). There is one document per phase,
plus [docs/architecture.md](docs/architecture.md).

| Phase | Scope |
|---|---|
| 1 | [Foundation & ingestion](docs/phase-1-foundation.md) |
| 2 | [Data Doctor](docs/phase-2-data-doctor.md) |
| 3 | [Model Doctor](docs/phase-3-model-doctor.md) |
| 4 | [Explainability & error analysis](docs/phase-4-explainability.md) |
| 5 | [Automatic optimisation](docs/phase-5-optimisation.md) |
| 6 | [Dashboard](docs/phase-6-dashboard.md) |
| 7 | [MLOps](docs/phase-7-mlops.md) |
| 8 | [Production: drift, metrics, security, CI/CD](docs/phase-8-production.md) |
| 9 | [AI assistant](docs/phase-9-assistant.md) |

## How it differs from AutoML, MLflow and dashboards

- **AutoML** finds the best model. ModelDoctor diagnoses *why* models fail,
  explains it, tests targeted fixes and tracks health over time. Model search
  is one component of it.
- **MLflow** tracks experiments and stores models. ModelDoctor uses it as
  infrastructure and adds diagnosis, root-cause analysis and automated
  corrective experiments.
- **A dashboard** shows "accuracy 92%". ModelDoctor says: "minority recall is
  only 54%. Likely cause: imbalance. Class weighting raised it to 79%.
  Promote v5 for validation."
