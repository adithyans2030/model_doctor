# ModelDoctor documentation

Read in order; each phase builds on the previous ones.

| Phase | Document | In one line |
|---|---|---|
| — | [architecture.md](architecture.md) | How the pieces fit, the data flow, and the key design decisions |
| 1 | [phase-1-foundation.md](phase-1-foundation.md) | Safe CSV ingestion, schema profiling, target validation |
| 2 | [phase-2-data-doctor.md](phase-2-data-doctor.md) | Data and feature health checks, the finding format, the data-quality score |
| 3 | [phase-3-model-doctor.md](phase-3-model-doctor.md) | Reproducible baselines, imbalance-aware metrics, model diagnosis, health score, background jobs |
| 4 | [phase-4-explainability.md](phase-4-explainability.md) | SHAP and permutation importance, local explanations, error slices, group performance |
| 5 | [phase-5-optimisation.md](phase-5-optimisation.md) | Diagnosis → plan → greedy experiments chosen by CV → candidate model |
| 6 | [phase-6-dashboard.md](phase-6-dashboard.md) | React + Vite + Tailwind dashboard for the whole workflow |
| 7 | [phase-7-mlops.md](phase-7-mlops.md) | Registry with human approval, MLflow, DVC, Docker, Postgres |
| 8 | [phase-8-production.md](phase-8-production.md) | Drift monitoring, Prometheus/Grafana, auth, rate limiting, audit log, CI/CD |
| 9 | [phase-9-assistant.md](phase-9-assistant.md) | Grounded AI assistant on your local Ollama models (streamed), with optional Claude and a rule-based fallback |

Each document lists what was built (with file paths), how it works, the
thresholds used, the API, results on the demo dataset, the tests that cover
it, and known limitations.
