# Phase 8: Production (drift monitoring, metrics, security, CI/CD)

**Goal:** know when a deployed model stops matching reality, observe the
service, and harden the API.

## What was built

| Piece | File |
|---|---|
| Drift statistics and findings | `backend/app/monitoring/drift.py` |
| Batch monitoring use case | `backend/app/services/monitoring.py` |
| Prometheus metrics (no extra dependency) | `backend/app/monitoring/metrics.py` |
| Gateway: auth, rate limit, audit, metrics, headers | `backend/app/core/security.py`, table `audit_logs` |
| API | `backend/app/api/routes/monitoring.py` |
| Prometheus and Grafana | `deploy/prometheus/`, `deploy/grafana/` (provisioned datasource + dashboard) |
| CI | `.github/workflows/ci.yml` |

## Drift monitoring

`POST /models/{id}/monitor` takes a CSV production batch. The reference is
the model's own training split (rebuilt exactly, as in Phase 4).

| Signal | Method | Severity |
|---|---|---|
| Numeric feature shift | PSI over 10 training-quantile bins + two-sample **Kolmogorov–Smirnov** test | PSI ≥ 0.10 moderate (warning), ≥ 0.25 major (critical) |
| Categorical shift | PSI over training categories (top 30 + "other") + **chi-square** test | same |
| Missing-rate shift | change in missing share | ≥ 10 points (warning) |
| Unseen categories | share of batch values never seen in training | ≥ 5% (warning) |
| Prediction drift | PSI of the predicted-class mix, batch vs test | same PSI bands |
| **Concept drift** | if the batch has labels: macro-F1 on the batch vs the test set | drop ≥ 0.05 warning, ≥ 0.10 critical |
| Schema change | columns the dataset never had | info |

Missing feature columns are rejected (`400 schema_mismatch`, listing them).
Columns the model excludes by design (IDs, constants, dropped leaks) don't
count as schema changes.

**Drift score** = 100 × (1 − mean(min(PSI / 0.5, 1))) − 200 × (macro-F1 drop
if labelled), clipped to 0–100. It feeds the Drift component of the health
score, and the drift findings join the diagnosis.

**Privacy:** the raw batch file is deleted straight after analysis; only the
computed report is stored.

## Observability

`GET /metrics` serves the Prometheus text format, written without a
client-library dependency:

- `modeldoctor_http_requests_total{method,route,status}` and
  `modeldoctor_http_request_duration_seconds` (histogram), labelled by
  **route template** (`/api/v1/models/{model_id}`), so ids don't explode
  label cardinality.
- `modeldoctor_jobs_total{kind,status}` (finished jobs).
- Read from the database at scrape time, so the values are correct with
  several workers: `modeldoctor_models{stage}`, `modeldoctor_jobs{kind,status}`,
  `modeldoctor_model_f1_macro{model,stage}` and
  `modeldoctor_drift_score{model}` for approved/production models.

The Grafana dashboard (provisioned automatically) shows drift score,
production F1, models by stage, request rate, p95 latency per route, 5xx
rate and finished jobs.

## Security

One middleware (`GatewayMiddleware`) handles:

| Control | Behaviour |
|---|---|
| **Authentication** | Optional. `MD_API_KEYS='["alice:<key>","ci:<key>"]'` makes every route except health/docs require `X-API-Key` or `Authorization: Bearer`. Keys are compared in constant time; the key's name becomes the actor. `/metrics` stays public for scraping unless `MD_METRICS_REQUIRE_AUTH=true`. |
| **Rate limiting** | Sliding 60 s window per actor (or client IP): `MD_RATE_LIMIT_PER_MINUTE` (default 300; 0 disables). `429 rate_limited` with `Retry-After`. |
| **Audit log** | Every POST/PUT/PATCH/DELETE under `/api/` is stored (actor, method, path, status, IP), including failed attempts. `GET /api/v1/audit-logs?actor=`. Audit failures never break requests. |
| **Security headers** | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`. |
| CORS | Allowed origins come from `MD_CORS_ORIGINS`; CORS wraps the gateway, so 401/429 responses still carry CORS headers. |

Together with the earlier phases, the security picture is:
- **Uploads (Phase 1):** content sniffing, size/row/column caps, UUID
  storage, no model uploads.
- **Model files (Phase 3):** only bundles ModelDoctor wrote itself are ever
  loaded.
- **Containers (Phase 7):** run as non-root users, with secrets only from
  the environment.
- **Batches (this phase):** raw production files aren't retained.

## CI/CD (`.github/workflows/ci.yml`)

| Job | Steps |
|---|---|
| `backend` | install, **pytest**, `alembic upgrade head` + `alembic check` (migrations match the models), the **end-to-end pipeline** on the demo data, upload the reports as an artifact |
| `backend-postgres` | the same migrations against a real **Postgres 16** service, then downgrade to base |
| `frontend` | `npm ci`, `tsc --noEmit`, `vite build` |
| `docker` | validate the Compose file and build both images |

Deployment to a cloud provider isn't included. The images are ready to push
to a registry; a deploy job would only need the registry and target
credentials.

## API

| Method | Path |
|---|---|
| POST | `/api/v1/models/{id}/monitor` (multipart `file`) |
| GET | `/api/v1/models/{id}/drift?limit=20` |
| GET | `/api/v1/audit-logs?limit=&actor=` |
| GET | `/metrics` |

## Tests (`tests/test_monitoring.py`)

- PSI properties: identical = 0, small shift < 0.1, big shift > 0.25.
- **Stable batch** (same generator, new seed): score ≥ 85, no critical
  findings.
- **Shifted batch:**
  - charges × 1.6 and a forced contract mix → major numeric and categorical
    drift (KS and chi-square)
  - an all-missing column → missing-rate shift
  - an extra column → schema change
  - flipped labels → concept drift with a macro-F1 drop > 0.1
  - score below 60, the batch file deleted, history stored, and the drift
    component and findings in the diagnosis
- Missing features rejected.
- Metrics: route templates, counters, histogram, stage and job gauges.
- Auth: health public, 401 without or with a wrong key, both header styles
  work, metrics public by default.
- The audit log records the authenticated actor, including failed attempts.
- Rate limiting: the 9th request with a limit of 8 gets 429 with
  Retry-After; another actor has a separate budget. Security headers present.
- The registry uses the authenticated actor, not the self-declared one.
- Key parsing and limiter helpers.

## Limitations

- The rate limiter and request counters live in each process. Several API
  workers would need Redis for shared limits (the DB-derived gauges are
  already worker-safe).
- Drift is measured per uploaded batch; streaming or scheduled monitoring
  would call the same service from a scheduler.
- API keys are a simple service-to-service scheme; user accounts with OIDC
  would come next.
