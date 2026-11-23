# Phase 10: Extended features

This phase closes the gaps listed after the first nine phases:
- regression
- more algorithms
- LIME
- bring-your-own-model
- user accounts
- frontend tests in the repo
- a DVC pipeline that has actually been run
- deployable Kubernetes manifests

## 1. Regression

Every phase now handles continuous targets as well as classification.

| Phase | Regression behaviour |
|---|---|
| Target (Ph. 1) | A numeric target with non-integer values or more than `MD_MAX_INTEGER_TARGET_CLASSES` distinct values becomes `regression`. The dataset stores a `target_summary` (mean, quartiles, skewness, a histogram, whether all values are non-negative), and a **skewed target** produces a warning. |
| Data Doctor (Ph. 2) | **Leakage probe:** a single-feature decision-tree **regressor's** cross-validated R² (warning ≥ 0.90, critical ≥ 0.95). **Relevance:** `mutual_info_regression`. **New check:** `data.skewed_target` (\|skew\| ≥ 1), which suggests the `target_transform` fix. |
| Model Doctor (Ph. 3) | Split stratified on target **deciles**, plain K-fold CV, primary metric **R²**. Also reports RMSE, MAE, median AE, MAPE (when the target has no zeros), explained variance, residual statistics, a residual histogram and a predicted-vs-actual sample. **Checks:** overfitting (R² gap), high variance, no skill (R² ≤ 0.05), underfitting, `model.heteroscedastic_errors` (\|error\| correlates with the prediction, r ≥ 0.3), `model.outlier_errors` (the worst 5% of rows cause ≥ 25% of the error), and suspicious performance (R² ≥ 0.99) linked to leakage. |
| Health score | Performance = 100 × max(0, R²). Robustness uses the R² gap and CV standard deviation. Fairness = 100 − 60 × (largest gap in group MAE ÷ overall MAE). |
| Explainability (Ph. 4) | SHAP on the prediction (TreeSHAP for XGBoost/LightGBM/RF, exact linear SHAP for Ridge). Permutation importance on R². **Error slices** by mean absolute error, and **large-error concentration** (where the worst 10% of errors fall). Per-group MAE and R². |
| Optimiser (Ph. 5) | Plan: drop leaks → **log-transform the target** (`TransformedTargetRegressor` with log1p/expm1; only when the target is non-negative) → deduplicate → feature selection → hyperparameter search. Each step is accepted by **CV R²**. |
| Registry (Ph. 7) | Comparison on R², RMSE, MAE, median AE, MAPE and explained variance, with lower-is-better handled correctly. Models of different tasks can't be compared. |
| Monitoring (Ph. 8) | Prediction drift = PSI of the **predicted values**. Concept drift = R² drop on a labelled batch. |
| Assistant (Ph. 9) | Context and rules know the task; error questions ("why are the errors large?") are answered from the regression findings. |

**Demo dataset:** `sample_data/house_prices_demo.csv` (`make_demo_data.py --kind house_prices`), with planted problems:
- a log-normal, skewed price
- a leaking `assessed_value`, i.e. a tax assessment recorded *after* the sale
- missing values, duplicates, an ID column and a constant column
- an irrelevant `listing_weekday`
- a noisier neighbourhood ("Hills")

**Results on it:**
- The leak is flagged as critical, and every experiment removes it.
- Without the leak, the true R² is 0.84.
- The fairness check finds the "Hills" neighbourhood rather than the random weekday column.

### Two bugs found while building it
- **SVR scored R² −0.04 on house prices.** SVR depends on the scale of the target, and prices are in the hundreds of thousands. SVR now standardises its target internally (`TransformedTargetRegressor` with `StandardScaler`): **R² 0.98**.
- **A false fairness warning.** The first threshold (a 30% MAE gap) flagged a random weekday column, because seven small groups vary that much by chance. The thresholds are now 0.5 for a warning and 1.0 for critical.

## 2. More algorithms: LightGBM and a PyTorch neural network

`app/ml/algorithms.py` is now a per-task catalogue:

| Key | Classification | Regression | Notes |
|---|---|---|---|
| `logistic_regression` | ✓ | | |
| `ridge` | | ✓ | |
| `random_forest` | ✓ | ✓ | |
| `xgboost` | ✓ | ✓ | |
| `lightgbm` | ✓ | ✓ | optional import; exact TreeSHAP via `pred_contrib` |
| `svm` | ✓ (Platt-calibrated SVC) | ✓ (SVR with scaled target) | |
| `pytorch_mlp` | ✓ | ✓ | opt-in; offered only if `torch` is installed |

`GET /api/v1/algorithms?task=…` lists what's available and what's trained by default. The PyTorch model (`app/ml/torch_models.py`) is a small MLP with a scikit-learn interface:
- Adam optimiser, dropout and weight decay.
- Classification supports balanced sample weights; regression trains on a standardised target.
- It sits inside the same pipeline as every other algorithm, so CV, permutation importance, LIME and hyperparameter search all work on it.

PyTorch is in `requirements-ml-extra.txt` rather than the Docker image, because it adds about 2 GB for one algorithm.

**Not added: TensorFlow and Hugging Face.** ModelDoctor models tabular data. TensorFlow would duplicate the PyTorch network, and Hugging Face models target text and images, which ModelDoctor doesn't model.

## 3. LIME

`POST /models/{id}/explain` takes `"method": "auto" | "shap" | "lime"`. LIME is implemented directly in `explain.py` rather than through the `lime` package:
1. Perturb the row by replacing random subsets of its features with the **typical training value** (median, or the most common category).
2. Weight each perturbation with an exponential kernel on the share of features replaced.
3. Fit a weighted ridge regression of the model output (P(class), or the predicted value) on "feature kept" indicators.
4. The coefficients are the contributions. The weighted R² of the fit is returned as **fidelity**.

**Validation:** on a linear model LIME must reproduce the exact linear SHAP values, and it does, with fidelity 1.0; a test checks this.

**A design note:** the first version replaced features with values from random training rows, and its fidelity was only about 0.1. That randomness is noise the linear surrogate can't explain. Switching to the typical-value baseline (the same baseline occlusion and linear SHAP use) fixed it.

## 4. Bring your own model (safe)

`POST /datasets/{id}/models/upload` (a multipart `.skops` file, plus an optional `name`):

1. **Pickle and joblib files are refused.** Loading them runs arbitrary code.
2. For a **skops** file, `skops.io.get_untrusted_types` lists every type the file would create. The file is **refused before loading** unless all of them come from scikit-learn, XGBoost, LightGBM, imbalanced-learn, NumPy, SciPy or builtins. A file carrying a custom class (the `Evil` test fixture) is rejected and the class named in the error. A pickle renamed to `.skops` is rejected as "Not a valid skops file".
3. The model must accept a DataFrame of the dataset's usable feature columns. `ExternalModel` maps its classes (for example 0/1 integers) onto ModelDoctor's labels and reorders the probability columns.
4. It's evaluated on ModelDoctor's **same held-out split**, diagnosed, explained (permutation, occlusion, LIME) and registered as `source: "uploaded"`. An info finding notes that ModelDoctor can't know whether the model saw the test rows during training.
5. Experiments are refused for uploaded models, since they work by retraining. The raw upload is deleted after loading.

## 5. User accounts

| Piece | Implementation |
|---|---|
| Passwords | **Argon2id** (`argon2-cffi`), rehashed on login if the parameters change. Minimum 8 characters. |
| Tokens | **JWT HS256** (PyJWT) with `sub`, `role`, `exp` and issuer `modeldoctor`; 12 h by default. The key comes from `MD_SECRET_KEY`, or is generated once and stored in `data/.secret_key`. |
| Endpoints | `GET /auth/status`, `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `GET /auth/users` (admin), `PATCH /auth/users/{id}` (admin: role, active) |
| Roles | **viewer** (read-only, plus the assistant), **member** (upload, train, experiment), **admin** (also approve and promote models, manage users) |
| Enforcement | The gateway accepts API keys (service accounts, treated as admin) **or** user Bearer tokens. With `MD_AUTH_REQUIRED=true`, or any API keys configured, everything except health, docs, metrics and the auth endpoints needs one. The user is **re-read on each request**, so deactivation and role changes take effect at once. Viewers get 403 on writes, and registry approval or promotion is admin-only. |
| Safety | The first account becomes admin. `MD_OPEN_REGISTRATION=false` means only admins can add users. Unknown email and wrong password give the same message, so accounts can't be discovered. Admins can't demote or deactivate themselves. |
| Dashboard | A login/register page (it says when the first account will become admin), the user and role in the top bar, an admin **Users** page, and approve/promote disabled for non-admins. |

The default local setup stays open (no sign-in), as before. The Kubernetes configuration turns sign-in on.

## 6. Frontend tests in the repo

| Suite | What it covers |
|---|---|
| **Vitest** (`npm test`, 8 tests) | The Markdown renderer, including that model output can't inject HTML. The NDJSON stream reader reassembling events split across network chunks. API error mapping and the unreachable-backend message. Number formatting. The error boundary containing a crash and retrying. |
| **Playwright** (`npm run test:e2e`, 3 tests) | Starts its **own** backend (fresh temporary database, rule-based assistant) and Vite server on separate ports, then drives Chrome through: the **classification workflow** (create project → upload → target → Data Doctor shows the leak → train → performance → SHAP and **LIME** explanations → experiment → **approve** → assistant answer); the **regression workflow** (target summary, R² leaderboard, predicted-vs-actual chart, error explanations); and the 404 page. Every test fails on any page error, console error or HTTP 5xx. |

CI runs both, plus the backend tests, migrations on SQLite and Postgres, the headless pipeline on both demo datasets, Docker builds and a check that the Kubernetes manifests render.

## 7. DVC, actually run

`dvc init` was run, and `dvc repro` executes four stages:
- `make_demo_data` → `diagnose` (churn)
- `make_house_data` → `diagnose_regression`

`dvc metrics show` then compares runs:

```
Path                             task            primary   baseline  leak-free  optimised  extra                    health
reports/metrics.json             classification  f1_macro  0.9496    0.5207     0.6474     minority recall 0.4898   54
reports/regression/metrics.json  regression      r2        0.9977    0.8396     0.8396     RMSE 198,275             69
```

On the regression data, no experiment step (log target, deduplication, tuning) beat the leak-free Ridge baseline on CV R², so the optimiser kept the baseline and said so. That's the intended behaviour: a step that doesn't help is rejected.

`dvc.lock` records the exact inputs and outputs. Change a parameter in `params.yaml` and `dvc repro` reruns only the stages affected.

## 8. Cloud deployment (Kubernetes + image publishing)

- **`.github/workflows/release.yml`:** pushing a `v*` tag builds both images and publishes them to **GHCR** (`ghcr.io/<owner>/model_doctor-api` / `-web`), tagged with the version, the commit sha and `latest`. The web image is built with `VITE_API_URL=""`, so it calls `/api` on its own host.
- **`deploy/k8s/`** (Kustomize, `kubectl apply -k deploy/k8s`):
  - namespace, ConfigMap (sign-in required, closed registration)
  - Secret created by hand; the command is in `config.yaml`, and no secrets are in the repo
  - Postgres StatefulSet with a persistent volume and readiness probe
  - API Deployment: non-root user, all capabilities dropped, no privilege escalation, health probes, resource limits, data volume, Prometheus scrape annotations
  - web Deployment (2 replicas)
  - an Ingress serving `/api` and `/` on one host with TLS, and settings for the 50 MB upload limit and streamed answers

**Verified:** the manifests render with `kubectl kustomize`, which CI also runs.

**Not verified:** there's no Kubernetes cluster on this machine, and publishing needs the workflow to run on GitHub. The API stays at **one replica**, because background jobs and the rate limiter run in-process (a known limitation; see Phase 8).

## Tests added in this phase

| File | Tests |
|---|---|
| `tests/test_regression.py` | 10: target and Data Doctor, algorithm catalogue and validation, training metrics and diagnosis, SHAP/LIME/error analysis (LIME = linear SHAP check), LIME on a classifier, the optimiser's regression plan, prediction/comparison/drift, the assistant, PyTorch on both tasks |
| `tests/test_uploads.py` | 3: an uploaded classifier evaluated, explained and compared (experiments refused); an uploaded regressor; unsafe and wrong files refused |
| `tests/test_auth.py` | 5: Argon2 hashing; first user is admin, plus the login flow and no account enumeration; viewer/member/admin rules including immediate deactivation; closed registration; open mode still working |
| `frontend/src/**/*.test.ts(x)` | 8 Vitest tests |
| `frontend/e2e/workflow.spec.ts` | 3 Playwright browser tests |

Backend total: **128 tests**, all passing.
