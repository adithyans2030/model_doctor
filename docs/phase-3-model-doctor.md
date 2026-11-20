# Phase 3: Model Doctor

**Goal:** train baseline models reproducibly, evaluate them with metrics that
hold up under class imbalance, diagnose why a model under-performs, and
combine everything into the Model Health Score.

## What was built

| Piece | File |
|---|---|
| Algorithm catalogue (defaults and search spaces) | `backend/app/ml/algorithms.py` |
| `TrainConfig` and the pipeline built from it | `backend/app/ml/pipeline.py` |
| Split, cross-validation, fit, evaluate, importances | `backend/app/ml/training.py` |
| Metrics, calibration (ECE), threshold analysis | `backend/app/ml/evaluation.py` |
| Model bundle save/load | `backend/app/ml/artifacts.py` |
| Model diagnostic checks | `backend/app/diagnostics/model_doctor.py` |
| Health score | `backend/app/diagnostics/health.py` |
| Plain-English summary and recommendations | `backend/app/diagnostics/narrative.py` |
| Background jobs | `backend/app/services/jobs.py`, table `jobs` |
| Training and prediction use cases | `backend/app/services/training.py`, table `models` |
| Combined diagnosis | `backend/app/services/diagnosis.py` |
| API | `backend/app/api/routes/models.py` (migration `0003_models`) |

## The training pipeline

```
labelled rows ──(stratified sample if > MD_MAX_TRAIN_ROWS)──► prepare_frame(spec)
      │
      └─ stratified 80/20 split (seed 42) ─► train ─┬─ 5-fold stratified CV (out-of-fold probabilities)
                                                    └─ fit on all train ─► evaluate on test (fixed)
```

- **Feature spec.** ID, text, empty and constant columns are excluded
  automatically. Dates become "days since 1970". Numbers stored as text are
  converted to numbers. The spec is saved with the model and replayed at
  prediction time.
- **Preprocessing (`ColumnTransformer`).**
  - Numeric columns: median imputation plus a missing-value indicator, then
    standard scaling.
  - Categorical columns: most-frequent imputation, then one-hot encoding.
    Categories seen fewer than 5 times are grouped as "infrequent" (at most 30
    per column), and unseen categories are handled safely.
- **Algorithms:** Logistic Regression, Random Forest, XGBoost, and an RBF SVM.
  The SVM gets probabilities from Platt scaling via
  `CalibratedClassifierCV(SVC(), ensemble=False)`, because scikit-learn 1.9
  deprecated `SVC(probability=True)`. The SVM is skipped above
  `MD_SVM_MAX_ROWS` training rows, since its cost grows with the square of the
  row count.
- **Positive class.** For binary targets the minority class is the "positive"
  class. It is used for PR-AUC, recall and threshold analysis.
- **Primary metric:** macro-F1, which can't be inflated by predicting the
  majority class. Accuracy is still reported.
- **Reproducibility.** The split is a deterministic function of (data, seed,
  test_size). Experiments in Phase 5 rebuild the same test set, so every
  comparison uses identical held-out rows.

### One configuration object for everything

```python
TrainConfig(algorithm="xgboost", params={}, class_weight=None, resampling=None,
            drop_columns=[], dedupe=False, threshold=None, calibration=None)
```

Baselines use the defaults. Each optimisation trial changes one or more
fields. Algorithms without a `class_weight` option (XGBoost) receive balanced
`sample_weight` instead. SMOTE runs inside an imblearn pipeline, so it only
touches the training folds and never sees validation or test data.

## Metrics stored per model (`metrics.test`)

Accuracy, balanced accuracy, macro and weighted F1, macro precision and
recall, ROC AUC (binary, or one-vs-rest macro), PR-AUC, log-loss, Brier score,
**ECE** with the reliability curve, per-class precision/recall/F1/support, the
worst class, the confusion matrix, and for binary targets a threshold sweep
(0.05–0.95). Also stored: CV mean ± std for train and validation folds, a
majority-class dummy baseline, split info and fit time.

## Diagnostic checks

| Check | Rule | Intervention |
|---|---|---|
| `model.overfitting` | CV train F1 − validation F1 ≥ 0.10 (critical ≥ 0.20) | hyperparameter tuning |
| `model.high_variance` | CV F1 std ≥ 0.05 | |
| `model.no_skill` | balanced accuracy ≤ dummy + 0.05 (critical) | class weights |
| `model.underfitting` | train and validation F1 both < 0.65 | hyperparameter tuning |
| `model.weak_class` | worst-class F1 < 0.60 and ≥ 0.25 below the best class; critical if recall < 50% | class weights, SMOTE, threshold (if imbalanced) |
| `model.misleading_accuracy` | accuracy − balanced accuracy ≥ 0.10 | |
| `model.poor_calibration` | ECE ≥ 0.08 | calibration |
| `model.suboptimal_threshold` | best-F1 threshold beats 0.5 by ≥ 0.03 F1 | threshold tuning |
| `model.suspicious_performance` / `model.dominant_feature` | macro-F1 ≥ 0.98, AUC ≥ 0.995, or one feature ≥ 50% of importance; critical if it is a known leak | remove the leak |

**Linking causes.** A model symptom names the data finding that explains it.
For example, `model.weak_class` reads `data.class_imbalance` and reports
"Class imbalance (ratio 9.26): the model sees few '1' examples…". This is the
root-cause step that separates a diagnosis from a dashboard.

**Resolved findings.** If the selected model doesn't use a column (for
example, the leaky one was dropped), the leakage and low-relevance findings
for that column are kept but marked `resolved` and downgraded to info. That
way the diagnosis doesn't recommend a fix that's already been applied.

## Model Health Score

| Component | Weight | Formula |
|---|---|---|
| Data | 0.25 | Data Doctor score |
| Performance | 0.30 | 100 × test macro-F1 |
| Robustness | 0.20 | 100 − 150×overfit gap − 300×CV std − 100×ECE |
| Fairness | 0.15 | 100 − 150×largest group gap in balanced accuracy (Phase 4) |
| Drift | 0.10 | latest drift score (Phase 8) |

Components that haven't been measured are `null` and the weights are
renormalised over the rest. Each component's `explanation` field shows its
formula with the actual numbers.

## Background jobs

`POST /datasets/{id}/train` returns `202` with a job, and training runs in
FastAPI's background thread pool. Poll `GET /jobs/{id}`; the `progress` field
shows "Training XGBoost (3/4)". Jobs still running when the server restarts
are marked failed at the next startup. `run_job(session_factory, settings,
job_id, worker)` doesn't depend on HTTP, so it can be moved to Celery or RQ
without changing the training code.

## Safe model storage

Bundles are written with joblib under `<model_id>.joblib`, and the file name
is derived from the id. joblib/pickle executes code when it loads a file, so
ModelDoctor **only loads files it wrote itself** and never accepts uploaded
model files.

## API

| Method | Path |
|---|---|
| POST | `/api/v1/datasets/{id}/train` with body `{algorithms, cv_folds, test_size, drop_columns, seed}` |
| GET | `/api/v1/jobs/{id}`, `/api/v1/datasets/{id}/jobs?kind=training` |
| GET | `/api/v1/datasets/{id}/models`, `/api/v1/models/{id}` |
| POST | `/api/v1/models/{id}/predict` with body `{"records": [...]}` (max 1000) |
| GET | `/api/v1/datasets/{id}/diagnosis[?model_id=]`: health, findings, recommendations, narrative |

## Results on the demo dataset

**With the leaky column**, all models look excellent (Logistic Regression
F1 0.95, accuracy 98%). The doctor reports *critical: one feature
('cancellation_request') drives 53% of the model*, linked to the Data
Doctor's leakage finding.

**After dropping `cancellation_request`** (`drop_columns`):

```
XGBoost v3       F1 0.57  accuracy 90%  balanced 55%  churn recall 12%
critical model.weak_class      recall 12%; likely cause: class imbalance (ratio 9.26)
critical model.overfitting     train F1 exceeds validation F1 by 0.36
warning  model.misleading_accuracy   accuracy 90% vs balanced accuracy 55%
warning  model.suboptimal_threshold  0.10 gives F1 0.33 vs 0.20 at 0.5
health 43  (data 34, performance 57, robustness 35)
```

This is the case from the project brief: about 90% accuracy hiding a model
that misses almost every churner. Phase 5 tries to fix it.

## Tests (`tests/test_model_doctor.py`)

- Metric unit tests: confusion matrix, thresholds, ECE.
- Each diagnostic rule on synthetic metrics.
- Health-score arithmetic and renormalisation.
- API tests:
  - a training job and its leaderboard
  - the leak case and the leak-free imbalance case
  - the combined diagnosis, including resolved findings
  - prediction, including the missing-column error
  - validation errors
  - multiclass training without CV
  - model files deleted along with their dataset
  - recovery of interrupted jobs

## Limitations

- SMOTE interpolates one-hot columns. SMOTENC would handle mixed types more
  faithfully.
- Jobs run in-process. A multi-worker deployment needs a real queue
  (the seam is already there).
- Regression isn't supported yet.
