# Phase 5: Automatic optimisation

**Goal:** don't stop at "try SMOTE". Run the corrective experiments the
diagnosis recommends, measure them fairly, and hand a better candidate model
to a human for approval.

## What was built

| Piece | File |
|---|---|
| Planner, greedy experiment runner, hyperparameter search | `backend/app/ml/optimize.py` |
| Experiment job (build the plan, run it, register the winner) | `backend/app/services/experiments.py` |
| Optional MLflow logging of every trial | `backend/app/services/tracking.py` |
| API | `backend/app/api/routes/experiments.py` |

## From diagnosis to plan

Every finding has an `interventions` list (Phase 2). The planner collects
them from the data findings (resolved for this model), the model findings and
any prediction findings:

| Intervention | Comes from | What the trial changes |
|---|---|---|
| `drop_leaky_features` | `feature.leakage` | the leaking columns are removed from **every** trial |
| `class_weight` | imbalance, weak class, no skill | `class_weight="balanced"` (balanced `sample_weight` for XGBoost) |
| `smote` | imbalance, weak class | SMOTE inside the pipeline (training folds only) |
| `drop_duplicates` | `data.duplicate_rows` | removes duplicate rows from the training split only |
| `feature_selection` | low relevance, strong correlation | drops irrelevant features plus the lower-MI column of each correlated pair |
| `hyperparameter_tuning` | overfitting, underfitting; **always** with the standard budget | `RandomizedSearchCV` (12 candidates × 3 folds) over the algorithm's search space |
| `calibration` | `model.poor_calibration` | sigmoid `CalibratedClassifierCV` (skipped for the SVM, already calibrated) |
| `threshold_tuning` | imbalance, sub-optimal threshold, FN concentration (binary only) | threshold chosen on **out-of-fold** probabilities |

## How trials run: a greedy sequence

```
Baseline (leaks removed)
  → best of {class weights, SMOTE}     each kept only if it beats the current best
  → remove duplicate training rows
  → feature selection
  → hyperparameter search              on the best configuration so far
  → calibration
  → decision threshold                 chosen on out-of-fold predictions of the best config
```

Each step starts from the **best configuration so far** and changes one
thing. The trial is kept only if it improves cross-validated macro-F1 by more
than 0.002.

### Why selection uses cross-validation, not the test set

If trials were chosen by test-set score, the "improvement" would partly be
test-set overfitting: with enough trials, one looks better by chance. So:

- **Selection** uses 3-fold stratified CV macro-F1 on the **training split**.
- **Reporting** uses the fixed held-out test set, the same rows as the
  baseline, rebuilt exactly as in Phase 4.
- **The threshold** is picked from out-of-fold training probabilities, never
  from test predictions.

You can see this in the demo run: feature selection had a *higher* test F1
(0.624) but lower CV F1, so it was correctly rejected.

## The result

The best configuration is registered as a new model (`source: "experiment"`,
name "… (optimised)", stage `candidate`). It doesn't replace anything: the
human-approval and registry workflow is in Phase 7.

The job result contains the plan, a summary per trial (config, CV F1, test
metrics, minority recall, change vs baseline, kept/rejected with the reason),
the best trial, the new model id, the improvement, the selection method and a
recommendation.

When the original model relied on a leaking feature, the optimised model
can score *lower* on the test set and still be the right choice. The
recommendation says so explicitly.

## Results on the demo dataset

Starting from the XGBoost baseline that used the leaking column:

```
KEEP Baseline (leaking features removed)  cv 0.544  test F1 0.572  churn recall 12%
KEEP Class weighting                      cv 0.599  test F1 0.618  churn recall 39%
     SMOTE oversampling                   cv 0.590  rejected (no CV gain)
     Remove duplicate training rows       cv 0.597  rejected
     Feature selection (drop age, …)      cv 0.592  rejected (higher test F1, lower CV F1)
KEEP Hyperparameter search                cv 0.602  test F1 0.619  churn recall 50%
     Decision threshold 0.45              cv 0.596  rejected
best: XGBoost + class weights + without cancellation_request + tuned
churn recall 12% → 50%, macro-F1 0.57 → 0.62, overfitting gap 0.35 → 0.25
```

The diagnosis narrative then reads:

> Automated experiment: XGBoost + class weights + without cancellation_request + tuned
> moved macro-F1 from 0.57 to 0.62 and minority recall from 12% to 50%.
> Recommendation: promote model v3 for further validation (requires approval).

## API

| Method | Path |
|---|---|
| POST | `/api/v1/models/{id}/experiments` with body `{"budget": "quick" \| "standard", "interventions": [...] \| null}` → 202 job |
| GET | `/api/v1/jobs/{id}` (poll; `progress` shows the current trial) |
| GET | `/api/v1/models/{id}/experiments` |
| GET | `/api/v1/interventions` (the catalogue) |

`interventions` restricts the plan, for example `["class_weight"]` to test
just one idea. The `quick` budget skips hyperparameter search.

## MLflow (optional)

With `MD_MLFLOW_TRACKING_URI` set and `mlflow` installed, every trained model
is logged as an MLflow run (params, test metrics, `diagnosis.json`). Each
experiment is logged as a parent run with one nested run per trial. If
MLflow isn't configured or fails, training continues; tracking problems are
logged and never break a job.

## Tests (`tests/test_optimisation.py`)

- Planner: intervention order, leaky columns, feature drops (the weaker
  correlated column chosen by mutual information), no threshold for
  multiclass, no calibration for SVM, restrictions, already-dropped leaks.
- End to end:
  - leak removal applied to every trial
  - imbalance trials run and the best model is registered as a candidate
  - minority recall improves
  - the diagnosis narrative picks up the experiment
  - experiments are listed per model
- Standard budget: hyperparameter search runs and stores the tuned params.
- Restricted plans, and validation of unknown interventions.
- MLflow: a fake `mlflow` module shows that runs, params, metrics and
  `diagnosis.json` are logged when tracking is enabled.

## Limitations

- The greedy search is fast and easy to explain, but it doesn't explore every
  combination (for example SMOTE together with tuning when class weighting
  won the earlier step).
- RandomizedSearchCV over a hand-written space. Optuna's Bayesian search
  could replace it behind the same function.
