# Phase 4: Explainability and error analysis

**Goal:** answer *why*. Which features drive the model, why a particular row
got its prediction, where the errors concentrate, and whether some groups get
worse predictions than others.

## What was built

| Piece | File |
|---|---|
| SHAP, permutation importance, local explanations | `backend/app/ml/explain.py` |
| Error slices, false-negative concentration, confidence analysis, group performance | `backend/app/ml/error_analysis.py` |
| Use cases (exact holdout rebuild, caching) | `backend/app/services/explanations.py` |
| API | `backend/app/api/routes/explain.py` |

Explanations are cached in `models.explanations` (JSON), so no new table was
needed.

## Exact holdout reconstruction

Every explanation is computed on **the same test rows the model was evaluated
on**. `holdout()` re-runs `prepare_training_data` with the model's stored
`seed`, `test_size` and target, which reproduces the split exactly (the split
is deterministic). Nothing is computed on training data, except the
"typical value" reference used for occlusion and linear SHAP.

## Global importance: two complementary views

| Method | What it measures | Models |
|---|---|---|
| **SHAP** (mean \|SHAP\| per column) | How much each feature moves the model's output | XGBoost: exact TreeSHAP built into XGBoost (`pred_contribs=True`). Logistic Regression: exact linear SHAP, φⱼ = wⱼ(xⱼ − baselineⱼ). Random Forest: `shap.TreeExplainer` when a numpy-compatible `shap` is installed |
| **Permutation importance** | How much macro-F1 drops when one column is shuffled (3 repeats, up to 2,000 test rows) | All models, using the model's own decision threshold |

SHAP is computed on the one-hot/transformed features and **summed back to the
original columns** using the column transformer's output names, so the result
says `contract`, not `cat__contract_Month-to-month`. For binary models the
values explain the positive (minority) class.

**Why linear SHAP is computed directly:** the `shap` 0.46 package installed in
this environment fails to *import* with numpy 2.3. For a linear model with
independent features, SHAP has an exact closed form, so no library is needed.
XGBoost includes its own TreeSHAP. Only Random Forest needs the `shap`
package; when it can't be imported, the endpoint still returns permutation
importance and per-prediction explanations fall back to occlusion.
`requirements.txt` asks for `shap>=0.48` so the Docker image has a working
version.

## Local explanations (one prediction)

`POST /models/{id}/explain` takes a dataset `row_id` or a new `record`:

1. Prepare the row exactly as at training time (same feature spec).
2. Get the probabilities and the predicted class (respecting the threshold).
3. Compute per-feature contributions **towards the predicted class**.
   - **SHAP** where supported. Units are log-odds for XGBoost and LR,
     probability for RF. For single-output binary models the sign is flipped
     when the predicted class is 0.
   - **Occlusion** otherwise (SVM, or RF without `shap`): replace one feature
     with its typical training value (median, or mode for categories) and
     measure how much the predicted-class probability changes.
4. A deterministic sentence, built only from the computed numbers:

```
Predicted '0' with 97% confidence. Mainly driven by contract = One year,
total_charges = 4099, age = 32. Pulling the other way: region = West,
num_support_calls = 2.
```

The response also includes the actual label (for dataset rows) and whether
the row is in the test sample.

## Error analysis

Instead of reading thousands of predictions:

- **Slices.** Every categorical value (if a column has ≤ 30 distinct values)
  and every numeric quartile, labelled by the actual min–max of its values.
  Dates are labelled as dates. A slice needs at least max(20 rows, 3% of the
  test set). Slices with ≥ 1.3× the average error rate are reported, ranked by
  *excess errors* = (slice error − overall error) × slice size, so large,
  bad segments come first.
- **False-negative concentration** (binary). Segments that hold ≥ 50% of all
  missed positives while being at least 15 percentage points smaller than
  that share of the test set.
- **Confidence.** The share of predictions below 60% confidence and their
  error rate, plus wrong predictions made with ≥ 90% confidence (possible
  label noise or a shortcut feature) and the 20 most confident errors.

Results on the demo data, XGBoost without the leak:

```
79% of false negatives occur where contract = Month-to-month (54% of test rows).
The model is wrong 15% of the time when contract = Month-to-month vs 10% overall (1.5× the average).
The model is wrong 19% of the time when tenure_months = 1 – 18 vs 10% overall (2.0× the average).
```

## Group performance (fairness)

For each categorical feature with 2–10 groups (≥ 20 rows each),
ModelDoctor computes per-group accuracy, macro-F1, balanced accuracy and
positive-class recall. The largest **balanced-accuracy gap** across groups
feeds the health score's Fairness component: 100 − 150 × gap.

Any categorical column can be analysed this way. ModelDoctor doesn't assume
which columns are sensitive attributes; the user decides which gaps matter.

## New findings

| Check | Rule |
|---|---|
| `prediction.error_cluster` | top slices with ≥ 1.5× the average error rate (warning) |
| `prediction.fn_concentration` | as defined above (warning, suggests threshold tuning) |
| `prediction.confident_errors` | ≥ 2% of test rows are wrong with ≥ 90% confidence (warning) |
| `prediction.uncertain` | ≥ 20% of predictions have confidence < 60% (info) |
| `fairness.group_gap` | balanced-accuracy gap ≥ 0.15 warning, ≥ 0.30 critical |

These findings join the diagnosis (`GET /datasets/{id}/diagnosis`) once
explanations have been computed for the model.

## API

| Method | Path |
|---|---|
| GET | `/api/v1/models/{id}/explanations[?refresh=true]`: global importance, error analysis, fairness, findings, 200 test-row samples |
| POST | `/api/v1/models/{id}/explain` with body `{"row_id": 123}` or `{"record": {...}}` |

## Tests (`tests/test_explainability.py`)

- **Additivity:** for LR and XGBoost, differences in summed SHAP values
  between rows equal differences in the model's log-odds (to 1e-3).
- Sign flipping for single-output binary models.
- Error analysis on synthetic data with a known failure segment: it finds the
  cluster, the 100% false-negative concentration, and a critical group gap.
- API:
  - the explanations payload for XGBoost (TreeSHAP) and LR (linear SHAP)
  - caching and refresh
  - row and record explanations (actual label, sorted contributions, units)
  - validation and 404s
  - the fairness component appearing in the health score

## Limitations

- LIME isn't included. SHAP plus occlusion covers the same need without
  another dependency.
- SHAP for calibrated models explains the underlying estimator, before
  calibration.
- Linear SHAP assumes independent features (interventional SHAP), the
  standard choice.
