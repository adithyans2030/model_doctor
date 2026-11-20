# Phase 2: Data Doctor

**Goal:** before training anything, find the data and feature problems that
make models fail, explain each one with evidence, and summarise data health in
a transparent score.

## What was built

| Piece | File |
|---|---|
| Finding format, scoring, intervention catalogue | `backend/app/diagnostics/findings.py` |
| Data Doctor checks | `backend/app/diagnostics/data_doctor.py` |
| Feature spec and cleaning (shared with Phase 3) | `backend/app/ml/features.py` |
| Report caching | `backend/app/services/reports.py`, table `reports` (migration `0002_reports`) |
| API | `GET /api/v1/datasets/{id}/data-report[?refresh=true]` |

## The finding: one format for every check

Every check in every phase produces the same structure:

```json
{
  "check": "feature.leakage",
  "category": "feature",
  "severity": "critical",
  "title": "Possible target leakage: 'cancellation_request' alone predicts the target",
  "evidence": {"column": "cancellation_request", "roc_auc": 0.951, "threshold": 0.95},
  "cause": "...", "impact": "...", "recommendation": "...",
  "interventions": ["drop_leaky_features"],
  "columns": ["cancellation_request"]
}
```

- `evidence` always holds the actual numbers, so a claim can be checked.
- `interventions` names actions from a fixed catalogue
  (`findings.INTERVENTIONS`). The Phase 5 optimiser reads this field to decide
  which experiments to run, which links each diagnosis to its fix.
- The Phase 9 assistant answers from these objects, so it explains real
  analysis rather than inventing a diagnosis.

## Checks and thresholds

| Check id | Severity rule | Intervention |
|---|---|---|
| `data.missing_values` | info <5%, warning 5–40%, critical ≥40% per column | |
| `data.empty_column` | warning | |
| `data.constant_column` | warning, 1 distinct value | |
| `data.near_constant` | info, top value ≥99% | |
| `data.id_column` | info, excluded from modelling | |
| `data.text_column` / `data.datetime_column` | info, explains how the column is treated | |
| `data.high_cardinality` | warning, more than 50 categories | |
| `data.invalid_values` | warning, non-numeric values in a numeric column | |
| `data.outliers` | values beyond 3×IQR; warning ≥0.25% of rows, else info | |
| `data.duplicate_rows` | warning ≥1%, else info | `drop_duplicates` |
| `data.conflicting_labels` | warning, identical feature rows with different labels (label noise) | |
| `data.class_imbalance` | majority/minority ratio: warning ≥3, critical ≥20 | `class_weight`, `smote`, `threshold_tuning` |
| `data.small_dataset` | info, fewer than 500 rows | |
| `feature.leakage` | single-feature probe score: warning ≥0.90, critical ≥0.95 | `drop_leaky_features` |
| `feature.low_relevance` | info, mutual information <0.002 and probe score <0.55 | `feature_selection` |
| `feature.multicollinearity` | Pearson \|r\| ≥0.8 info, ≥0.9 warning; Cramér's V ≥0.9 warning | `feature_selection` (if strong) |

### How leakage is detected

For each feature, a depth-4 decision tree is trained on **that feature alone**
(plus a missing-value indicator) with 3-fold stratified cross-validation, on a
stratified sample of up to 20,000 rows. Binary targets are scored with
out-of-fold ROC AUC (symmetric, so either direction counts); multiclass targets
with balanced accuracy. A real feature rarely predicts the target almost
perfectly on its own. When one does, it is usually recorded after the outcome
(a cancellation flag for churn) or derived from the label.

The same probe scores and mutual information give the feature-relevance ranking
returned in `feature_target`.

## The data-quality score

`score = 100 − Σ penalties`, with critical = 20, warning = 8 and info = 2
points. Repeats of one check (for example many columns with a few missing
values) are capped at 25 points in total, so one noisy check can't dominate. The
report lists every penalty, so the number can always be explained. It is a
summary indicator, not a substitute for the individual findings.

## Report contents

- `summary`: rows, columns, usable features, excluded columns with the reason,
  % missing cells, duplicates, type counts, class distribution, imbalance ratio
  and severity counts.
- `findings`: sorted critical → warning → info.
- `column_stats`: numeric columns get mean/std/quartiles/skew/outliers and a
  20-bin histogram clipped to the 1st–99th percentile; categorical columns get
  their top 10 values.
- `feature_target`: per-feature probe score and mutual information.
- `correlations`: flagged correlated pairs.

**Caching:** the report is stored in `reports` keyed by the target column. It
is reused until the target changes or `?refresh=true` is passed.

## Results on the demo dataset

All 9 planted problems are detected:

```
critical feature.leakage            'cancellation_request' alone predicts the target (AUC 0.95)
warning  data.class_imbalance       ratio 9.3 (9.7% churn)
warning  data.constant_column       'country'
warning  data.missing_values        6.0% in 'age'
warning  data.outliers              25 extreme values in 'monthly_charges'
info     data.duplicate_rows        50 rows (1.0%)
info     data.id_column             'customer_id'
info     feature.low_relevance      age, has_partner, payment_method
info     feature.multicollinearity  tenure_months ~ total_charges (r = 0.81)
```

The region effect (higher churn in "West") isn't a data problem. Phase 4's
error analysis surfaces it.

## Tests (`tests/test_data_doctor.py`)

- Ground-truth test: every planted demo problem is detected with the right
  column and intervention.
- Report structure, ordering, score arithmetic, histograms and the relevance
  ranking.
- Caching: reuse, `refresh`, and recomputation when the target changes.
- A report for a dataset without a target.
- Column-level checks on targeted synthetic data (critical missing values,
  high cardinality, invalid values, near-constant, text, datetime,
  small dataset).
- Conflicting labels and critical multiclass imbalance.
- The score cap, and rejection of unknown interventions.

## Limitations and next steps

- Outliers use one univariate rule (3×IQR). Multivariate methods such as
  isolation forest could be added.
- The leakage probe only catches single-feature leaks. Leaks spread across
  several features show up in Phase 3 as suspiciously high scores and a
  dominant feature.
- The train-vs-production distribution checks are in Phase 8 (drift).
