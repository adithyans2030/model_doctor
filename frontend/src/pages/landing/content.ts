/* Every figure on the landing page, with where it comes from. Demo numbers are from
   running the dashboard flow on sample_data/churn_demo.csv (a synthetic dataset). */

export const GITHUB = "https://github.com/adithyans2030/model_doctor";
export const DOCS = `${GITHUB}/tree/main/docs`;
export const LICENSE_URL = `${GITHUB}/blob/main/LICENSE`;
export const AUTHOR = "Adithyan Sabu";

export const PROOF = [
  { value: "128", label: "backend tests", source: "pytest --collect-only in backend/" },
  { value: "37", label: "diagnostic check types", source: "distinct finding ids across data, feature, model and prediction checks" },
  { value: "9 / 9", label: "planted problems found", source: "Data Doctor on churn_demo.csv" },
  { value: "5% → 49%", label: "churn recall after fixes", source: "synthetic demo dataset, held-out test set" },
] as const;

/** Diagnosis of the original baseline (Logistic Regression v1) on the demo dataset. */
export const DEMO_DIAGNOSIS = {
  health: 75,
  components: [
    { name: "Data", value: 34 },
    { name: "Performance", value: 95 },
    { name: "Robustness", value: 95 },
  ],
  counts: { critical: 2, warning: 4, info: 7 },
  findings: [
    { severity: "critical", check: "feature.leakage", title: "'cancellation_request' alone predicts the target", evidence: "single-feature ROC AUC 0.951" },
    { severity: "critical", check: "model.dominant_feature", title: "One feature drives 53% of the model", evidence: "importance 0.532" },
    { severity: "warning", check: "data.class_imbalance", title: "Class imbalance detected", evidence: "ratio 9.26 : 1" },
    { severity: "warning", check: "data.missing_values", title: "6.0% missing values in 'age'", evidence: "301 rows" },
    { severity: "warning", check: "data.outliers", title: "25 extreme outliers in 'monthly_charges'", evidence: "max 1199.6, median 70.1" },
  ],
} as const;

/** Leak-free baseline vs the best experiment (candidate v6). */
export const DEMO_RESULT = [
  { label: "Churn recall", before: "5%", after: "49%" },
  { label: "Macro-F1", before: "0.52", after: "0.65" },
  { label: "Balanced accuracy", before: "0.52", after: "0.69" },
] as const;

export const DEMO_SHAP = [
  { feature: "cancellation_request", value: 1.221 },
  { feature: "contract", value: 0.626 },
  { feature: "num_support_calls", value: 0.36 },
  { feature: "tenure_months", value: 0.306 },
  { feature: "signup_date", value: 0.306 },
] as const;

export const DEMO_TRIALS = [
  { name: "Leak-free baseline", kept: true },
  { name: "Class weighting", kept: true },
  { name: "SMOTE oversampling", kept: false },
  { name: "Remove duplicate rows", kept: false },
  { name: "Feature selection", kept: true },
  { name: "Hyperparameter search", kept: true },
  { name: "Decision threshold 0.65", kept: true },
] as const;
