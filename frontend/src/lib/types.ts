/* Types for the ModelDoctor REST API (/api/v1). They describe the existing backend
   responses; nothing here changes the API contract. Shapes were taken from live
   responses on the demo datasets. */

export type Severity = "critical" | "warning" | "info";
export type Task = "classification" | "regression";
export type Stage = "candidate" | "approved" | "production" | "rejected" | "archived";
export type Role = "viewer" | "member" | "admin";

export interface MeanStd { mean: number; std: number }
export interface HistogramData { edges: number[]; counts: number[] }

/* ---------- projects & datasets ---------- */

export interface Project {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  dataset_count: number;
}

export interface ColumnProfile {
  name: string;
  pandas_dtype: string;
  inferred_type: string;
  n_missing: number;
  missing_pct: number;
  n_unique: number;
  numeric_stored_as_text: boolean;
  sample_values: string[];
}

export interface ClassCount { label: string; count: number; share: number }

export interface TargetSummary {
  n: number; mean: number; std: number; min: number; q1: number; median: number; q3: number; max: number;
  skew: number; n_unique: number; all_non_negative: boolean;
  histogram: HistogramData;
}

export interface Dataset {
  id: string;
  project_id: string;
  original_filename: string;
  n_rows: number;
  n_cols: number;
  target_column: string | null;
  task_type: string | null;
  created_at: string;
  size_bytes?: number;
  columns?: ColumnProfile[];
  n_classes?: number | null;
  class_distribution?: ClassCount[] | null;
  target_summary?: TargetSummary | null;
}

export interface DatasetPreview { columns: string[]; rows: (string | number | boolean | null)[][] }
export interface SetTargetResponse { warnings: string[] }

/* ---------- findings & reports ---------- */

export type EvidenceValue = string | number | boolean | null | EvidenceValue[] | { [k: string]: EvidenceValue };

export interface Finding {
  check: string;
  category: string;
  severity: Severity;
  title: string;
  evidence: Record<string, EvidenceValue>;
  cause: string | null;
  impact: string | null;
  recommendation: string | null;
  interventions: string[];
  columns: string[];
  resolved?: boolean;
}

export type SeverityCounts = Record<Severity, number>;

export interface ColumnStat {
  name: string;
  inferred_type: string;
  missing_pct: number;
  n_unique: number;
  median?: number | null;
  n_outliers?: number;
  histogram?: HistogramData | null;
}

export interface DataReport {
  id: string;
  dataset_id: string;
  created_at: string;
  summary: {
    n_rows: number;
    n_cols: number;
    n_features_usable: number | null;
    excluded_columns: Record<string, string>;
    missing_cells_pct: number;
    duplicate_rows: number;
    duplicate_pct: number;
    type_counts: Record<string, number>;
    target: string | null;
    task_type: string | null;
    class_distribution: ClassCount[] | null;
    target_summary: TargetSummary | null;
    imbalance_ratio: number | null;
    severity_counts: SeverityCounts;
  };
  score: { value: number; penalties: { check: string; severity: Severity; points: number; title: string }[] };
  findings: Finding[];
  column_stats: ColumnStat[];
  feature_target: { column: string; score: number | null; metric: string; mutual_info: number }[];
  correlations: { a: string; b: string; method: string; value: number }[];
}

/* ---------- jobs ---------- */

export type JobStatus = "queued" | "running" | "completed" | "failed";

interface JobBase {
  id: string;
  dataset_id?: string;
  status: JobStatus;
  progress: string | null;
  error: string | null;
  created_at: string;
}

export interface TrainResult {
  model_ids: string[];
  best_model_id: string | null;
  skipped: { algorithm: string; reason: string }[];
}

export interface TrainJob extends JobBase {
  kind: "train";
  params: { algorithms: string[]; drop_columns: string[]; cv_folds: number };
  result: TrainResult | null;
}

export interface TrialMetrics {
  primary: number;
  // classification
  f1_macro?: number; accuracy?: number; balanced_accuracy?: number; roc_auc?: number | null; pr_auc?: number | null;
  ece?: number | null; minority_recall?: number;
  // regression
  r2?: number; rmse?: number; mae?: number;
}

export interface ModelConfig {
  algorithm: string;
  params: Record<string, number | string | boolean | null>;
  class_weight: string | null;
  resampling: string | null;
  drop_columns: string[];
  dedupe: boolean;
  threshold: number | null;
  calibration: string | null;
  target_transform: string | null;
}

export interface Trial {
  step: number;
  name: string;
  intervention: string | null;
  description: string;
  config: ModelConfig;
  kept: boolean;
  reason: string;
  cv_primary: number;
  metrics: TrialMetrics;
  delta_vs_baseline: Partial<TrialMetrics>;
  model_id?: string;
  version?: number;
}

export interface ExperimentResult {
  base_model_id: string;
  task: Task;
  primary_metric: "f1_macro" | "r2";
  base_model_primary: number;
  baseline_primary: number;
  plan: {
    interventions: string[];
    leaky_columns: string[];
    feature_drops: string[];
    notes: string[];
    labels: Record<string, string>;
  };
  trials: Trial[];
  best: Trial;
  best_model_id: string | null;
  improvement: Partial<TrialMetrics>;
  selection: string;
  seconds: number;
  recommendation: string;
}

export interface ExperimentJob extends JobBase {
  kind: "experiment";
  params: { base_model_id: string; budget: string; interventions: string[] | null };
  result: ExperimentResult | null;
}

export type Job = TrainJob | ExperimentJob;

/* ---------- models ---------- */

export interface ClassMetric { label: string; precision: number; recall: number; f1: number; support: number }

export interface Scores {
  primary_metric: "f1_macro" | "r2";
  primary: number;
  f1_macro?: number; accuracy?: number; balanced_accuracy?: number; roc_auc?: number | null;
  r2?: number; rmse?: number; mae?: number; mape?: number | null;
  worst_class: ClassMetric | null;
}

export interface ModelSummary {
  id: string;
  project_id: string;
  dataset_id: string;
  name: string;
  algorithm: string;
  version: number;
  source: string;
  stage: Stage;
  created_at: string;
  n_findings: number;
  task: Task;
  scores: Scores;
}

export interface Split { seed: number; test_size: number; n_train: number; n_test: number; n_labelled: number }

export interface ThresholdPoint { threshold: number; precision: number; recall: number; f1: number }

export interface ClassificationTestMetrics {
  n: number;
  accuracy: number;
  balanced_accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  precision_macro: number;
  recall_macro: number;
  roc_auc: number | null;
  pr_auc: number | null;
  ece: number | null;
  per_class: ClassMetric[];
  worst_class: ClassMetric;
  positive_class: ClassMetric | null;
  confusion_matrix: { labels: string[]; matrix: number[][] };
  threshold: number | null;
  threshold_analysis: { curve: ThresholdPoint[]; best_threshold: number; best_f1: number; f1_at_0_5: number } | null;
}

export interface RegressionTestMetrics {
  n: number;
  r2: number;
  rmse: number;
  mae: number;
  median_ae: number;
  mape: number | null;
  residual_mean: number;
  residual_std: number;
  error_vs_prediction_corr: number | null;
  top5pct_error_share: number;
  residual_histogram: HistogramData;
  points: { actual: number; predicted: number }[];
}

interface CvBlock<K extends string> { folds: number; val: Record<K, MeanStd>; train: Record<K, MeanStd> }

export interface ClassificationMetrics {
  task: "classification";
  primary_metric: "f1_macro";
  test: ClassificationTestMetrics;
  cv: CvBlock<"accuracy" | "balanced_accuracy" | "f1_macro"> | null;
  dummy: { accuracy: number; balanced_accuracy: number; f1_macro: number } | null;
  split: Split;
  positive_label: string | null;
  classes: string[];
  n_features: number;
}

export interface RegressionMetrics {
  task: "regression";
  primary_metric: "r2";
  test: RegressionTestMetrics;
  cv: CvBlock<"r2" | "rmse" | "mae"> | null;
  dummy: { r2?: number; rmse: number; mae?: number } | null;
  split: Split;
  n_features: number;
}

interface ModelDetailBase extends ModelSummary {
  config: ModelConfig;
  feature_spec: { target: string; numeric: string[]; categorical: string[]; datetime: string[]; excluded: Record<string, string> };
  findings: Finding[];
  importances: { feature: string; importance: number }[] | null;
}
export interface ClassificationModel extends ModelDetailBase { task: "classification"; metrics: ClassificationMetrics }
export interface RegressionModel extends ModelDetailBase { task: "regression"; metrics: RegressionMetrics }
export type ModelDetail = ClassificationModel | RegressionModel;

export interface AlgorithmInfo { key: string; label: string; available: boolean; default: boolean; note: string }

/* ---------- explainability ---------- */

export interface ErrorSlice {
  feature: string; value: string; n: number; share: number; lift: number | null;
  error_rate?: number; mae?: number;
}

export interface FairnessGroup {
  value: string; n: number;
  balanced_accuracy?: number | null; positive_recall?: number | null;
  mae?: number; r2?: number | null;
}

export interface Explanations {
  model_id: string;
  computed_at: string;
  global_importance: {
    shap: { method: string; explained_class?: string; values: { feature: string; mean_abs_shap: number }[] } | null;
    permutation: { metric: string; baseline: number; values: { feature: string; importance: number; std: number }[] };
  };
  error_analysis: {
    summary: {
      n_test: number;
      n_errors?: number; error_rate?: number; false_negatives?: number; false_positives?: number;
      mae?: number; large_error_threshold?: number | null;
    };
    slices: ErrorSlice[];
    statements: string[];
  };
  fairness: {
    metric: string;
    max_gap: number | null;
    worst_feature: string | null;
    features: { feature: string; gap: number; groups: FairnessGroup[] }[];
  };
  findings: Finding[];
  samples: {
    row_id: number;
    true: string | number;
    predicted: string | number;
    confidence?: number;
    correct?: boolean;
    abs_error?: number;
  }[];
}

export interface LocalExplanation {
  model_id: string;
  row_id: number;
  actual: string | number | null;
  prediction: string | number;
  confidence?: number | null;
  fidelity: number | null;
  method: string;
  units: string;
  contributions: { feature: string; value: string; contribution: number }[];
  explanation: string;
}

/* ---------- registry ---------- */

export interface StageEvent {
  id: string;
  model_id: string;
  from_stage: Stage | null;
  to_stage: Stage;
  actor: string;
  note: string | null;
  created_at: string;
}

export interface Registry {
  project_id: string;
  production: ModelSummary | null;
  models: ModelSummary[];
  events: StageEvent[];
}

type ConfigValue = ModelConfig[keyof ModelConfig];

export interface ModelComparison {
  a: { id: string; name: string; version: number; stage: Stage };
  b: { id: string; name: string; version: number; stage: Stage };
  same_test_split: boolean;
  metrics: { metric: string; a: number | null; b: number | null; delta: number | null; better: "a" | "b" | null }[];
  config_diff: Record<string, { a: ConfigValue; b: ConfigValue }>;
  findings: { resolved_in_b: string[]; new_in_b: string[]; shared: string[] };
  winner: "a" | "b" | null;
  summary: string[];
}

/* ---------- monitoring ---------- */

export interface FeatureDrift {
  feature: string;
  type: string;
  psi: number;
  test: string;
  statistic: number | null;
  p_value: number | null;
  severity: string;
  ref_mean?: number | null;
  batch_mean?: number | null;
  missing_ref?: number;
  missing_batch?: number;
}

export interface DriftReport {
  score: number;
  drifted_features: number;
  n_features: number;
  features: FeatureDrift[];
  prediction_drift: { psi: number; kind?: string; reference?: Record<string, number>; batch?: Record<string, number> };
  performance: {
    metric: string;
    n_labelled: number;
    test_primary: number;
    batch_primary: number;
    delta_primary: number;
    batch_minority_recall?: number;
  } | null;
  findings: Finding[];
  severity_counts: SeverityCounts;
  thresholds: { psi_moderate: number; psi_major: number };
  batch: { filename: string; n_rows: number };
}

export interface DriftCheck { id: string; model_id: string; created_at: string; report: DriftReport }

/* ---------- diagnosis ---------- */

export interface Recommendation {
  intervention: string;
  label: string;
  severity: Severity;
  because: { check: string; title: string }[];
}

export interface Diagnosis {
  dataset_id: string;
  model: { id: string; name: string; version: number; stage: Stage } | null;
  health: { overall: number | null; components: Record<string, { value: number | null; explanation: string }> };
  severity_counts: SeverityCounts;
  findings: Finding[];
  recommendations: Recommendation[];
  narrative: string[];
  latest_experiment: Partial<ExperimentResult> | null;
}

/* ---------- users & assistant ---------- */

export interface User {
  id: string; email: string; name: string; role: Role;
  is_active: boolean; created_at: string; last_login_at: string | null;
}

export interface Providers {
  provider: "ollama" | "anthropic" | "rules";
  ollama_available: boolean;
  ollama_url: string;
  default_model: string;
  local_models: { name: string; parameters: string | null; size_gb: number }[];
  loaded_models: string[];
}

export type AssistantEvent =
  | { type: "meta"; mode: "ollama" | "anthropic" | "rules"; model: string | null }
  | { type: "delta"; text: string }
  | { type: "done"; sources?: string[]; note?: string };
