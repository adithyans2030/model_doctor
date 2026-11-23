export type Severity = "critical" | "warning" | "info";

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

export interface TargetSummary {
  n: number; mean: number; std: number; min: number; q1: number; median: number; q3: number; max: number;
  skew: number; n_unique: number; all_non_negative: boolean;
  histogram: { edges: number[]; counts: number[] };
}

export type Task = "classification" | "regression";

export interface Scores {
  primary_metric: "f1_macro" | "r2";
  primary: number;
  // classification
  f1_macro?: number; accuracy?: number; balanced_accuracy?: number; roc_auc?: number | null;
  // regression
  r2?: number; rmse?: number; mae?: number; mape?: number | null;
  worst_class: { label: string; recall: number; f1: number } | null;
}

export interface User {
  id: string; email: string; name: string; role: "viewer" | "member" | "admin";
  is_active: boolean; created_at: string; last_login_at: string | null;
}

export interface Finding {
  check: string;
  category: string;
  severity: Severity;
  title: string;
  evidence: Record<string, unknown>;
  cause: string | null;
  impact: string | null;
  recommendation: string | null;
  interventions: string[];
  columns: string[];
  resolved?: boolean;
}

export interface DataReport {
  id: string;
  summary: Record<string, any>;
  score: { value: number; penalties: { check: string; severity: Severity; points: number; title: string }[] };
  findings: Finding[];
  column_stats: any[];
  feature_target: { column: string; score: number | null; metric: string; mutual_info: number }[];
  correlations: { a: string; b: string; method: string; value: number }[];
}

export interface Job {
  id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed";
  params: Record<string, any>;
  result: Record<string, any> | null;
  progress: string | null;
  error: string | null;
  created_at: string;
}

export interface ModelSummary {
  id: string;
  project_id: string;
  dataset_id: string;
  name: string;
  algorithm: string;
  version: number;
  source: string;
  stage: string;
  created_at: string;
  n_findings: number;
  task: Task;
  scores: Scores;
}

export interface ModelDetail extends ModelSummary {
  config: Record<string, any>;
  feature_spec: Record<string, any>;
  metrics: Record<string, any>;
  findings: Finding[];
  importances: { feature: string; importance: number }[] | null;
}

export interface Diagnosis {
  dataset_id: string;
  model: { id: string; name: string; version: number; stage: string } | null;
  health: { overall: number | null; components: Record<string, { value: number | null; explanation: string }> };
  severity_counts: Record<Severity, number>;
  findings: Finding[];
  recommendations: { intervention: string; label: string; severity: Severity; because: { check: string; title: string }[] }[];
  narrative: string[];
  latest_experiment: Record<string, any> | null;
}
