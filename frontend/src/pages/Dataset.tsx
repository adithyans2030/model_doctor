import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Assistant } from "@/components/Assistant";
import { Bars, Histogram } from "@/components/charts";
import {
  Crumbs, Empty, ErrorBox, FindingsList, HealthPanel, Loading, Section, SeverityBadge, Spinner, StageBadge, Stat, Tabs, cx,
} from "@/components/ui";
import { api, fmt } from "@/lib/api";
import { useJob, useResource } from "@/lib/hooks";
import type { DataReport, Dataset, Diagnosis, Job, ModelSummary } from "@/lib/types";

type Tab = "overview" | "data" | "models" | "diagnosis" | "assistant";

export default function DatasetPage() {
  const { id = "" } = useParams();
  const ds = useResource<Dataset>(`/datasets/${id}`);
  const [tab, setTab] = useState<Tab>("overview");

  if (ds.error) return <ErrorBox error={ds.error} />;
  if (!ds.data) return <Loading />;
  const d = ds.data;
  const needsTarget = !d.target_column;

  return (
    <div>
      <Crumbs><Link to="/">Projects</Link> / <Link to={`/projects/${d.project_id}`}>Project</Link> / <span>{d.original_filename}</span></Crumbs>
      <h1>{d.original_filename}</h1>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[13px] text-fg-2">
        <span>{fmt.int(d.n_rows)} rows</span><span>·</span><span>{d.n_cols} columns</span><span>·</span>
        <span>target: {d.target_column ? <b>{d.target_column}</b> : <span className="text-fg-3">not set</span>}</span>
        {d.task_type && <span className="badge">{d.task_type.replace("_", " ")}</span>}
      </div>
      <Tabs<Tab> value={tab} onChange={setTab} tabs={[
        ["overview", "Overview"], ["data", "Data Doctor"], ["models", "Models"], ["diagnosis", "Diagnosis"], ["assistant", "Assistant"],
      ]} />
      {tab === "overview" && <Overview d={d} onChanged={ds.reload} />}
      {tab !== "overview" && tab !== "data" && needsTarget && <div className="notice">Select a target column on the Overview tab first.</div>}
      {tab === "data" && <DataDoctor id={id} key={d.target_column || "none"} />}
      {tab === "models" && !needsTarget && <Models d={d} />}
      {tab === "diagnosis" && !needsTarget && <DiagnosisView id={id} />}
      {tab === "assistant" && !needsTarget && <Assistant datasetId={id} />}
    </div>
  );
}

function Overview({ d, onChanged }: { d: Dataset; onChanged: () => void }) {
  const navigate = useNavigate();
  const [target, setTarget] = useState(d.target_column || "");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const preview = useResource<{ columns: string[]; rows: unknown[][] }>(`/datasets/${d.id}/preview?rows=8`);

  async function saveTarget() {
    setError(null);
    try {
      const r = await api<{ warnings: string[] }>(`/datasets/${d.id}/target`, { json: { column: target } });
      setWarnings(r.warnings);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove() {
    if (!confirm("Delete this dataset and its models?")) return;
    await api(`/datasets/${d.id}`, { method: "DELETE" });
    navigate(`/projects/${d.project_id}`);
  }

  const columns = d.columns || [];
  const candidates = columns.filter((c) => ["categorical", "boolean", "numeric"].includes(c.inferred_type));
  const typeCounts = Object.entries(columns.reduce<Record<string, number>>((a, c) => ({ ...a, [c.inferred_type]: (a[c.inferred_type] || 0) + 1 }), {}));
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Section title="Prediction target">
          <div className="flex flex-wrap gap-2">
            <select className="field" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">Choose a column…</option>
              {candidates.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.inferred_type}, {c.n_unique} values)</option>)}
            </select>
            <button className="btn btn-primary" disabled={!target || target === d.target_column} onClick={saveTarget}>Set target</button>
          </div>
          <ErrorBox error={error} />
          {warnings.map((w) => <div key={w} className="notice mt-2 text-[13px]"><SeverityBadge severity="warning" /> {w}</div>)}
          {d.class_distribution && (
            <div className="mt-3">
              <div className="mb-1 text-[13px] text-fg-2">Class distribution</div>
              <Bars items={d.class_distribution.map((c) => ({ label: c.label, value: c.share, note: `${fmt.int(c.count)} rows` }))}
                format={(v) => fmt.pct(v, 1)} max={1} />
            </div>
          )}
        </Section>
        <Section title="File" actions={<button className="btn btn-danger" onClick={remove}>Delete dataset</button>}>
          <table className="tbl"><tbody>
            <tr><td className="text-fg-3">Size</td><td>{fmt.int(Math.round((d.size_bytes || 0) / 1024))} KB</td></tr>
            <tr><td className="text-fg-3">Uploaded</td><td>{fmt.date(d.created_at)}</td></tr>
            <tr><td className="text-fg-3">Columns by type</td><td>{typeCounts.map(([k, v]) => `${v} ${k}`).join(", ")}</td></tr>
          </tbody></table>
        </Section>
      </div>
      <Section title="Schema">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead><tr><th>Column</th><th>Type</th><th className="num">Missing</th><th className="num">Unique</th><th>Samples</th></tr></thead>
            <tbody>
              {columns.map((c) => (
                <tr key={c.name}>
                  <td><b>{c.name}</b>{c.name === d.target_column && <span className="badge ml-1.5">target</span>}</td>
                  <td>{c.inferred_type}{c.numeric_stored_as_text && <span className="text-xs text-fg-3"> (stored as text)</span>}</td>
                  <td className="num">{c.missing_pct ? `${c.missing_pct}%` : "–"}</td>
                  <td className="num">{fmt.int(c.n_unique)}</td>
                  <td className="font-mono text-xs text-fg-2">{c.sample_values.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Preview">
        {!preview.data ? <Loading /> : (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr>{preview.data.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{preview.data.rows.map((r, i) => (
                <tr key={i}>{r.map((v, j) => <td key={j} className="text-xs">{v === null ? <span className="text-fg-3">–</span> : String(v)}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

function DataDoctor({ id }: { id: string }) {
  const [refresh, setRefresh] = useState(0);
  const report = useResource<DataReport>(`/datasets/${id}/data-report${refresh ? `?refresh=true&n=${refresh}` : ""}`);
  if (report.error) return <ErrorBox error={report.error} />;
  if (!report.data) return <Loading label="Running the Data Doctor" />;
  const r = report.data;
  const s = r.summary;
  const excluded = Object.entries(s.excluded_columns || {}).map(([k, v]) => `${k} (${v})`).join(", ");
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="DATA QUALITY" value={<span className="text-5xl">{r.score.value}</span>} sub={`100 − penalties (${r.score.penalties.length})`} />
        <Stat label="MISSING CELLS" value={`${s.missing_cells_pct}%`} sub={`${s.duplicate_rows} duplicate rows (${s.duplicate_pct}%)`} />
        <Stat label="USABLE FEATURES" value={s.n_features_usable ?? "–"} sub={excluded || "none excluded"} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Findings" actions={<button className="btn" onClick={() => setRefresh((n) => n + 1)}>Re-run</button>}>
          <FindingsList findings={r.findings} />
        </Section>
        <div className="space-y-4">
          {r.feature_target.length > 0 && (
            <Section title="Feature relevance">
              <p className="mb-2 text-xs text-fg-3">Single-feature probe score ({r.feature_target[0].metric}). Near 1.0 = suspicious (possible leakage); near 0.5 = no signal.</p>
              <Bars items={r.feature_target.map((f) => ({ label: f.column, value: f.score ?? 0, note: `mutual information ${f.mutual_info}` }))} max={1} />
            </Section>
          )}
          <Section title="Numeric distributions">
            <div className="grid gap-4 sm:grid-cols-2">
              {r.column_stats.filter((c) => c.histogram).map((c) => (
                <div key={c.name}>
                  <div className="text-[13px]"><b>{c.name}</b> <span className="text-fg-3">median {c.median?.toPrecision(4)}{c.n_outliers ? `, ${c.n_outliers} outliers` : ""}</span></div>
                  <Histogram edges={c.histogram.edges} counts={c.histogram.counts} />
                </div>
              ))}
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

const ALGOS: [string, string][] = [
  ["logistic_regression", "Logistic Regression"], ["random_forest", "Random Forest"], ["xgboost", "XGBoost"], ["svm", "SVM (RBF)"],
];

function Models({ d }: { d: Dataset }) {
  const models = useResource<ModelSummary[]>(`/datasets/${d.id}/models`);
  const [algos, setAlgos] = useState<string[]>(ALGOS.map((a) => a[0]));
  const [drop, setDrop] = useState<string[]>([]);
  const [folds, setFolds] = useState(5);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId, () => models.reload());

  async function train() {
    setError(null);
    try {
      const j = await api<Job>(`/datasets/${d.id}/train`, { json: { algorithms: algos, drop_columns: drop, cv_folds: folds } });
      setJobId(j.id);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const running = !!job && (job.status === "queued" || job.status === "running");
  const featureCols = (d.columns || []).filter((c) => c.name !== d.target_column).map((c) => c.name);
  const skipped: { algorithm: string; reason: string }[] = job?.result?.skipped || [];

  return (
    <div className="space-y-4">
      <Section title="Train baseline models">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-4">
            {ALGOS.map(([k, label]) => (
              <label key={k} className="inline-flex items-center gap-1.5">
                <input type="checkbox" checked={algos.includes(k)} className="accent-accent"
                  onChange={(e) => setAlgos((a) => (e.target.checked ? [...a, k] : a.filter((x) => x !== k)))} />{label}
              </label>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <label className="inline-flex items-center gap-2">CV folds
              <select className="field" value={folds} onChange={(e) => setFolds(+e.target.value)}>
                {[0, 3, 5, 10].map((n) => <option key={n} value={n}>{n || "off"}</option>)}
              </select>
            </label>
            <details>
              <summary className="cursor-pointer text-[13px] text-fg-2">Exclude columns ({drop.length})</summary>
              <div className="mt-1.5 flex flex-wrap gap-3">
                {featureCols.map((c) => (
                  <label key={c} className="inline-flex items-center gap-1 text-[13px]">
                    <input type="checkbox" checked={drop.includes(c)} className="accent-accent"
                      onChange={(e) => setDrop((x) => (e.target.checked ? [...x, c] : x.filter((y) => y !== c)))} />{c}
                  </label>
                ))}
              </div>
            </details>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn btn-primary" disabled={!algos.length || running} onClick={train}>{running ? "Training…" : "Train"}</button>
            {job && <span className="text-[13px] text-fg-2">{running && <Spinner />} {job.status}: {job.progress || ""}</span>}
          </div>
          {job?.status === "failed" && <ErrorBox error={job.error} />}
          {skipped.length > 0 && <p className="text-xs text-fg-3">Skipped: {skipped.map((s) => `${s.algorithm} (${s.reason})`).join(", ")}</p>}
          <ErrorBox error={error} />
        </div>
      </Section>
      <Section title="Leaderboard (held-out test set, ranked by macro-F1)">
        {!models.data ? <Loading /> : models.data.length === 0 ? <Empty>No models trained yet.</Empty> : (
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Model</th><th>Stage</th><th className="num">Accuracy</th><th className="num">Balanced acc.</th><th className="num">Macro-F1</th><th className="num">ROC AUC</th><th className="num">Weakest class recall</th><th className="num">Findings</th></tr></thead>
              <tbody>
                {[...models.data].sort((a, b) => b.scores.f1_macro - a.scores.f1_macro).map((m) => (
                  <tr key={m.id}>
                    <td><Link to={`/models/${m.id}`}>{m.name}</Link> <span className="text-xs text-fg-3">{m.source}</span></td>
                    <td><StageBadge stage={m.stage} /></td>
                    <td className="num">{fmt.pct(m.scores.accuracy, 1)}</td>
                    <td className="num">{fmt.pct(m.scores.balanced_accuracy, 1)}</td>
                    <td className="num"><b>{fmt.num(m.scores.f1_macro)}</b></td>
                    <td className="num">{fmt.num(m.scores.roc_auc)}</td>
                    <td className="num">{m.scores.worst_class?.label}: {fmt.pct(m.scores.worst_class?.recall)}</td>
                    <td className="num">{m.n_findings}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

function DiagnosisView({ id }: { id: string }) {
  const diag = useResource<Diagnosis>(`/datasets/${id}/diagnosis`);
  if (diag.error) return <ErrorBox error={diag.error} />;
  if (!diag.data) return <Loading label="Diagnosing" />;
  const d = diag.data;
  return (
    <div className="space-y-4">
      <HealthPanel d={d} />
      <Section title="Summary">
        <div className="space-y-1.5">{d.narrative.map((l, i) => <p key={i}>{l}</p>)}</div>
        {d.model && <Link to={`/models/${d.model.id}`} className="mt-2 inline-block">Open {d.model.name} →</Link>}
      </Section>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Findings"><FindingsList findings={d.findings} /></Section>
        <Section title="Recommended fixes">
          {d.recommendations.length === 0 ? <Empty>Nothing to fix.</Empty> : (
            <div className="space-y-2.5">
              {d.recommendations.map((r) => (
                <div key={r.intervention} className={cx("rounded-r-lg border-l-[3px] bg-surface px-3 py-2.5",
                  r.severity === "critical" ? "border-l-critical" : r.severity === "warning" ? "border-l-warning" : "border-l-line")}>
                  <div className="flex items-center justify-between gap-2"><b>{r.label}</b><SeverityBadge severity={r.severity} /></div>
                  <div className="mt-1 text-[13px] text-fg-2">Because: {r.because.map((b) => b.title).join("; ")}</div>
                </div>
              ))}
              {d.model && <p className="text-xs text-fg-3">Run these automatically from the model page → Experiments.</p>}
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}
