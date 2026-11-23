import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Assistant } from "@/components/Assistant";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Bars, ConfusionMatrix, DivergingBars, Histogram, ScatterChart, ThresholdChart } from "@/components/charts";
import {
  Crumbs, Empty, ErrorBox, FindingsList, Loading, Section, SeverityBadge, Spinner, StageBadge, Stat, Tabs, cx,
} from "@/components/ui";
import { api, fmt } from "@/lib/api";
import { isAdmin, useAuth } from "@/lib/auth";
import { useJob, useResource } from "@/lib/hooks";
import type { Job, ModelDetail, ModelSummary } from "@/lib/types";

type Tab = "performance" | "explain" | "experiments" | "registry" | "monitoring" | "assistant";

export default function ModelPage() {
  const { id = "" } = useParams();
  const model = useResource<ModelDetail>(`/models/${id}`);
  const [tab, setTab] = useState<Tab>("performance");
  if (model.error) return <ErrorBox error={model.error} />;
  if (!model.data) return <Loading />;
  const m = model.data;
  return (
    <div>
      <Crumbs>
        <Link to="/">Projects</Link> / <Link to={`/projects/${m.project_id}`}>Project</Link> /{" "}
        <Link to={`/datasets/${m.dataset_id}`}>Dataset</Link> / <span>{m.name}</span>
      </Crumbs>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1>{m.name}</h1>
        <div className="flex gap-2"><StageBadge stage={m.stage} /><span className="badge">{m.source}</span></div>
      </div>
      <p className="mt-1 font-mono text-xs text-fg-2">{describe(m.config)}</p>
      <Tabs<Tab> value={tab} onChange={setTab} tabs={[
        ["performance", "Performance"], ["explain", "Explainability"], ["experiments", "Experiments"],
        ["registry", "Registry"], ["monitoring", "Monitoring"], ["assistant", "Assistant"],
      ]} />
      {tab === "performance" && <Performance m={m} />}
      {tab === "explain" && <Explain m={m} />}
      {tab === "experiments" && <Experiments m={m} />}
      {tab === "registry" && <Registry m={m} onChanged={model.reload} />}
      {tab === "monitoring" && <Monitoring m={m} />}
      {tab === "assistant" && <ErrorBoundary label="the assistant"><Assistant datasetId={m.dataset_id} modelId={m.id} /></ErrorBoundary>}
    </div>
  );
}

function describe(c: Record<string, any>) {
  const parts = [c.algorithm];
  if (c.class_weight) parts.push(`class_weight=${c.class_weight}`);
  if (c.resampling) parts.push(c.resampling);
  if (c.dedupe) parts.push("deduplicated");
  if (c.drop_columns?.length) parts.push(`without ${c.drop_columns.join(", ")}`);
  if (c.calibration) parts.push(`${c.calibration} calibration`);
  if (c.threshold != null) parts.push(`threshold ${c.threshold}`);
  if (c.params && Object.keys(c.params).length) parts.push(`tuned ${JSON.stringify(c.params)}`);
  return parts.join(" · ");
}

function RegressionPerformance({ m }: { m: ModelDetail }) {
  const t = m.metrics.test;
  const cv = m.metrics.cv;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="R² (primary)" value={fmt.num(t.r2)} sub={cv?.val?.r2 ? `CV ${fmt.num(cv.val.r2.mean)} ± ${fmt.num(cv.val.r2.std)}` : "no CV"} />
        <Stat label="RMSE" value={fmt.sig(t.rmse)} sub={`mean-only baseline ${fmt.sig(m.metrics.dummy?.rmse)}`} />
        <Stat label="MAE / MEDIAN AE" value={`${fmt.sig(t.mae)} / ${fmt.sig(t.median_ae)}`} sub={t.mape != null ? `MAPE ${fmt.pct(t.mape, 1)}` : "MAPE n/a (zeros in target)"} />
        <Stat label="ERROR PROFILE" value={fmt.pct(t.top5pct_error_share)} sub="share of total error from the worst 5% of rows" />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Diagnosis"><FindingsList findings={m.findings} empty="No model-level problems found." /></Section>
        <div className="space-y-4">
          <Section title="Predicted vs actual (test sample)">
            <ScatterChart points={t.points} />
          </Section>
          <Section title="Residuals (actual − predicted)">
            <Histogram edges={t.residual_histogram.edges} counts={t.residual_histogram.counts} />
            <p className="mt-1 text-xs text-fg-3">
              Mean residual {fmt.sig(t.residual_mean)} (bias) · std {fmt.sig(t.residual_std)} · error-vs-prediction correlation {fmt.num(t.error_vs_prediction_corr, 2)}
            </p>
          </Section>
        </div>
      </div>
      {m.importances && (
        <Section title="Model-native feature importance">
          <Bars items={m.importances.map((i) => ({ label: i.feature, value: i.importance }))} format={(v) => fmt.pct(v, 1)} />
        </Section>
      )}
    </div>
  );
}

function Performance({ m }: { m: ModelDetail }) {
  if (m.task === "regression") return <RegressionPerformance m={m} />;
  const t = m.metrics.test;
  const cv = m.metrics.cv;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="MACRO-F1 (primary)" value={fmt.num(t.f1_macro)} sub={cv ? `CV ${fmt.num(cv.val.f1_macro.mean)} ± ${fmt.num(cv.val.f1_macro.std)}` : "no CV"} />
        <Stat label="ACCURACY / BALANCED" value={`${fmt.pct(t.accuracy)} / ${fmt.pct(t.balanced_accuracy)}`}
          sub={`majority-class baseline: ${fmt.pct(m.metrics.dummy?.balanced_accuracy)} balanced`} />
        <Stat label={`RECALL · ${(t.positive_class || t.worst_class).label}`} value={fmt.pct((t.positive_class || t.worst_class).recall)}
          sub={`weakest class F1 ${fmt.num(t.worst_class.f1)}`} />
        <Stat label="ROC AUC / PR AUC" value={`${fmt.num(t.roc_auc)} / ${fmt.num(t.pr_auc)}`} sub={`calibration error (ECE) ${fmt.num(t.ece)}`} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Diagnosis"><FindingsList findings={m.findings} empty="No model-level problems found." /></Section>
        <div className="space-y-4">
          <Section title="Confusion matrix (test set)">
            <ConfusionMatrix labels={t.confusion_matrix.labels} matrix={t.confusion_matrix.matrix} />
          </Section>
          <Section title="Per-class performance">
            <table className="tbl">
              <thead><tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
              <tbody>{t.per_class.map((c: any) => (
                <tr key={c.label}><td>{c.label}</td><td className="num">{fmt.num(c.precision)}</td><td className="num">{fmt.num(c.recall)}</td>
                  <td className="num">{fmt.num(c.f1)}</td><td className="num">{c.support}</td></tr>
              ))}</tbody>
            </table>
          </Section>
        </div>
      </div>
      {t.threshold_analysis && (
        <Section title={`Decision threshold (positive class '${m.metrics.positive_label}')`}>
          <ThresholdChart curve={t.threshold_analysis.curve} best={t.threshold_analysis.best_threshold} current={t.threshold ?? 0.5} />
          <p className="mt-1 text-[13px] text-fg-2">
            F1 at 0.5: {fmt.num(t.threshold_analysis.f1_at_0_5)} · best {fmt.num(t.threshold_analysis.best_f1)} at {t.threshold_analysis.best_threshold}
            {t.threshold != null ? ` · this model uses ${t.threshold}` : ""}
          </p>
        </Section>
      )}
      {m.importances && (
        <Section title="Model-native feature importance">
          <Bars items={m.importances.map((i) => ({ label: i.feature, value: i.importance }))} format={(v) => fmt.pct(v, 1)} />
        </Section>
      )}
    </div>
  );
}

function Explain({ m }: { m: ModelDetail }) {
  const ex = useResource<any>(`/models/${m.id}/explanations`);
  const [row, setRow] = useState("");
  const [method, setMethod] = useState<"auto" | "lime">("auto");
  const [local, setLocal] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const regression = m.task === "regression";

  async function explain(rowId: number, how = method) {
    setError(null);
    try {
      setLocal(await api(`/models/${m.id}/explain`, { json: { row_id: rowId, method: how } }));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  if (ex.error) return <ErrorBox error={ex.error} />;
  if (!ex.data) return <Loading label="Computing SHAP, permutation importance and error analysis" />;
  const g = ex.data.global_importance;
  const ea = ex.data.error_analysis;
  const fair = ex.data.fairness;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="What drives the model">
          {g.shap ? (
            <>
              <p className="mb-2 text-xs text-fg-3">{g.shap.method}: mean |SHAP| {regression ? "on the prediction" : `for class '${g.shap.explained_class}'`}</p>
              <Bars items={g.shap.values.map((v: any) => ({ label: v.feature, value: v.mean_abs_shap }))} />
            </>
          ) : <p className="text-xs text-fg-3">SHAP isn't available for this model type; permutation importance below is model-agnostic.</p>}
          <p className="mt-4 mb-2 text-xs text-fg-3">Permutation importance: drop in {regression ? "R²" : "macro-F1"} when a feature is shuffled (baseline {g.permutation.baseline})</p>
          <Bars items={g.permutation.values.map((v: any) => ({ label: v.feature, value: Math.max(0, v.importance), note: `± ${v.std}` }))} />
        </Section>
        <Section title="Where the errors are">
          <div className="mb-2 text-[13px] text-fg-2">
            {regression
              ? `Mean absolute error ${fmt.sig(ea.summary.mae)} on ${ea.summary.n_test} test rows · "large error" = above ${fmt.sig(ea.summary.large_error_threshold)}`
              : `${ea.summary.n_errors} errors of ${ea.summary.n_test} (${fmt.pct(ea.summary.error_rate, 1)})`}
            {!regression && ea.summary.false_negatives != null && ` · ${ea.summary.false_negatives} false negatives · ${ea.summary.false_positives} false positives`}
          </div>
          <div className="space-y-1.5">{ea.statements.map((s: string) => <div key={s} className="notice text-[13px]">{s}</div>)}</div>
          {ea.slices.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="tbl">
                <thead><tr><th>Slice</th><th className="num">Rows</th><th className="num">{regression ? "MAE" : "Error rate"}</th><th className="num">Lift</th></tr></thead>
                <tbody>{ea.slices.slice(0, 8).map((s: any) => (
                  <tr key={s.feature + s.value}><td>{s.feature} = {s.value}</td><td className="num">{s.n}</td>
                    <td className="num">{regression ? fmt.sig(s.mae) : fmt.pct(s.error_rate)}</td><td className="num">{s.lift}×</td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </Section>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Explain a prediction">
          <div className="flex flex-wrap gap-2">
            <select className="field min-w-0 flex-1" value={row} onChange={(e) => { setRow(e.target.value); if (e.target.value) explain(+e.target.value); }}>
              <option value="">Pick a test row…</option>
              {ex.data.samples.map((s: any) => (
                <option key={s.row_id} value={s.row_id}>
                  {regression
                    ? `row ${s.row_id}: actual ${fmt.sig(s.true)}, predicted ${fmt.sig(s.predicted)} (error ${fmt.sig(s.abs_error)})`
                    : `row ${s.row_id}: true ${s.true}, predicted ${s.predicted} (${fmt.pct(s.confidence)})${s.correct ? "" : " ✗"}`}
                </option>
              ))}
            </select>
            <select className="field" value={method} aria-label="Explanation method"
              onChange={(e) => { const v = e.target.value as "auto" | "lime"; setMethod(v); if (row) explain(+row, v); }}>
              <option value="auto">SHAP (or occlusion)</option>
              <option value="lime">LIME</option>
            </select>
          </div>
          <ErrorBox error={error} />
          {local && (
            <div className="mt-3">
              <p className="font-semibold">{local.explanation}</p>
              <p className="mb-2 text-xs text-fg-3">
                Actual: {local.actual == null ? "unknown" : regression ? fmt.sig(local.actual) : local.actual} · method: {local.method} ({local.units})
                {local.fidelity != null && ` · local fit (fidelity) ${fmt.num(local.fidelity, 2)}`}
              </p>
              <DivergingBars posLabel={regression ? "pushes the prediction up" : `towards '${local.prediction}'`}
                negLabel={regression ? "pushes it down" : "away from it"}
                items={local.contributions.slice(0, 10).map((c: any) => ({ label: c.feature, value: c.contribution, detail: c.value }))} />
            </div>
          )}
        </Section>
        <Section title="Group performance">
          {fair.features.length === 0 ? <Empty>No categorical groups large enough to compare.</Empty> : (
            <div className="space-y-3">
              <p className="text-[13px] text-fg-2">
                {regression ? "Largest gap in mean absolute error, relative to the overall error" : "Largest balanced-accuracy gap"}:{" "}
                <b>{fmt.num(fair.max_gap)}</b> across '{fair.worst_feature}'.
              </p>
              {fair.features.slice(0, 3).map((f: any) => (
                <div key={f.feature}>
                  <div className="text-[13px]"><b>{f.feature}</b> <span className="text-fg-3">gap {f.gap}</span></div>
                  {regression ? (
                    <Bars format={(v) => fmt.sig(v)} items={f.groups.map((r: any) => ({ label: r.value, value: r.mae, note: `${r.n} rows · R² ${fmt.num(r.r2, 2)}` }))} />
                  ) : (
                    <Bars max={1} items={f.groups.filter((r: any) => r.balanced_accuracy != null).map((r: any) => (
                      { label: r.value, value: r.balanced_accuracy, note: `${r.n} rows · positive recall ${fmt.pct(r.positive_recall)}` }))} />
                  )}
                </div>
              ))}
            </div>
          )}
        </Section>
      </div>
      <Section title="Prediction findings"><FindingsList findings={ex.data.findings} empty="No prediction-level problems found." /></Section>
    </div>
  );
}

function Experiments({ m }: { m: ModelDetail }) {
  const list = useResource<Job[]>(`/models/${m.id}/experiments`);
  const [jobId, setJobId] = useState<string | null>(null);
  const [budget, setBudget] = useState("standard");
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId, () => list.reload());
  const running = !!job && (job.status === "queued" || job.status === "running");

  async function start() {
    setError(null);
    try {
      setJobId((await api<Job>(`/models/${m.id}/experiments`, { json: { budget } })).id);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const r = list.data?.find((j) => j.status === "completed")?.result;
  return (
    <div className="space-y-4">
      <Section title="Automated corrective experiments">
        <p className="mb-3 text-[13px] text-fg-2">
          ModelDoctor turns the diagnosis into a plan, runs each fix, and keeps a step only if it improves the <b>cross-validated</b>{" "}
          {m.task === "regression" ? "R²" : "macro-F1"} on training data. The test set is only used to report results.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <select className="field" value={budget} onChange={(e) => setBudget(e.target.value)}>
            <option value="standard">Standard (includes hyperparameter search)</option>
            <option value="quick">Quick</option>
          </select>
          <button className="btn btn-primary" disabled={running} onClick={start}>{running ? "Running…" : "Run experiments"}</button>
          {job && <span className="text-[13px] text-fg-2">{running && <Spinner />} {job.status}: {job.progress}</span>}
        </div>
        {job?.status === "failed" && <ErrorBox error={job.error} />}
        <ErrorBox error={error} />
      </Section>
      {r && (
        <Section title="Latest result">
          <div className="notice mb-3">
            <b>{r.best.description}</b><br />
            {r.task === "regression" ? (
              <>R² {fmt.num(r.baseline_primary)} → {fmt.num(r.best.metrics.r2)} · RMSE {fmt.sig(r.best.metrics.rmse - r.improvement.rmse)} → {fmt.sig(r.best.metrics.rmse)}</>
            ) : (
              <>Macro-F1 {fmt.num(r.baseline_primary)} → {fmt.num(r.best.metrics.f1_macro)} ({r.improvement.primary >= 0 ? "+" : ""}{fmt.num(r.improvement.primary)}) ·
              minority recall {fmt.pct(r.best.metrics.minority_recall - r.improvement.minority_recall)} → {fmt.pct(r.best.metrics.minority_recall)}</>
            )}
            <div className="mt-1.5 text-[13px]">{r.recommendation}</div>
            {r.best_model_id && <Link to={`/models/${r.best_model_id}`} className="mt-1.5 inline-block">Open candidate model v{r.best.version} →</Link>}
          </div>
          {r.plan.notes.map((n: string) => <p key={n} className="text-xs text-fg-3">{n}</p>)}
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Step</th><th>Trial</th>
                <th className="num">CV {r.task === "regression" ? "R²" : "macro-F1"} (selection)</th>
                <th className="num">Test {r.task === "regression" ? "R²" : "macro-F1"}</th>
                <th className="num">{r.task === "regression" ? "Test RMSE" : "Minority recall"}</th><th>Decision</th></tr></thead>
              <tbody>{r.trials.map((t: any) => (
                <tr key={t.step}>
                  <td>{t.step}</td><td>{t.name}</td>
                  <td className="num">{fmt.num(t.cv_primary)}</td><td className="num">{fmt.num(t.metrics.primary)}</td>
                  <td className="num">{r.task === "regression" ? fmt.sig(t.metrics.rmse) : fmt.pct(t.metrics.minority_recall)}</td>
                  <td className="text-xs">
                    {t.kept
                      ? <span className="inline-flex items-center gap-1 font-semibold"><span className="size-2 rounded-full bg-good" />✓ kept</span>
                      : <span className="text-fg-3">rejected</span>}{" "}
                    <span className="text-fg-3">{t.reason}</span>
                  </td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-fg-3">{r.selection}</p>
        </Section>
      )}
      {list.data && list.data.length === 0 && !job && <Empty>No experiments yet for this model.</Empty>}
    </div>
  );
}

const NEXT: Record<string, [string, string][]> = {
  candidate: [["approved", "Approve"], ["rejected", "Reject"], ["archived", "Archive"]],
  approved: [["production", "Promote to production"], ["rejected", "Reject"], ["candidate", "Back to candidate"]],
  production: [["archived", "Archive"]],
  rejected: [["candidate", "Reopen"], ["archived", "Archive"]],
  archived: [["candidate", "Restore"]],
};

function Registry({ m, onChanged }: { m: ModelDetail; onChanged: () => void }) {
  const { user, status } = useAuth();
  const admin = isAdmin(user, status);
  const events = useResource<any[]>(`/models/${m.id}/events`);
  const siblings = useResource<ModelSummary[]>(`/datasets/${m.dataset_id}/models`);
  const [note, setNote] = useState("");
  const [actor, setActor] = useState("");
  const [ack, setAck] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [other, setOther] = useState("");
  const [cmp, setCmp] = useState<any>(null);

  async function move(stage: string) {
    setError(null);
    try {
      await api(`/models/${m.id}/stage`, { json: { stage, note: note || null, actor: actor || null, acknowledge_risks: ack } });
      setNote(""); setAck(false); onChanged(); events.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function compare(otherId: string) {
    setOther(otherId);
    setCmp(otherId ? await api(`/models/compare/${otherId}/${m.id}`).catch((e) => ({ error: (e as Error).message })) : null);
  }
  const critical = m.findings.filter((f) => f.severity === "critical");
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Section title="Human approval">
        <p className="mb-2 text-[13px] text-fg-2">Current stage: <StageBadge stage={m.stage} />. Only an approved model can be promoted; promoting archives the previous production model.</p>
        {critical.length > 0 && (
          <div className="notice mb-2 text-[13px]">
            <SeverityBadge severity="critical" /> {critical.length} unresolved critical finding(s): {critical.map((f) => f.title).join("; ")}
            <label className="mt-1.5 flex items-center gap-1.5">
              <input type="checkbox" className="accent-accent" checked={ack} onChange={(e) => setAck(e.target.checked)} /> I acknowledge these risks
            </label>
          </div>
        )}
        <div className="space-y-2.5">
          <input className="field w-full" placeholder="Your name (ignored when signed in with an API key)" value={actor} onChange={(e) => setActor(e.target.value)} />
          <textarea className="field w-full resize-y" placeholder="Note (required to approve, reject or promote)" rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="flex flex-wrap gap-2">
            {NEXT[m.stage].map(([s, label]) => (
              <button key={s} className={cx("btn", (s === "production" || s === "approved") && "btn-primary")}
                disabled={!admin && (s === "production" || s === "approved")}
                title={!admin && (s === "production" || s === "approved") ? "Only admins can approve or promote models" : undefined}
                onClick={() => move(s)}>{label}</button>
            ))}
          </div>
          <ErrorBox error={error} />
        </div>
        <h3 className="mt-4 mb-1">History</h3>
        {(events.data || []).length === 0 ? <p className="text-xs text-fg-3">No transitions yet.</p> : (
          <table className="tbl"><tbody>{events.data!.map((e) => (
            <tr key={e.id}><td className="text-xs text-fg-3">{fmt.date(e.created_at)}</td><td className="text-xs">{e.from_stage} → <b>{e.to_stage}</b></td>
              <td className="text-xs">{e.actor}</td><td className="text-xs text-fg-2">{e.note}</td></tr>
          ))}</tbody></table>
        )}
      </Section>
      <Section title="Compare with another version">
        <select className="field w-full" value={other} onChange={(e) => compare(e.target.value)}>
          <option value="">Choose a model…</option>
          {(siblings.data || []).filter((s) => s.id !== m.id).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        {cmp?.error && <ErrorBox error={cmp.error} />}
        {cmp && !cmp.error && (
          <div className="mt-3">
            {cmp.summary.map((s: string) => <p key={s} className="mb-1 text-[13px]">{s}</p>)}
            <table className="tbl mt-2">
              <thead><tr><th>Metric</th><th className="num">v{cmp.a.version}</th><th className="num">v{cmp.b.version} (this)</th><th className="num">Δ</th></tr></thead>
              <tbody>{cmp.metrics.map((r: any) => (
                <tr key={r.metric}><td>{r.metric}</td><td className="num">{fmt.num(r.a)}</td><td className="num">{fmt.num(r.b)}</td>
                  <td className="num">{r.delta == null ? "–" : `${r.delta > 0 ? "+" : ""}${r.delta.toFixed(3)}`}{r.better === "b" ? " ✓" : ""}</td></tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

function Monitoring({ m }: { m: ModelDetail }) {
  const history = useResource<any[]>(`/models/${m.id}/drift`);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload() {
    if (!file) return;
    setBusy(true); setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      await api(`/models/${m.id}/monitor`, { form });
      history.reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const latest = history.data?.[0]?.report;
  return (
    <div className="space-y-4">
      <Section title="Check a production batch for drift">
        <p className="mb-3 text-[13px] text-fg-2">
          Upload a CSV with the model's feature columns. Include the target column too and ModelDoctor will also measure performance on the
          batch (concept drift). Raw batch data is not kept; only the report is stored.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input type="file" accept=".csv,text/csv" className="text-[13px] file:mr-3 file:cursor-pointer file:rounded-lg file:border file:border-line file:bg-surface-2 file:px-3 file:py-1.5 file:text-fg" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button className="btn btn-primary" disabled={!file || busy} onClick={upload}>{busy ? "Analysing…" : "Analyse drift"}</button>
        </div>
        <ErrorBox error={error} />
      </Section>
      {latest && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="DRIFT SCORE" value={latest.score} sub="100 = no drift" />
            <Stat label="DRIFTED FEATURES" value={`${latest.drifted_features} / ${latest.n_features}`} sub={`PSI ≥ ${latest.thresholds.psi_moderate}`} />
            <Stat label="PREDICTION MIX SHIFT" value={fmt.num(latest.prediction_drift.psi, 2)} sub={latest.prediction_drift.kind === "values" ? "PSI of predicted values vs test" : "PSI of predicted classes vs test"} />
            {latest.performance && (
              <Stat label={`BATCH ${latest.performance.metric === "r2" ? "R²" : "MACRO-F1"}`} value={fmt.num(latest.performance.batch_primary)}
                sub={`test ${fmt.num(latest.performance.test_primary)} (Δ ${fmt.num(latest.performance.delta_primary)})`} />
            )}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Section title="Population stability by feature (PSI)">
              <Bars format={(v) => v.toFixed(3)} items={latest.features.map((f: any) => ({
                label: f.feature, value: f.psi, note: `${f.severity} · ${f.test} p=${f.p_value == null ? "–" : f.p_value.toExponential(2)}`,
              }))} />
            </Section>
            <Section title="Drift findings"><FindingsList findings={latest.findings} empty="No drift detected." /></Section>
          </div>
        </>
      )}
      {history.data && history.data.length > 1 && (
        <Section title="History">
          <table className="tbl">
            <thead><tr><th>When</th><th>Batch</th><th className="num">Rows</th><th className="num">Drift score</th></tr></thead>
            <tbody>{history.data.map((h) => (
              <tr key={h.id}><td className="text-xs">{fmt.date(h.created_at)}</td><td>{h.report.batch.filename}</td>
                <td className="num">{h.report.batch.n_rows}</td><td className="num">{h.report.score}</td></tr>
            ))}</tbody>
          </table>
        </Section>
      )}
    </div>
  );
}
