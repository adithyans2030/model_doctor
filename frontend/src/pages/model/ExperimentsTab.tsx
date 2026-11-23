import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check, FlaskConical, Info, Minus, Play } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Dumbbell } from "@/components/charts";
import { QueryView } from "@/components/domain/QueryView";
import {
  Badge, Button, Card, CardSkeleton, EmptyState, InlineError, MetricTile, TBody, TD, TH, THead, TR, Table, buttonClasses, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { cx } from "@/lib/cx";
import { isRunning, qk, useJob, useModelExperiments, useStartExperiment } from "@/lib/queries";
import type { ExperimentJob, ExperimentResult, ModelDetail, TrialMetrics } from "@/lib/types";

export function ExperimentsTab({ m }: { m: ModelDetail }) {
  const list = useModelExperiments(m.id);
  const qc = useQueryClient();
  const toast = useToast();
  const start = useStartExperiment(m.id);
  const [budget, setBudget] = useState<"standard" | "quick">("standard");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob<ExperimentJob>(jobId, (j) => {
    qc.invalidateQueries({ queryKey: qk.experiments(m.id) });
    qc.invalidateQueries({ queryKey: qk.datasetModels(m.dataset_id) });
    qc.invalidateQueries({ queryKey: qk.diagnosis(m.dataset_id) });
    qc.invalidateQueries({ queryKey: ["project", m.project_id] });
    toast(j.status === "completed"
      ? { title: "Experiments finished", description: j.result?.recommendation, tone: "success" }
      : { title: "Experiments failed", description: j.error ?? undefined, tone: "danger" });
  });
  const running = isRunning(job);
  const metric = m.task === "regression" ? "R²" : "macro-F1";

  async function run() {
    const j = await start.mutateAsync(budget).catch(() => null);
    if (j) setJobId(j.id);
  }

  return (
    <div className="space-y-5">
      <Card title="Automated corrective experiments"
        description={`Turns the diagnosis into a plan, applies one fix at a time, and keeps a step only if cross-validated ${metric} on the training split improves. The test set is used only to report results.`}>
        <div className="flex flex-wrap items-center gap-3">
          <label htmlFor="budget" className="sr-only">Search budget</label>
          <select id="budget" className="field" value={budget} onChange={(e) => setBudget(e.target.value as "standard" | "quick")}>
            <option value="standard">Standard (includes hyperparameter search)</option>
            <option value="quick">Quick</option>
          </select>
          <Button variant="primary" icon={<Play size={14} />} loading={running || start.isPending} onClick={run}>
            {running ? "Running…" : "Run experiments"}
          </Button>
          <span className="text-sm text-fg-2" aria-live="polite">{job && <>{job.status}{job.progress ? `: ${job.progress}` : ""}</>}</span>
        </div>
        {job?.status === "failed" && <InlineError error={job.error} className="mt-3" />}
        <InlineError error={start.error} className="mt-3" />
      </Card>

      <QueryView query={list} label="Loading experiments" skeleton={<CardSkeleton lines={8} />}
        isEmpty={(l) => !l.some((j) => j.status === "completed" && j.result)}
        empty={running ? <CardSkeleton lines={8} /> : <Card><EmptyState icon={FlaskConical} title="No experiments yet" description="Run the optimiser to test the recommended fixes against a leak-free baseline." /></Card>}>
        {(l) => <ExperimentResultView r={l.find((j) => j.status === "completed" && j.result)!.result!} />}
      </QueryView>
    </div>
  );
}

type Row = { key: keyof TrialMetrics; label: string; fmt: (v: number | null | undefined) => string; lowerIsBetter?: boolean; unit?: boolean };

const CLASSIFICATION_ROWS: Row[] = [
  { key: "minority_recall", label: "Minority-class recall", fmt: (v) => fmt.pct(v), unit: true },
  { key: "f1_macro", label: "Macro-F1", fmt: (v) => fmt.num(v), unit: true },
  { key: "balanced_accuracy", label: "Balanced accuracy", fmt: (v) => fmt.num(v), unit: true },
  { key: "roc_auc", label: "ROC AUC", fmt: (v) => fmt.num(v), unit: true },
  { key: "pr_auc", label: "PR AUC", fmt: (v) => fmt.num(v), unit: true },
];
const REGRESSION_ROWS: Row[] = [
  { key: "r2", label: "R²", fmt: (v) => fmt.num(v), unit: true },
  { key: "rmse", label: "RMSE", fmt: (v) => fmt.sig(v), lowerIsBetter: true },
  { key: "mae", label: "MAE", fmt: (v) => fmt.sig(v), lowerIsBetter: true },
];

function ExperimentResultView({ r }: { r: ExperimentResult }) {
  const regression = r.task === "regression";
  const rows = regression ? REGRESSION_ROWS : CLASSIFICATION_ROWS;
  const base = r.trials[0].metrics;
  const best = r.best.metrics;
  const val = (t: TrialMetrics, k: keyof TrialMetrics) => (typeof t[k] === "number" ? (t[k] as number) : null);
  const tiles = rows.filter((row) => val(best, row.key) != null).slice(0, 4);
  const cvRange = r.trials.map((t) => t.cv_primary);
  const cvMin = Math.min(...cvRange), cvMax = Math.max(...cvRange);
  const metric = regression ? "R²" : "macro-F1";

  return (
    <div className="space-y-5">
      <section aria-labelledby="exp-result" className="rounded-lg border border-line bg-surface">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <h2 id="exp-result" className="text-sm font-semibold">Latest result</h2>
            <p className="mt-0.5 font-mono text-xs text-fg-3">{r.best.description}</p>
          </div>
          {r.best_model_id && (
            <Link to={`/app/models/${r.best_model_id}?tab=approval`} className={buttonClasses({ variant: "primary", size: "sm" })}>
              Review candidate v{r.best.version} <ArrowRight size={14} aria-hidden />
            </Link>
          )}
        </div>
        <div className="p-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {tiles.map((row, i) => {
              const before = val(base, row.key), after = val(best, row.key);
              return (
                <MetricTile key={row.key} emphasis={i === 0} label={row.label} before={row.fmt(before)} value={row.fmt(after)}
                  delta={before != null && after != null ? {
                    value: after - before, higherIsBetter: !row.lowerIsBetter, label: "vs leak-free baseline",
                    format: row.key === "minority_recall" ? (v) => `${(v * 100).toFixed(0)} pts` : row.unit ? (v) => v.toFixed(3) : (v) => fmt.sig(v),
                  } : undefined} />
              );
            })}
          </div>
          <p className="mt-4 text-sm text-fg">{r.recommendation}</p>
          <p className="mt-2 flex items-start gap-1.5 text-xs text-fg-3">
            <Info size={13} aria-hidden className="mt-px shrink-0" />
            <span>
              Before = the original configuration with leaking features removed ({r.plan.leaky_columns.length ? r.plan.leaky_columns.join(", ") : "none found"}).
              The original model scored {fmt.num(r.base_model_primary)} {metric} only because of the leak, so that number isn't comparable.
            </span>
          </p>
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card title="Before vs after" description="Held-out test set, same split for both.">
          <Dumbbell label="Metrics before and after the kept fixes" beforeLabel="Leak-free baseline" afterLabel={`Best (v${r.best.version ?? "–"})`}
            rows={rows.filter((row) => row.unit).map((row) => ({ metric: row.label, before: val(base, row.key), after: val(best, row.key) }))} />
          {!regression && <p className="mt-4 text-xs text-fg-3">Per-trial precision isn't reported by the experiment API; open the candidate's Performance tab for its per-class precision.</p>}
        </Card>

        <Card title="Trials" description={`Greedy search: each step builds on the last kept one. Bars show cross-validated ${metric}, the selection criterion.`} flush>
          <Table label="Experiment trials">
            <THead>
              <TR><TH>Step</TH><TH>Trial</TH><TH>CV {metric}</TH><TH numeric>Test {metric}</TH>{!regression && <TH numeric>Minority recall</TH>}<TH>Decision</TH></TR>
            </THead>
            <TBody>
              {r.trials.map((t) => {
                const w = cvMax > cvMin ? 0.15 + 0.85 * ((t.cv_primary - cvMin) / (cvMax - cvMin)) : 1;
                return (
                  <TR key={t.step} className={t.step === r.best.step ? "bg-accent-subtle/60" : undefined}>
                    <TD className="font-mono text-fg-3">{t.step}</TD>
                    <TD className="min-w-40">
                      <span className="text-fg">{t.name}</span>
                      <span className="mt-0.5 block text-xs text-fg-3">{t.reason}</span>
                    </TD>
                    <TD className="min-w-36">
                      <span className="flex items-center gap-2">
                        <span aria-hidden className="h-1.5 w-20 overflow-hidden rounded-full bg-surface-3">
                          <span className={cx("block h-full rounded-full", t.kept ? "bg-series-1" : "bg-series-muted")} style={{ width: `${w * 100}%` }} />
                        </span>
                        <span className="font-mono text-xs">{fmt.num(t.cv_primary)}</span>
                      </span>
                    </TD>
                    <TD numeric>{fmt.num(t.metrics.primary)}</TD>
                    {!regression && <TD numeric>{fmt.pct(t.metrics.minority_recall)}</TD>}
                    <TD>
                      {t.kept
                        ? <Badge tone="success" icon={<Check size={11} />}>{t.step === 0 ? "baseline" : "kept"}</Badge>
                        : <Badge icon={<Minus size={11} />}>rejected</Badge>}
                    </TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>
          <div className="space-y-1 border-t border-line px-4 py-3 text-xs text-fg-3">
            {r.plan.notes.map((n) => <p key={n}>{n}</p>)}
            <p>{r.selection}</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
