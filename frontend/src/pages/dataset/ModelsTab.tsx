import { useQueryClient } from "@tanstack/react-query";
import { Boxes, Play, ShieldCheck, Trophy, Upload } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FileDrop } from "@/components/domain/FileDrop";
import { QueryView } from "@/components/domain/QueryView";
import { StageBadge } from "@/components/domain/severity";
import {
  Badge, Button, Card, EmptyState, InlineError, Spinner, TBody, TD, TH, THead, TR, Table, TableSkeleton, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { cx } from "@/lib/cx";
import { isRunning, qk, useAlgorithms, useDatasetModels, useJob, useTrain, useUploadModel } from "@/lib/queries";
import type { Dataset, TrainJob } from "@/lib/types";

export function ModelsTab({ d }: { d: Dataset }) {
  const regression = d.task_type === "regression";
  const models = useDatasetModels(d.id);
  return (
    <div className="space-y-5">
      <TrainCard d={d} />
      <Card title="Leaderboard" description={`Held-out test set, ranked by ${regression ? "R²" : "macro-F1"}.`} flush>
        <QueryView query={models} label="Loading models" skeleton={<TableSkeleton rows={4} cols={7} />}
          isEmpty={(m) => m.length === 0}
          empty={<EmptyState compact icon={Boxes} title="No models trained yet" description="Pick algorithms above and train baselines." />}>
          {(list) => {
            const ranked = [...list].sort((a, b) => b.scores.primary - a.scores.primary);
            return (
              <Table label="Model leaderboard">
                <THead>
                  {regression ? (
                    <TR><TH>#</TH><TH>Model</TH><TH>Stage</TH><TH numeric>R²</TH><TH numeric>RMSE</TH><TH numeric>MAE</TH><TH numeric>MAPE</TH><TH numeric>Findings</TH></TR>
                  ) : (
                    <TR><TH>#</TH><TH>Model</TH><TH>Stage</TH><TH numeric>Accuracy</TH><TH numeric>Balanced acc.</TH><TH numeric>Macro-F1</TH><TH numeric>ROC AUC</TH><TH numeric>Weakest-class recall</TH><TH numeric>Findings</TH></TR>
                  )}
                </THead>
                <TBody>
                  {ranked.map((m, i) => (
                    <TR key={m.id}>
                      <TD className="font-mono text-fg-3">{i === 0 ? <Trophy size={14} aria-label="Best" className="text-warning" /> : i + 1}</TD>
                      <TD className="whitespace-nowrap"><Link to={`/app/models/${m.id}`} className="text-fg">{m.name}</Link> <Badge className="ml-1">{m.source}</Badge></TD>
                      <TD><StageBadge stage={m.stage} /></TD>
                      {regression ? (
                        <>
                          <TD numeric className="text-fg">{fmt.num(m.scores.r2)}</TD>
                          <TD numeric>{fmt.sig(m.scores.rmse)}</TD>
                          <TD numeric>{fmt.sig(m.scores.mae)}</TD>
                          <TD numeric>{fmt.pct(m.scores.mape, 1)}</TD>
                        </>
                      ) : (
                        <>
                          <TD numeric>{fmt.pct(m.scores.accuracy, 1)}</TD>
                          <TD numeric>{fmt.pct(m.scores.balanced_accuracy, 1)}</TD>
                          <TD numeric className="font-medium text-fg">{fmt.num(m.scores.f1_macro)}</TD>
                          <TD numeric>{fmt.num(m.scores.roc_auc)}</TD>
                          <TD numeric>{m.scores.worst_class?.label}: {fmt.pct(m.scores.worst_class?.recall)}</TD>
                        </>
                      )}
                      <TD numeric>{m.n_findings}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            );
          }}
        </QueryView>
      </Card>
      <UploadModelCard d={d} />
    </div>
  );
}

function TrainCard({ d }: { d: Dataset }) {
  const task = d.task_type === "regression" ? "regression" : "classification";
  const catalogue = useAlgorithms(task);
  const qc = useQueryClient();
  const toast = useToast();
  const train = useTrain(d.id);
  const [algos, setAlgos] = useState<string[] | null>(null);
  const [drop, setDrop] = useState<string[]>([]);
  const [folds, setFolds] = useState(5);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob<TrainJob>(jobId, (j) => {
    qc.invalidateQueries({ queryKey: qk.datasetModels(d.id) });
    qc.invalidateQueries({ queryKey: qk.diagnosis(d.id) });
    qc.invalidateQueries({ queryKey: ["project", d.project_id] });
    toast(j.status === "completed"
      ? { title: "Training complete", description: `${j.result?.model_ids.length ?? 0} models added to the leaderboard.`, tone: "success" }
      : { title: "Training failed", description: j.error ?? undefined, tone: "danger" });
  });
  const chosen = algos ?? (catalogue.data ?? []).filter((a) => a.default).map((a) => a.key);
  const running = isRunning(job);
  const featureCols = (d.columns ?? []).filter((c) => c.name !== d.target_column).map((c) => c.name);
  const toggle = (k: string, on: boolean) => setAlgos(on ? [...chosen, k] : chosen.filter((x) => x !== k));

  async function start() {
    const j = await train.mutateAsync({ algorithms: chosen, drop_columns: drop, cv_folds: folds }).catch(() => null);
    if (j) setJobId(j.id);
  }

  return (
    <Card title="Train baseline models" description="Each model is evaluated on the same fixed, stratified hold-out split.">
      <div className="space-y-4">
        <fieldset>
          <legend className="mb-2 text-xs font-medium text-fg-3">Algorithms</legend>
          {catalogue.isPending ? <div className="flex gap-2"><Spinner /><span className="text-sm text-fg-3">Loading algorithms…</span></div> : (
            <div className="flex flex-wrap gap-2">
              {(catalogue.data ?? []).map((a) => {
                const on = chosen.includes(a.key);
                return (
                  <label key={a.key} title={a.note || undefined}
                    className={cx(
                      "inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border px-2.5 text-sm transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-ring",
                      on ? "border-accent/60 bg-accent-subtle text-fg" : "border-line text-fg-2 hover:bg-surface-2",
                      !a.available && "cursor-not-allowed opacity-50",
                    )}>
                    <input type="checkbox" checked={on} disabled={!a.available} className="size-3.5 accent-[var(--accent)]"
                      onChange={(e) => toggle(a.key, e.target.checked)} />
                    {a.label}
                    {a.note && <span className="text-xs text-fg-3">({a.note})</span>}
                  </label>
                );
              })}
            </div>
          )}
        </fieldset>
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs font-medium text-fg-3">
            <span className="mb-1 block">Cross-validation folds</span>
            <select className="field" value={folds} onChange={(e) => setFolds(+e.target.value)}>
              {[0, 3, 5, 10].map((n) => <option key={n} value={n}>{n || "off"}</option>)}
            </select>
          </label>
          <details className="min-w-0 flex-1">
            <summary className="cursor-pointer text-sm text-fg-2">Exclude columns <span className="font-mono text-fg-3">({drop.length})</span></summary>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5">
              {featureCols.map((c) => (
                <label key={c} className="inline-flex items-center gap-1.5 font-mono text-xs text-fg-2">
                  <input type="checkbox" checked={drop.includes(c)} className="accent-[var(--accent)]"
                    onChange={(e) => setDrop((x) => (e.target.checked ? [...x, c] : x.filter((y) => y !== c)))} />{c}
                </label>
              ))}
            </div>
          </details>
        </div>
        <div className="flex flex-wrap items-center gap-3 border-t border-line pt-4">
          <Button variant="primary" icon={<Play size={14} />} disabled={!chosen.length} loading={running || train.isPending} onClick={start}>
            {running ? "Training…" : "Train"}
          </Button>
          <span className="text-sm text-fg-2" aria-live="polite">
            {job && <>{job.status}{job.progress ? `: ${job.progress}` : ""}</>}
          </span>
        </div>
        {job?.status === "failed" && <InlineError error={job.error} />}
        {job?.result?.skipped && job.result.skipped.length > 0 && (
          <p className="text-xs text-fg-3">Skipped: {job.result.skipped.map((s) => `${s.algorithm} (${s.reason})`).join(", ")}</p>
        )}
        <InlineError error={train.error} />
      </div>
    </Card>
  );
}

function UploadModelCard({ d }: { d: Dataset }) {
  const navigate = useNavigate();
  const upload = useUploadModel(d.id);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  async function submit() {
    if (!file) return;
    const m = await upload.mutateAsync({ file, name: name.trim() || undefined }).catch(() => null);
    if (m) navigate(`/app/models/${m.id}`);
  }
  return (
    <Card title="Bring your own model" description="Diagnose a model you trained elsewhere, on this dataset's split.">
      <div className="grid gap-5 md:grid-cols-2">
        <div className="min-w-0 space-y-2 break-words text-sm text-fg-2">
          <p>Save it with <code className="rounded-sm bg-surface-2 px-1 text-xs">skops.io.dump</code>. It must accept a DataFrame with this dataset's feature columns.</p>
          <p className="flex items-start gap-2">
            <ShieldCheck size={15} aria-hidden className="mt-0.5 shrink-0 text-success" />
            <span>Every type in the file is checked against an allowlist (scikit-learn, XGBoost, LightGBM, imbalanced-learn, NumPy, SciPy) <em>before</em> loading. Pickle and joblib files are refused because they can run code.</span>
          </p>
        </div>
        <div className="min-w-0 space-y-3">
          <FileDrop accept=".skops" file={file} onFile={setFile} label="Choose a .skops file" disabled={upload.isPending} />
          <div className="flex gap-2">
            <label htmlFor="model-name" className="sr-only">Model name (optional)</label>
            <input id="model-name" className="field min-w-0 flex-1" placeholder="Name (optional)" maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />
            <Button variant="primary" icon={<Upload size={14} />} disabled={!file} loading={upload.isPending} onClick={submit}>
              {upload.isPending ? "Evaluating…" : "Upload & diagnose"}
            </Button>
          </div>
          <InlineError error={upload.error} />
        </div>
      </div>
    </Card>
  );
}
