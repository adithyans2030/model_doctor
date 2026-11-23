import { Boxes, Database, History, Trash2, Upload } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FileDrop } from "@/components/domain/FileDrop";
import { Meta, PageHeader } from "@/components/domain/PageHeader";
import { QueryView } from "@/components/domain/QueryView";
import { StageBadge } from "@/components/domain/severity";
import { TrendChart } from "@/components/TrendChart";
import {
  Badge, Button, Card, CardSkeleton, ChartSkeleton, ConfirmDialog, EmptyState, ErrorState, InlineError, TBody, TD, TH,
  THead, TR, Table, TableSkeleton, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { useDeleteProject, useProject, useProjectDatasets, useRegistry, useUploadDataset } from "@/lib/queries";
import type { ModelSummary, Registry } from "@/lib/types";

export default function ProjectPage() {
  const { id = "" } = useParams();
  const project = useProject(id);
  const datasets = useProjectDatasets(id);
  const registry = useRegistry(id);
  const navigate = useNavigate();
  const toast = useToast();
  const remove = useDeleteProject();
  const [confirm, setConfirm] = useState(false);

  if (project.isError) return <ErrorState error={project.error} onRetry={() => project.refetch()} />;
  if (project.isPending) return <CardSkeleton lines={6} />;
  const p = project.data;

  return (
    <div className="space-y-5">
      <PageHeader
        crumbs={[{ label: "Projects", to: "/app" }, { label: p.name }]}
        title={p.name}
        meta={<>
          {p.description && <span>{p.description}</span>}
          <Meta label="datasets">{p.dataset_count}</Meta>
          <Meta label="created">{new Date(p.created_at).toLocaleDateString()}</Meta>
        </>}
        actions={<Button variant="danger" size="sm" icon={<Trash2 size={14} />} onClick={() => setConfirm(true)}>Delete project</Button>}
      />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <UploadCard projectId={id} />
        <Card title="Datasets" flush>
          <QueryView query={datasets} label="Loading datasets" skeleton={<TableSkeleton rows={2} cols={4} />}
            isEmpty={(d) => d.length === 0}
            empty={<EmptyState compact icon={Database} title="No datasets yet" description="Upload a CSV to profile it." />}>
            {(list) => (
              <Table label="Datasets">
                <THead><TR><TH>File</TH><TH numeric>Rows</TH><TH>Target</TH><TH>Task</TH></TR></THead>
                <TBody>
                  {list.map((d) => (
                    <TR key={d.id}>
                      <TD><Link to={`/app/datasets/${d.id}`} className="font-mono text-sm">{d.original_filename}</Link></TD>
                      <TD numeric>{fmt.int(d.n_rows)}</TD>
                      <TD>{d.target_column ? <span className="font-mono text-sm">{d.target_column}</span> : <span className="text-fg-3">not set</span>}</TD>
                      <TD className="text-xs text-fg-2">{d.task_type?.replace("_", " ") || "–"}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            )}
          </QueryView>
        </Card>
      </div>

      <QueryView query={registry} label="Loading model registry"
        skeleton={<div className="space-y-5"><CardSkeleton lines={2} /><Card title="Model registry" flush><TableSkeleton rows={4} cols={6} /></Card></div>}>
        {(r) => <RegistryView registry={r} />}
      </QueryView>

      <ConfirmDialog open={confirm} title="Delete this project?" confirmLabel="Delete project" busy={remove.isPending}
        description="Its datasets, models, experiments and approval history are deleted too. This can't be undone."
        onCancel={() => setConfirm(false)}
        onConfirm={async () => {
          const ok = await remove.mutateAsync(id).then(() => true, () => false);
          if (ok) { toast({ title: `Deleted ${p.name}`, tone: "success" }); navigate("/app"); }
        }}>
        <InlineError error={remove.error} />
      </ConfirmDialog>
    </div>
  );
}

function UploadCard({ projectId }: { projectId: string }) {
  const navigate = useNavigate();
  const upload = useUploadDataset(projectId);
  const [file, setFile] = useState<File | null>(null);
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    const ds = await upload.mutateAsync(file).catch(() => null);
    if (ds) navigate(`/app/datasets/${ds.id}`);
  }
  return (
    <Card title="Upload a dataset" description="CSV only: comma, semicolon, tab or pipe separated.">
      <form className="space-y-3" onSubmit={submit}>
        <FileDrop accept=".csv,text/csv" file={file} onFile={setFile} label="Choose a CSV" disabled={upload.isPending}
          hint="ModelDoctor profiles the file, infers column types and suggests a target." />
        <InlineError error={upload.error} />
        <Button type="submit" variant="primary" icon={<Upload size={14} />} loading={upload.isPending} disabled={!file}>
          {upload.isPending ? "Uploading and profiling…" : "Upload & profile"}
        </Button>
      </form>
    </Card>
  );
}

const metricOf = (m: ModelSummary) => (m.task === "regression" ? m.scores.r2 : m.scores.f1_macro) ?? m.scores.primary;

function RegistryView({ registry: r }: { registry: Registry }) {
  if (r.models.length === 0) {
    return (
      <Card title="Model registry">
        <EmptyState compact icon={Boxes} title="No models yet" description="Train baselines from a dataset; they appear here as candidates." />
      </Card>
    );
  }
  const names = Object.fromEntries(r.models.map((m) => [m.id, m.name]));
  const regression = r.models[0].task === "regression";
  const metricName = regression ? "R²" : "Macro-F1";
  const byVersion = [...r.models].sort((a, b) => a.version - b.version);
  const prod = r.production;
  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card title="In production">
          {prod ? (
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <Link to={`/app/models/${prod.id}`} className="text-md font-medium text-fg">{prod.name}</Link>
                <StageBadge stage="production" />
              </div>
              <dl className="mt-4 grid grid-cols-3 gap-3">
                {(regression
                  ? [["R²", fmt.num(prod.scores.r2)], ["RMSE", fmt.sig(prod.scores.rmse)], ["MAE", fmt.sig(prod.scores.mae)]]
                  : [["Macro-F1", fmt.num(prod.scores.f1_macro)], ["Balanced acc.", fmt.pct(prod.scores.balanced_accuracy, 1)],
                    [`Recall · ${prod.scores.worst_class?.label ?? "–"}`, fmt.pct(prod.scores.worst_class?.recall)]]
                ).map(([k, v]) => (
                  <div key={k}><dt className="text-xs text-fg-3">{k}</dt><dd className="mt-0.5 font-mono text-lg">{v}</dd></div>
                ))}
              </dl>
              <p className="mt-4 text-xs text-fg-3">Models move candidate → approved (human sign-off) → production. Promoting archives the previous production model.</p>
            </div>
          ) : (
            <EmptyState compact title="Nothing in production yet" description="Approve a candidate, then promote it from its Approval tab." />
          )}
        </Card>
        <Card title={`${metricName} by version`} description="Held-out test set. A high score from a model with critical findings (such as target leakage) is not trustworthy; check its findings.">
          {byVersion.length < 2 ? <ChartSkeleton height={120} /> : (
            <TrendChart label={`${metricName} by model version`} yLabel={metricName} domain={[0, 1]} height={180}
              points={byVersion.map((m) => ({ x: `v${m.version}`, y: +(metricOf(m) ?? 0).toFixed(3), detail: `${m.name} · ${m.stage}` }))} />
          )}
        </Card>
      </div>

      <Card title="Model registry" description={`${r.models.length} versions`} flush>
        <Table label="Model registry">
          <THead>
            <TR>
              <TH>Version</TH><TH>Model</TH><TH>Stage</TH><TH>Source</TH>
              {regression ? <><TH numeric>R²</TH><TH numeric>RMSE</TH></> : <><TH numeric>Macro-F1</TH><TH numeric>Balanced acc.</TH><TH numeric>Minority recall</TH></>}
              <TH numeric>Findings</TH>
            </TR>
          </THead>
          <TBody>
            {r.models.map((m) => (
              <TR key={m.id} className={m.stage === "production" ? "bg-success-subtle/40" : undefined}>
                <TD className="font-mono text-fg-2">v{m.version}</TD>
                <TD><Link to={`/app/models/${m.id}`} className="text-fg">{m.name}</Link></TD>
                <TD><StageBadge stage={m.stage} /></TD>
                <TD><Badge>{m.source}</Badge></TD>
                {regression ? (
                  <><TD numeric>{fmt.num(m.scores.r2)}</TD><TD numeric>{fmt.sig(m.scores.rmse)}</TD></>
                ) : (
                  <><TD numeric>{fmt.num(m.scores.f1_macro)}</TD><TD numeric>{fmt.num(m.scores.balanced_accuracy)}</TD><TD numeric>{fmt.pct(m.scores.worst_class?.recall)}</TD></>
                )}
                <TD numeric>{m.n_findings}</TD>
              </TR>
            ))}
          </TBody>
        </Table>
      </Card>

      <Card title="Audit trail" description="Every stage change, who made it and why.">
        {r.events.length === 0 ? (
          <EmptyState compact icon={History} title="No stage changes yet" />
        ) : (
          <ol className="relative space-y-4 border-l border-line pl-5">
            {r.events.map((e) => (
              <li key={e.id} className="relative">
                <span aria-hidden className="absolute top-1.5 -left-[24.5px] size-2 rounded-full border-2 border-surface bg-fg-3" />
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="text-fg">{names[e.model_id] || e.model_id.slice(0, 8)}</span>
                  {e.from_stage && <><StageBadge stage={e.from_stage} /><span aria-label="to" className="text-fg-3">→</span></>}
                  <StageBadge stage={e.to_stage} />
                </div>
                <p className="mt-1 text-xs text-fg-3">
                  {e.actor} · <time dateTime={e.created_at}>{fmt.date(e.created_at)}</time>{e.note && <> · <span className="text-fg-2">“{e.note}”</span></>}
                </p>
              </li>
            ))}
          </ol>
        )}
      </Card>
    </div>
  );
}
