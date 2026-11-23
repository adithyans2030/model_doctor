import { CheckCircle2, Crosshair, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { Bars, Histogram, SegmentedBar } from "@/components/charts";
import { QueryView } from "@/components/domain/QueryView";
import {
  Badge, Button, Card, EmptyState, InlineError, MetricTile, TBody, TD, TH, THead, TR, Table, TableSkeleton, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { usePreview, useSetTarget } from "@/lib/queries";
import type { Dataset, TargetSummary } from "@/lib/types";

const TYPE_COLOR: Record<string, string> = {
  numeric: "var(--series-1)",
  categorical: "var(--series-2)",
  boolean: "var(--series-3)",
  datetime: "var(--seq-1)",
};
const typeColor = (t: string) => TYPE_COLOR[t] ?? "var(--series-muted)";

export function OverviewTab({ d }: { d: Dataset }) {
  const columns = d.columns ?? [];
  const missingCells = columns.reduce((a, c) => a + c.n_missing, 0);
  const missingPct = d.n_rows && columns.length ? missingCells / (d.n_rows * columns.length) : 0;
  const typeCounts = Object.entries(columns.reduce<Record<string, number>>((a, c) => ({ ...a, [c.inferred_type]: (a[c.inferred_type] ?? 0) + 1 }), {}))
    .sort((a, b) => b[1] - a[1]);
  const missing = columns.filter((c) => c.missing_pct > 0).sort((a, b) => b.missing_pct - a.missing_pct);
  const dist = d.class_distribution ?? null;
  const shares = dist?.map((c) => c.share) ?? [];
  const imbalance = shares.length > 1 ? Math.max(...shares) / Math.max(Math.min(...shares), 1e-9) : null;

  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile label="Rows" value={fmt.int(d.n_rows)} hint={d.size_bytes ? `${fmt.int(Math.round(d.size_bytes / 1024))} KB uploaded` : undefined} />
        <MetricTile label="Columns" value={columns.length || d.n_cols} hint={typeCounts.map(([k, v]) => `${v} ${k}`).slice(0, 3).join(" · ")} />
        <MetricTile label="Missing cells" value={fmt.pct(missingPct, 1)} hint={`${missing.length} column${missing.length === 1 ? "" : "s"} affected`} />
        <MetricTile label={imbalance ? "Class imbalance" : "Target"} value={imbalance ? `${imbalance.toFixed(1)} : 1` : d.target_column ?? "not set"}
          hint={imbalance ? "majority : minority rows" : d.task_type?.replace("_", " ")} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <TargetCard d={d} />
        <Card title="Missing values by column" description="Share of rows with no value. Imputation is applied during training.">
          {missing.length === 0 ? (
            <EmptyState compact icon={CheckCircle2} title="No missing values" />
          ) : (
            <Bars label="Missing values by column" items={missing.map((c) => ({ label: c.name, value: c.missing_pct / 100, note: `${fmt.int(c.n_missing)} rows` }))}
              format={(v) => fmt.pct(v, 1)} />
          )}
          <div className="mt-5 border-t border-line pt-4">
            <h3 className="mb-2.5 text-xs font-medium text-fg-3">Column types</h3>
            <SegmentedBar label="Column types" segments={typeCounts.map(([k, v]) => ({ label: k, value: v, color: typeColor(k) }))} />
          </div>
        </Card>
      </div>

      <Card title="Schema" description="Inferred column types. Identifier and constant columns are excluded from training automatically." flush>
        <Table label="Schema">
          <THead><TR><TH>Column</TH><TH>Type</TH><TH numeric>Missing</TH><TH numeric>Unique</TH><TH>Sample values</TH></TR></THead>
          <TBody>
            {columns.map((c) => (
              <TR key={c.name}>
                <TD className="whitespace-nowrap">
                  <span className="font-mono text-sm text-fg">{c.name}</span>
                  {c.name === d.target_column && <Badge tone="accent" className="ml-2" icon={<Crosshair size={11} />}>target</Badge>}
                </TD>
                <TD className="whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5 text-sm text-fg-2">
                    <i aria-hidden className="size-2 rounded-full" style={{ background: typeColor(c.inferred_type) }} />{c.inferred_type}
                  </span>
                  {c.numeric_stored_as_text && <span className="ml-1 text-xs text-fg-3">(stored as text)</span>}
                </TD>
                <TD numeric>
                  {c.missing_pct ? (
                    <span className="inline-flex items-center justify-end gap-2">
                      <span aria-hidden className="h-1.5 w-12 overflow-hidden rounded-full bg-surface-3">
                        <span className="block h-full bg-warning" style={{ width: `${Math.min(100, c.missing_pct * 5)}%` }} />
                      </span>
                      {c.missing_pct}%
                    </span>
                  ) : <span className="text-fg-3">–</span>}
                </TD>
                <TD numeric>{fmt.int(c.n_unique)}</TD>
                <TD className="max-w-80 truncate font-mono text-xs text-fg-3" title={c.sample_values.join(", ")}>{c.sample_values.join(", ")}</TD>
              </TR>
            ))}
          </TBody>
        </Table>
      </Card>

      <Preview id={d.id} />
    </div>
  );
}

function TargetCard({ d }: { d: Dataset }) {
  const toast = useToast();
  const setTarget = useSetTarget(d.id);
  const [target, setTargetValue] = useState(d.target_column ?? "");
  const [warnings, setWarnings] = useState<string[]>([]);
  const candidates = (d.columns ?? []).filter((c) => ["categorical", "boolean", "numeric"].includes(c.inferred_type));
  const dist = d.class_distribution;
  const minShare = dist?.length ? Math.min(...dist.map((c) => c.share)) : null;

  async function save() {
    const r = await setTarget.mutateAsync(target).catch(() => null);
    if (r) {
      setWarnings(r.warnings);
      toast({ title: `Target set to '${target}'`, tone: "success" });
    }
  }

  return (
    <Card title="Prediction target" description="The column the models learn to predict.">
      <div className="flex flex-wrap gap-2">
        <label htmlFor="target-col" className="sr-only">Target column</label>
        <select id="target-col" className="field min-w-0 flex-1" value={target} onChange={(e) => setTargetValue(e.target.value)}>
          <option value="">Choose a column…</option>
          {candidates.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.inferred_type}, {c.n_unique} values)</option>)}
        </select>
        <Button variant="primary" disabled={!target || target === d.target_column} loading={setTarget.isPending} onClick={save}>Set target</Button>
      </div>
      <InlineError error={setTarget.error} className="mt-2" />
      {warnings.map((w) => (
        <p key={w} className="mt-2 flex items-start gap-1.5 rounded-md bg-warning-subtle px-2.5 py-2 text-sm text-fg">
          <TriangleAlert size={14} aria-hidden className="mt-[3px] shrink-0 text-warning" />{w}
        </p>
      ))}
      {d.target_summary && <TargetSummaryView s={d.target_summary} />}
      {dist && (
        <div className="mt-5">
          <h3 className="mb-2 text-xs font-medium text-fg-3">Class balance</h3>
          <Bars label="Class balance" max={1} format={(v) => fmt.pct(v, 1)}
            items={dist.map((c) => ({ label: c.label, value: c.share, note: `${fmt.int(c.count)} rows`, tone: c.share === minShare && dist.length > 1 ? "highlight" : "default" }))} />
          {minShare != null && minShare < 0.25 && (
            <p className="mt-2 text-xs text-fg-3">The minority class (highlighted) is {fmt.pct(minShare, 1)} of rows. Accuracy will look good even if it is never predicted, so ModelDoctor ranks models by macro-F1.</p>
          )}
        </div>
      )}
      {!d.target_column && !dist && (
        <EmptyState compact icon={Crosshair} title="No target yet" description="ModelDoctor suggests binary or low-cardinality columns first." />
      )}
    </Card>
  );
}

function TargetSummaryView({ s }: { s: TargetSummary }) {
  return (
    <div className="mt-5">
      <h3 className="mb-2 text-xs font-medium text-fg-3">Target distribution (regression)</h3>
      <Histogram label="Target distribution" edges={s.histogram.edges} counts={s.histogram.counts} />
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-4">
        {[["Mean", fmt.sig(s.mean)], ["Median", fmt.sig(s.median)], ["Min / max", `${fmt.sig(s.min)} / ${fmt.sig(s.max)}`],
          ["Skewness", `${s.skew.toFixed(2)}${Math.abs(s.skew) >= 1 ? " (skewed)" : ""}`]].map(([k, v]) => (
          <div key={k}><dt className="text-xs text-fg-3">{k}</dt><dd className="font-mono">{v}</dd></div>
        ))}
      </dl>
    </div>
  );
}

function Preview({ id }: { id: string }) {
  const preview = usePreview(id);
  return (
    <Card title="Preview" description="First 8 rows as uploaded." flush>
      <QueryView query={preview} label="Loading preview" skeleton={<TableSkeleton rows={4} cols={6} />}>
        {(p) => (
          <Table label="Data preview" dense>
            <THead><TR>{p.columns.map((c) => <TH key={c} className="font-mono">{c}</TH>)}</TR></THead>
            <TBody>{p.rows.map((r, i) => (
              <TR key={i}>{r.map((v, j) => <TD key={j} className="whitespace-nowrap font-mono text-xs">{v === null ? <span className="text-fg-3">null</span> : String(v)}</TD>)}</TR>
            ))}</TBody>
          </Table>
        )}
      </QueryView>
    </Card>
  );
}
