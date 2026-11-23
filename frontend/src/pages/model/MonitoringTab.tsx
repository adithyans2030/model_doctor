import { Activity, Upload } from "lucide-react";
import { useState } from "react";
import { Bars } from "@/components/charts";
import { FindingsList } from "@/components/domain/FindingCard";
import { FileDrop } from "@/components/domain/FileDrop";
import { QueryView } from "@/components/domain/QueryView";
import { TrendChart } from "@/components/TrendChart";
import {
  Badge, Button, Card, CardSkeleton, EmptyState, InlineError, MetricTile, MetricTileSkeleton, TBody, TD, TH, THead, TR, Table, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { useDrift, useMonitorBatch } from "@/lib/queries";
import type { DriftCheck, ModelDetail } from "@/lib/types";

export function MonitoringTab({ m }: { m: ModelDetail }) {
  const history = useDrift(m.id);
  const monitor = useMonitorBatch(m.id);
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);

  async function analyse() {
    if (!file) return;
    const r = await monitor.mutateAsync(file).catch(() => null);
    if (r) {
      setFile(null);
      toast({ title: `Drift score ${r.score} / 100`, description: `${r.drifted_features} of ${r.n_features} features drifted in ${r.batch.filename}.`, tone: r.drifted_features ? "info" : "success" });
    }
  }

  return (
    <div className="space-y-5">
      <Card title="Check a production batch for drift"
        description="Upload a CSV with the model's feature columns. Include the target column too to measure performance on the batch (concept drift). Only the report is stored, not the rows.">
        <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <FileDrop accept=".csv,text/csv" file={file} onFile={setFile} label="Choose a batch CSV" disabled={monitor.isPending} />
          <Button variant="primary" icon={<Upload size={14} />} disabled={!file} loading={monitor.isPending} onClick={analyse}>
            {monitor.isPending ? "Analysing…" : "Analyse drift"}
          </Button>
        </div>
        <InlineError error={monitor.error} className="mt-3" />
      </Card>
      <QueryView query={history} label="Loading drift history"
        skeleton={<div className="space-y-5"><div className="grid gap-3 sm:grid-cols-4">{[0, 1, 2, 3].map((i) => <MetricTileSkeleton key={i} />)}</div><CardSkeleton lines={6} /></div>}
        isEmpty={(h) => h.length === 0}
        empty={<Card><EmptyState icon={Activity} title="No batches checked yet" description="Each upload is compared with the training data using PSI plus KS or chi-square tests, per feature." /></Card>}>
        {(h) => <DriftView history={h} />}
      </QueryView>
    </div>
  );
}

function DriftView({ history }: { history: DriftCheck[] }) {
  const latest = history[0].report;
  const t = latest.thresholds;
  const chronological = [...history].reverse();
  const sevTone = (s: string) => (s === "major" ? "danger" : s === "moderate" ? "warning" : "default") as "danger" | "default";
  return (
    <div className="space-y-5">
      <div>
        <h2 className="mb-3 flex flex-wrap items-center gap-2 text-sm font-semibold">
          Latest batch <Badge mono>{latest.batch.filename}</Badge><span className="text-xs font-normal text-fg-3">{fmt.int(latest.batch.n_rows)} rows · {fmt.date(history[0].created_at)}</span>
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricTile emphasis label="Drift score" value={`${latest.score} / 100`} hint="100 means no drift"
            delta={history[1] ? { value: latest.score - history[1].report.score, format: (v) => v.toFixed(0), label: "vs previous batch" } : undefined} />
          <MetricTile label="Drifted features" value={`${latest.drifted_features} / ${latest.n_features}`} hint={`PSI ≥ ${t.psi_moderate}`} />
          <MetricTile label="Prediction-mix shift" value={fmt.num(latest.prediction_drift.psi, 3)} hint="PSI of predictions vs the test set" />
          {latest.performance ? (
            <MetricTile label={`Batch ${latest.performance.metric === "r2" ? "R²" : "macro-F1"}`} value={fmt.num(latest.performance.batch_primary)}
              delta={{ value: latest.performance.delta_primary, label: "vs test set" }} hint={`test ${fmt.num(latest.performance.test_primary)} · ${fmt.int(latest.performance.n_labelled)} labelled rows`} />
          ) : (
            <MetricTile label="Batch performance" value="–" hint="Include the target column to measure it" />
          )}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Drift score over time" description="One point per uploaded batch.">
          {chronological.length < 2 ? (
            <EmptyState compact title="One batch so far" description="Upload another batch to see the trend." />
          ) : (
            <TrendChart label="Drift score by batch" yLabel="Drift score"
              points={chronological.map((c) => ({ x: new Date(c.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }) + ` · ${c.report.batch.filename.replace(/\.csv$/, "")}`, y: c.report.score, detail: `${c.report.drifted_features} drifted features` }))} />
          )}
        </Card>
        <Card title="Population stability by feature" description={`PSI on the latest batch. ${t.psi_moderate} = moderate, ${t.psi_major} = major drift.`}>
          <Bars label="Population stability index by feature" format={(v) => v.toFixed(3)}
            markers={[{ value: t.psi_moderate, label: "moderate" }, { value: t.psi_major, label: "major" }]}
            items={[...latest.features].sort((a, b) => b.psi - a.psi).map((f) => ({
              label: f.feature, value: f.psi, tone: sevTone(f.severity),
              note: `${f.severity} · ${f.test} p=${f.p_value == null ? "–" : f.p_value.toExponential(2)}`,
            }))} />
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Drift findings"><FindingsList findings={latest.findings} empty="No drift detected in the latest batch." /></Card>
        <Card title="Batch history" flush>
          <Table label="Drift checks">
            <THead><TR><TH>When</TH><TH>Batch</TH><TH numeric>Rows</TH><TH numeric>Drifted</TH><TH numeric>Score</TH></TR></THead>
            <TBody>{history.map((c) => (
              <TR key={c.id}>
                <TD className="whitespace-nowrap text-xs text-fg-3">{fmt.date(c.created_at)}</TD>
                <TD className="font-mono text-xs">{c.report.batch.filename}</TD>
                <TD numeric>{fmt.int(c.report.batch.n_rows)}</TD>
                <TD numeric>{c.report.drifted_features}</TD>
                <TD numeric className="text-fg">{c.report.score}</TD>
              </TR>
            ))}</TBody>
          </Table>
        </Card>
      </div>
    </div>
  );
}
