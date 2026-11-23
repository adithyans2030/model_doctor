import { RotateCw } from "lucide-react";
import { Bars, Histogram } from "@/components/charts";
import { FindingsList } from "@/components/domain/FindingCard";
import { ScoreRing } from "@/components/domain/HealthScore";
import { QueryView } from "@/components/domain/QueryView";
import { SeverityBadge } from "@/components/domain/severity";
import { Button, Card, CardSkeleton, InlineError, MetricTile, MetricTileSkeleton } from "@/components/ui";
import { fmt } from "@/lib/api";
import { useDataReport, useRefreshDataReport } from "@/lib/queries";
import type { DataReport } from "@/lib/types";

const LEAK = 0.95;

export function DataDoctorTab({ id }: { id: string }) {
  const report = useDataReport(id);
  return (
    <QueryView query={report} label="Running the Data Doctor"
      skeleton={
        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{[0, 1, 2, 3].map((i) => <MetricTileSkeleton key={i} />)}</div>
          <div className="grid gap-5 lg:grid-cols-2"><CardSkeleton lines={8} /><CardSkeleton lines={6} /></div>
        </div>
      }>
      {(r) => <DataDoctorView id={id} r={r} />}
    </QueryView>
  );
}

function DataDoctorView({ id, r }: { id: string; r: DataReport }) {
  const refresh = useRefreshDataReport(id);
  const s = r.summary;
  const excluded = Object.entries(s.excluded_columns ?? {});
  const relevance = [...r.feature_target].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const numeric = r.column_stats.filter((c) => c.histogram);
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="flex items-center gap-4 rounded-lg border border-line bg-surface p-4 sm:col-span-2 lg:col-span-1">
          <ScoreRing value={r.score.value} size={76} label="Data quality score" />
          <div className="min-w-0">
            <div className="text-xs font-medium text-fg-3">Data quality</div>
            <p className="mt-1 text-xs text-fg-3">100 minus {r.score.penalties.length} listed penalties</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {(["critical", "warning"] as const).map((sev) => s.severity_counts?.[sev] ? <SeverityBadge key={sev} severity={sev} count={s.severity_counts[sev]} /> : null)}
            </div>
          </div>
        </div>
        <MetricTile label="Missing cells" value={`${s.missing_cells_pct}%`} hint={`${fmt.int(s.duplicate_rows)} duplicate rows (${s.duplicate_pct}%)`} />
        <MetricTile label="Usable features" value={s.n_features_usable ?? "–"} hint={excluded.length ? `excluded: ${excluded.map(([k]) => k).join(", ")}` : "none excluded"} />
        <MetricTile label="Imbalance ratio" value={s.imbalance_ratio ? `${s.imbalance_ratio.toFixed(1)} : 1` : "–"} hint="majority : minority class" />
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card title="Findings" description="Ranked by severity. Each has the measured evidence and a recommended fix."
          actions={<Button size="sm" icon={<RotateCw size={13} />} loading={refresh.isPending} onClick={() => refresh.mutate()}>Re-run</Button>}>
          <InlineError error={refresh.error} className="mb-3" />
          <FindingsList findings={r.findings} empty="No data problems found." />
        </Card>
        <div className="space-y-5">
          {relevance.length > 0 && (
            <Card title="Feature relevance" description={`Single-feature probe (${relevance[0].metric}). ≈0.5 means no signal; ≥${LEAK} alone predicts the target, which usually means leakage.`}>
              <Bars label="Single-feature relevance to the target" max={1} markers={[{ value: 0.5, label: "no signal" }, { value: LEAK, label: "leak" }]}
                items={relevance.map((f) => ({
                  label: f.column, value: f.score ?? 0, note: `mutual information ${f.mutual_info}`,
                  tone: (f.score ?? 0) >= LEAK ? "danger" : "default",
                }))} />
            </Card>
          )}
          {r.score.penalties.length > 0 && (
            <Card title="Score breakdown" description="How the quality score was computed.">
              <ul className="space-y-1.5 text-sm">
                {r.score.penalties.map((p, i) => (
                  <li key={i} className="flex items-start justify-between gap-3">
                    <span className="min-w-0 text-fg-2">{p.title}</span>
                    <span className="shrink-0 font-mono text-danger">−{p.points}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>

      {numeric.length > 0 && (
        <Card title="Numeric distributions" description="Clipped to the 1st–99th percentile so a few outliers don't flatten the chart; outlier counts come from the Data Doctor.">
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
            {numeric.map((c) => (
              <div key={c.name}>
                <div className="mb-1.5 flex items-baseline justify-between gap-2 text-sm">
                  <span className="truncate font-mono text-fg">{c.name}</span>
                  <span className="shrink-0 text-xs text-fg-3">median {c.median?.toPrecision(4) ?? "–"}{c.n_outliers ? ` · ${c.n_outliers} outliers` : ""}</span>
                </div>
                {c.histogram && <Histogram label={`Distribution of ${c.name}`} edges={c.histogram.edges} counts={c.histogram.counts} />}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
