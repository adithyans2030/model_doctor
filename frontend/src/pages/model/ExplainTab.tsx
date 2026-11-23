import { MousePointerClick, Users } from "lucide-react";
import { useState } from "react";
import { Bars, DivergingBars } from "@/components/charts";
import { FindingsList } from "@/components/domain/FindingCard";
import { QueryView } from "@/components/domain/QueryView";
import { Badge, Card, CardSkeleton, EmptyState, InlineError, LoadingRegion, SkeletonText, TBody, TD, TH, THead, TR, Table } from "@/components/ui";
import { fmt } from "@/lib/api";
import { useExplainRow, useExplanations } from "@/lib/queries";
import type { Explanations, ModelDetail } from "@/lib/types";

export function ExplainTab({ m }: { m: ModelDetail }) {
  const ex = useExplanations(m.id);
  return (
    <QueryView query={ex} label="Computing SHAP, permutation importance and error analysis"
      skeleton={<div className="grid gap-5 lg:grid-cols-2"><CardSkeleton lines={9} /><CardSkeleton lines={9} /></div>}>
      {(data) => <ExplainView m={m} ex={data} />}
    </QueryView>
  );
}

function ExplainView({ m, ex }: { m: ModelDetail; ex: Explanations }) {
  const regression = m.task === "regression";
  const g = ex.global_importance;
  const ea = ex.error_analysis;
  const fair = ex.fairness;
  return (
    <div className="space-y-5">
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="What drives the model" description={g.shap ? `${g.shap.method}: mean |SHAP| ${regression ? "on the prediction" : `for class '${g.shap.explained_class}'`}` : "SHAP isn't available for this model type."}>
          {g.shap && <Bars label="Mean absolute SHAP value by feature" items={g.shap.values.map((v) => ({ label: v.feature, value: v.mean_abs_shap }))} />}
          <h3 className="mt-6 mb-1 text-xs font-medium text-fg-3">Permutation importance</h3>
          <p className="mb-2 text-xs text-fg-3">Drop in {regression ? "R²" : "macro-F1"} when a feature is shuffled (baseline {g.permutation.baseline}). Model-agnostic.</p>
          <Bars label="Permutation importance by feature" items={g.permutation.values.map((v) => ({ label: v.feature, value: Math.max(0, v.importance), note: `± ${v.std}` }))} />
        </Card>
        <Card title="Where the errors are"
          description={regression
            ? `MAE ${fmt.sig(ea.summary.mae)} on ${ea.summary.n_test} test rows · "large error" means above ${fmt.sig(ea.summary.large_error_threshold)}`
            : `${ea.summary.n_errors} errors in ${ea.summary.n_test} test rows (${fmt.pct(ea.summary.error_rate, 1)})${ea.summary.false_negatives != null ? ` · ${ea.summary.false_negatives} false negatives · ${ea.summary.false_positives} false positives` : ""}`}>
          <ul className="space-y-1.5">
            {ea.statements.map((s) => <li key={s} className="rounded-md bg-surface-2 px-3 py-2 text-sm text-fg-2">{s}</li>)}
          </ul>
          {ea.slices.length > 0 && (
            <div className="mt-4 -mx-4 border-t border-line">
              <Table label="Error slices" dense>
                <THead><TR><TH>Slice</TH><TH numeric>Rows</TH><TH numeric>{regression ? "MAE" : "Error rate"}</TH><TH numeric>Lift</TH></TR></THead>
                <TBody>{ea.slices.slice(0, 8).map((s) => (
                  <TR key={s.feature + s.value}>
                    <TD className="font-mono text-xs">{s.feature} = {s.value}</TD><TD numeric>{s.n}</TD>
                    <TD numeric>{regression ? fmt.sig(s.mae) : fmt.pct(s.error_rate)}</TD><TD numeric>{s.lift ?? "–"}×</TD>
                  </TR>
                ))}</TBody>
              </Table>
            </div>
          )}
        </Card>
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <LocalExplain m={m} ex={ex} />
        <Card title="Group performance" description={fair.features.length ? `Largest ${regression ? "relative MAE" : "balanced-accuracy"} gap: ${fmt.num(fair.max_gap)} across '${fair.worst_feature}'.` : undefined}>
          {fair.features.length === 0 ? <EmptyState compact icon={Users} title="No categorical groups large enough to compare" /> : (
            <div className="space-y-5">
              {fair.features.slice(0, 3).map((f) => (
                <div key={f.feature}>
                  <div className="mb-1.5 flex items-center gap-2 text-sm"><span className="font-mono">{f.feature}</span><Badge>gap {f.gap}</Badge></div>
                  {regression ? (
                    <Bars label={`MAE by ${f.feature}`} format={(v) => fmt.sig(v)}
                      items={f.groups.map((r) => ({ label: r.value, value: r.mae ?? 0, note: `${r.n} rows · R² ${fmt.num(r.r2, 2)}` }))} />
                  ) : (
                    <Bars label={`Balanced accuracy by ${f.feature}`} max={1}
                      items={f.groups.filter((r) => r.balanced_accuracy != null).map((r) => ({ label: r.value, value: r.balanced_accuracy ?? 0, note: `${r.n} rows · positive recall ${fmt.pct(r.positive_recall)}` }))} />
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
      <Card title="Prediction findings"><FindingsList findings={ex.findings} empty="No prediction-level problems found." /></Card>
    </div>
  );
}

function LocalExplain({ m, ex }: { m: ModelDetail; ex: Explanations }) {
  const regression = m.task === "regression";
  const explain = useExplainRow(m.id);
  const [row, setRow] = useState("");
  const [method, setMethod] = useState<"auto" | "lime">("auto");
  const run = (r: string, how: "auto" | "lime") => { if (r) explain.mutate({ row_id: +r, method: how }); };
  const local = explain.data;
  return (
    <Card title="Explain a prediction" description="Per-feature contributions for one test row.">
      <div className="flex flex-wrap gap-2">
        <label htmlFor="explain-row" className="sr-only">Test row</label>
        <select id="explain-row" className="field min-w-0 flex-1" value={row} onChange={(e) => { setRow(e.target.value); run(e.target.value, method); }}>
          <option value="">Pick a test row…</option>
          {ex.samples.map((s) => (
            <option key={s.row_id} value={s.row_id}>
              {regression
                ? `row ${s.row_id}: actual ${fmt.sig(Number(s.true))}, predicted ${fmt.sig(Number(s.predicted))} (error ${fmt.sig(s.abs_error)})`
                : `row ${s.row_id}: true ${s.true}, predicted ${s.predicted} (${fmt.pct(s.confidence)})${s.correct ? "" : " ✗"}`}
            </option>
          ))}
        </select>
        <select className="field" value={method} aria-label="Explanation method"
          onChange={(e) => { const v = e.target.value as "auto" | "lime"; setMethod(v); run(row, v); }}>
          <option value="auto">SHAP (or occlusion)</option>
          <option value="lime">LIME</option>
        </select>
      </div>
      <InlineError error={explain.error} className="mt-3" />
      {explain.isPending && <LoadingRegion label="Explaining prediction" className="mt-4"><SkeletonText lines={6} /></LoadingRegion>}
      {!local && !explain.isPending && <EmptyState compact icon={MousePointerClick} title="Pick a row to explain" description="Misclassified rows are marked ✗." />}
      {local && !explain.isPending && (
        <div className="mt-4">
          <p className="text-sm font-medium">{local.explanation}</p>
          <p className="mt-1 mb-3 text-xs text-fg-3">
            Actual: {local.actual == null ? "unknown" : regression ? fmt.sig(Number(local.actual)) : local.actual} · method: {local.method} ({local.units})
            {local.fidelity != null && ` · local fit (fidelity) ${fmt.num(local.fidelity, 2)}`}
          </p>
          <DivergingBars label={`Feature contributions for row ${local.row_id}`}
            posLabel={regression ? "pushes the prediction up" : `towards '${local.prediction}'`} negLabel={regression ? "pushes it down" : "away from it"}
            items={local.contributions.slice(0, 10).map((c) => ({ label: c.feature, value: c.contribution, detail: c.value }))} />
        </div>
      )}
    </Card>
  );
}
