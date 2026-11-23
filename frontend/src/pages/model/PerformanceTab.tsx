import { Bars, ConfusionMatrix, Histogram, ScatterChart, ThresholdChart } from "@/components/charts";
import { FindingsList } from "@/components/domain/FindingCard";
import { Card, MetricTile, TBody, TD, TH, THead, TR, Table } from "@/components/ui";
import { fmt } from "@/lib/api";
import type { ClassificationModel, ModelDetail, RegressionModel } from "@/lib/types";

export function PerformanceTab({ m }: { m: ModelDetail }) {
  return m.task === "regression" ? <RegressionPerformance m={m} /> : <ClassificationPerformance m={m} />;
}

function Importance({ m }: { m: ModelDetail }) {
  if (!m.importances?.length) return null;
  return (
    <Card title="Model-native feature importance" description="As reported by the fitted model. See Explainability for SHAP and permutation importance.">
      <Bars label="Model-native feature importance" items={m.importances.map((i) => ({ label: i.feature, value: i.importance }))} format={(v) => fmt.pct(v, 1)} />
    </Card>
  );
}

function ClassificationPerformance({ m }: { m: ClassificationModel }) {
  const t = m.metrics.test;
  const cv = m.metrics.cv;
  const pos = t.positive_class ?? t.worst_class;
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile emphasis label="Macro-F1 (primary)" value={fmt.num(t.f1_macro)}
          hint={cv ? `CV ${fmt.num(cv.val.f1_macro.mean)} ± ${fmt.num(cv.val.f1_macro.std)}` : "no cross-validation"} />
        <MetricTile label="Accuracy / balanced" value={`${fmt.pct(t.accuracy)} / ${fmt.pct(t.balanced_accuracy)}`}
          hint={`majority-class baseline: ${fmt.pct(m.metrics.dummy?.balanced_accuracy)} balanced`} />
        <MetricTile label={`Recall · class ${pos.label}`} value={fmt.pct(pos.recall)} hint={`precision ${fmt.pct(pos.precision)} · weakest-class F1 ${fmt.num(t.worst_class.f1)}`} />
        <MetricTile label="ROC AUC / PR AUC" value={`${fmt.num(t.roc_auc)} / ${fmt.num(t.pr_auc)}`} hint={`calibration error (ECE) ${fmt.num(t.ece)}`} />
      </div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card title="Diagnosis" description="Model-level findings, ranked by severity.">
          <FindingsList findings={m.findings} empty="No model-level problems found." />
        </Card>
        <div className="space-y-5">
          <Card title="Confusion matrix" description={`Test set, ${fmt.int(t.n)} rows. Shading is the share of each actual class.`}>
            <ConfusionMatrix labels={t.confusion_matrix.labels} matrix={t.confusion_matrix.matrix} />
          </Card>
          <Card title="Per-class performance" flush>
            <Table label="Per-class performance">
              <THead><TR><TH>Class</TH><TH numeric>Precision</TH><TH numeric>Recall</TH><TH numeric>F1</TH><TH numeric>Support</TH></TR></THead>
              <TBody>{t.per_class.map((c) => (
                <TR key={c.label}>
                  <TD className="font-mono">{c.label}</TD><TD numeric>{fmt.num(c.precision)}</TD><TD numeric>{fmt.num(c.recall)}</TD>
                  <TD numeric>{fmt.num(c.f1)}</TD><TD numeric>{fmt.int(c.support)}</TD>
                </TR>
              ))}</TBody>
            </Table>
          </Card>
        </div>
      </div>
      {t.threshold_analysis && (
        <Card title={`Decision threshold · positive class '${m.metrics.positive_label}'`}
          description={`F1 at 0.5: ${fmt.num(t.threshold_analysis.f1_at_0_5)} · best ${fmt.num(t.threshold_analysis.best_f1)} at ${t.threshold_analysis.best_threshold}${t.threshold != null ? ` · this model uses ${t.threshold}` : ""}`}>
          <ThresholdChart curve={t.threshold_analysis.curve} best={t.threshold_analysis.best_threshold} current={t.threshold ?? 0.5} />
        </Card>
      )}
      <Importance m={m} />
    </div>
  );
}

function RegressionPerformance({ m }: { m: RegressionModel }) {
  const t = m.metrics.test;
  const cv = m.metrics.cv;
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile emphasis label="R² (primary)" value={fmt.num(t.r2)} hint={cv?.val.r2 ? `CV ${fmt.num(cv.val.r2.mean)} ± ${fmt.num(cv.val.r2.std)}` : "no cross-validation"} />
        <MetricTile label="RMSE" value={fmt.sig(t.rmse)} hint={`mean-only baseline ${fmt.sig(m.metrics.dummy?.rmse)}`} />
        <MetricTile label="MAE / median AE" value={`${fmt.sig(t.mae)} / ${fmt.sig(t.median_ae)}`} hint={t.mape != null ? `MAPE ${fmt.pct(t.mape, 1)}` : "MAPE n/a (zeros in target)"} />
        <MetricTile label="Error profile" value={fmt.pct(t.top5pct_error_share)} hint="share of total error from the worst 5% of rows" />
      </div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card title="Diagnosis" description="Model-level findings, ranked by severity.">
          <FindingsList findings={m.findings} empty="No model-level problems found." />
        </Card>
        <div className="space-y-5">
          <Card title="Predicted vs actual" description="Sample of test rows.">
            <ScatterChart points={t.points} />
          </Card>
          <Card title="Residuals (actual − predicted)"
            description={`Mean ${fmt.sig(t.residual_mean)} (bias) · std ${fmt.sig(t.residual_std)} · error-vs-prediction correlation ${fmt.num(t.error_vs_prediction_corr, 2)}`}>
            <Histogram label="Residual distribution" edges={t.residual_histogram.edges} counts={t.residual_histogram.counts} />
          </Card>
        </div>
      </div>
      <Importance m={m} />
    </div>
  );
}
