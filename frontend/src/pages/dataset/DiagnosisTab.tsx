import { ArrowRight, FlaskConical, Stethoscope } from "lucide-react";
import { Link } from "react-router-dom";
import { Assistant } from "@/components/Assistant";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { FindingsList } from "@/components/domain/FindingCard";
import { HealthPanel } from "@/components/domain/HealthScore";
import { QueryView } from "@/components/domain/QueryView";
import { SEVERITY, SeverityBadge, findingAnchor } from "@/components/domain/severity";
import { Badge, Card, CardSkeleton, EmptyState, buttonClasses } from "@/components/ui";
import { useDiagnosis } from "@/lib/queries";
import type { Diagnosis } from "@/lib/types";

export function DiagnosisTab({ id }: { id: string }) {
  const diag = useDiagnosis(id);
  return (
    <QueryView query={diag} label="Diagnosing"
      skeleton={<div className="space-y-5"><CardSkeleton lines={5} /><div className="grid gap-5 lg:grid-cols-[7fr_5fr]"><CardSkeleton lines={10} /><CardSkeleton lines={6} /></div></div>}>
      {(d) => <DiagnosisReport id={id} d={d} />}
    </QueryView>
  );
}

function DiagnosisReport({ id, d }: { id: string; d: Diagnosis }) {
  if (!d.model && d.findings.length === 0) {
    return <EmptyState icon={Stethoscope} title="Nothing to diagnose yet" description="Train baseline models on the Models tab; the diagnosis combines data and model findings." />;
  }
  return (
    <div className="space-y-5">
      <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <HealthPanel d={d} />
        <Card title="Summary" description="Generated from the findings below, not by the AI.">
          <ul className="space-y-2 text-sm">
            {d.narrative.map((line, i) => {
              const critical = line.startsWith("Critical:");
              return (
                <li key={i} className="flex gap-2.5">
                  <span aria-hidden className={critical ? "mt-[7px] size-1.5 shrink-0 rounded-full bg-danger" : "mt-[7px] size-1.5 shrink-0 rounded-full bg-fg-3"} />
                  <span className={critical ? "text-fg" : "text-fg-2"}>{line}</span>
                </li>
              );
            })}
          </ul>
          {d.model && (
            <Link to={`/app/models/${d.model.id}`} className={buttonClasses({ size: "sm", className: "mt-4" })}>
              Open {d.model.name} <ArrowRight size={14} aria-hidden />
            </Link>
          )}
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card title="Issues, ranked by severity" description="Each issue shows the measured evidence, an explanation and the recommended fix.">
          <FindingsList findings={d.findings} anchors empty="No problems found." />
        </Card>
        <div className="space-y-5">
          <Card title="Fix plan" description="Recommended interventions, most severe first. The optimiser tests each one with cross-validation.">
            {d.recommendations.length === 0 ? <EmptyState compact title="Nothing to fix" /> : (
              <ol className="space-y-3">
                {d.recommendations.map((r, i) => (
                  <li key={r.intervention} className="flex gap-3">
                    <span className="grid size-6 shrink-0 place-items-center rounded-full border border-line bg-surface-2 font-mono text-xs text-fg-2">{i + 1}</span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium">{r.label}</span>
                        <SeverityBadge severity={r.severity} />
                      </div>
                      <p className="mt-1 text-xs text-fg-3">
                        Because:{" "}
                        {r.because.map((b, j) => (
                          <span key={`${b.check}-${j}`}>
                            {j > 0 && "; "}
                            <a href={`#${findingAnchor(b.check)}`} className={`underline decoration-current/40 underline-offset-2 ${SEVERITY[r.severity].text}`}>{b.title}</a>
                          </span>
                        ))}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            )}
            {d.model && d.recommendations.length > 0 && (
              <Link to={`/app/models/${d.model.id}?tab=experiments`} className={buttonClasses({ variant: "primary", size: "sm", className: "mt-5" })}>
                <FlaskConical size={14} aria-hidden /> Test these fixes
              </Link>
            )}
          </Card>
          {d.latest_experiment?.best && (
            <Card title="Latest experiment">
              <p className="text-sm text-fg-2">{d.latest_experiment.best.description}</p>
              <p className="mt-2 flex flex-wrap gap-1.5"><Badge tone="success">kept by cross-validation</Badge></p>
            </Card>
          )}
          <ErrorBoundary label="the assistant">
            <Assistant datasetId={id} modelId={d.model?.id} compact linkCitations
              suggestions={["What should I fix first?", "Why is my recall so low?", "Explain the leakage finding"]} />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}
