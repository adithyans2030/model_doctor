import { CheckCircle2, ChevronRight, FlaskConical, ListFilter } from "lucide-react";
import { useId, useState } from "react";
import { Badge, EmptyState } from "@/components/ui";
import { cx } from "@/lib/cx";
import type { EvidenceValue, Finding, Severity } from "@/lib/types";
import { SEVERITY, SeverityBadge, findingAnchor, sortBySeverity } from "./severity";

function formatEvidence(v: EvidenceValue): string {
  if (v === null) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : String(+v.toFixed(4));
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.length > 6 ? `${v.slice(0, 6).map(formatEvidence).join(", ")}, … (${v.length})` : v.map(formatEvidence).join(", ");
  if (typeof v === "object") return Object.entries(v).map(([k, x]) => `${k}: ${formatEvidence(x)}`).join(" · ");
  return v;
}

const humanize = (s: string) => s.replaceAll("_", " ");

/**
 * One diagnosed issue: what was measured (evidence), why it matters (cause and impact),
 * and what to do (recommendation + the fixes the optimiser can test).
 */
export function FindingCard({ finding: f, defaultOpen, anchor }: { finding: Finding; defaultOpen?: boolean; anchor?: boolean }) {
  const [open, setOpen] = useState(defaultOpen ?? f.severity === "critical");
  const bodyId = useId();
  const evidence = Object.entries(f.evidence);
  return (
    <article
      id={anchor ? findingAnchor(f.check) : undefined}
      className="relative scroll-mt-24 overflow-hidden rounded-lg border border-line bg-surface"
    >
      <span aria-hidden className={cx("absolute inset-y-0 left-0 w-[3px]", SEVERITY[f.severity].rail)} />
      <h3 className="m-0">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((o) => !o)}
          className="flex w-full cursor-pointer items-start gap-3 py-3 pl-4 pr-3 text-left hover:bg-surface-2/50"
        >
          <ChevronRight size={16} aria-hidden className={cx("mt-0.5 shrink-0 text-fg-3 transition-transform", open && "rotate-90")} />
          <span className="min-w-0 flex-1">
            <span className={cx("block text-sm font-medium leading-5", f.resolved ? "text-fg-2" : "text-fg")}>{f.title}</span>
            <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
              <SeverityBadge severity={f.severity} />
              <Badge mono>{f.check}</Badge>
              {f.resolved && <Badge tone="success" icon={<CheckCircle2 size={11} />}>Resolved</Badge>}
            </span>
          </span>
        </button>
      </h3>
      <div id={bodyId} hidden={!open} className="space-y-3 border-t border-line py-3 pl-11 pr-4 text-sm">
        {evidence.length > 0 && (
          <div>
            <h4 className="mb-1.5 text-xs font-medium text-fg-3">Evidence</h4>
            <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-4 gap-y-1 rounded-md bg-surface-2 px-3 py-2 font-mono text-xs">
              {evidence.slice(0, 8).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-fg-3">{k}</dt>
                  <dd className="min-w-0 break-words text-fg">{formatEvidence(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
        {(f.cause || f.impact) && (
          <div>
            <h4 className="mb-1 text-xs font-medium text-fg-3">Explanation</h4>
            {f.cause && <p className="text-fg-2"><span className="text-fg">Likely cause.</span> {f.cause}</p>}
            {f.impact && <p className="mt-1 text-fg-2"><span className="text-fg">Impact.</span> {f.impact}</p>}
          </div>
        )}
        {(f.recommendation || f.interventions.length > 0) && (
          <div className="rounded-md border border-line bg-surface-2/60 px-3 py-2.5">
            <h4 className="mb-1 text-xs font-medium text-fg-3">Recommended fix</h4>
            {f.recommendation && <p className="text-fg">{f.recommendation}</p>}
            {f.interventions.length > 0 && (
              <p className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-fg-3">
                <FlaskConical size={13} aria-hidden /> Testable by the optimiser:
                {f.interventions.map((i) => <Badge key={i} tone="accent">{humanize(i)}</Badge>)}
              </p>
            )}
          </div>
        )}
      </div>
    </article>
  );
}

const FILTERS: (Severity | "all")[] = ["all", "critical", "warning", "info"];

/** Findings ranked by severity, with a severity filter. */
export function FindingsList({ findings, empty = "No findings.", anchors }: { findings: Finding[]; empty?: string; anchors?: boolean }) {
  const [filter, setFilter] = useState<Severity | "all">("all");
  if (!findings.length) return <EmptyState compact icon={CheckCircle2} title={empty} />;
  const ranked = sortBySeverity(findings);
  const shown = filter === "all" ? ranked : ranked.filter((f) => f.severity === filter);
  const count = (s: Severity) => findings.filter((f) => f.severity === s).length;
  const seen = new Set<string>();
  return (
    <div className="space-y-3">
      <div role="group" aria-label="Filter findings by severity" className="flex flex-wrap items-center gap-1.5">
        <ListFilter size={14} aria-hidden className="mr-0.5 text-fg-3" />
        {FILTERS.map((s) => {
          const n = s === "all" ? findings.length : count(s);
          if (s !== "all" && n === 0) return null;
          return (
            <button key={s} type="button" aria-pressed={filter === s} onClick={() => setFilter(s)}
              className={cx(
                "inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-md border px-2.5 text-xs transition-colors",
                filter === s ? "border-line-strong bg-surface-3 text-fg" : "border-line text-fg-2 hover:bg-surface-2",
              )}>
              {s === "all" ? "All" : SEVERITY[s].label}
              <span className="font-mono text-fg-3">{n}</span>
            </button>
          );
        })}
      </div>
      <div className="space-y-2">
        {shown.map((f, i) => {
          const first = !seen.has(f.check);
          seen.add(f.check);
          return <FindingCard key={`${f.check}-${i}`} finding={f} anchor={anchors && first} />;
        })}
      </div>
    </div>
  );
}
