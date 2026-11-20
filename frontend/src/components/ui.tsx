import { useState, type ReactNode } from "react";
import type { Diagnosis, Finding, Severity } from "@/lib/types";

const SEV: Record<Severity, { icon: string; label: string; dot: string; border: string }> = {
  critical: { icon: "▲", label: "Critical", dot: "bg-critical", border: "border-l-critical" },
  warning: { icon: "◆", label: "Warning", dot: "bg-warning", border: "border-l-warning" },
  info: { icon: "●", label: "Info", dot: "bg-fg-3", border: "border-l-line" },
};

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  // Status colour is never alone: an icon shape and a text label carry the meaning too.
  const s = SEV[severity];
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-fg">
      <span className={cx("size-2.5 rounded-full", s.dot)} aria-hidden />
      <span aria-hidden>{s.icon}</span>
      {s.label}
    </span>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: [T, string][]; value: T; onChange: (t: T) => void }) {
  return (
    <div className="my-4 flex gap-0.5 overflow-x-auto border-b border-line" role="tablist">
      {tabs.map(([key, label]) => (
        <button key={key} role="tab" aria-selected={value === key} onClick={() => onChange(key)}
          className={cx(
            "-mb-px whitespace-nowrap border-b-2 px-3.5 py-2 transition-colors",
            value === key ? "border-accent font-semibold text-fg" : "border-transparent text-fg-2 hover:text-fg",
          )}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Spinner() {
  return <span className="inline-block size-3.5 animate-spin rounded-full border-2 border-line border-t-accent align-[-2px]" />;
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return <div className="p-6 text-center text-fg-3"><Spinner /> {label}…</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="p-6 text-center text-fg-3">{children}</div>;
}

export function ErrorBox({ error }: { error: string | null | undefined }) {
  if (!error) return null;
  return <div className="notice mt-2 text-critical" role="alert">{error}</div>;
}

function evidenceRows(ev: Record<string, unknown>) {
  return Object.entries(ev).filter(([, v]) => v === null || ["string", "number", "boolean"].includes(typeof v));
}

export function FindingCard({ f }: { f: Finding }) {
  const rows = evidenceRows(f.evidence);
  return (
    <div className={cx("rounded-r-lg border-l-[3px] bg-surface px-3 py-2.5", SEV[f.severity].border, f.resolved && "opacity-70")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SeverityBadge severity={f.severity} />
        <span className="badge font-mono">{f.check}</span>
      </div>
      <div className="mt-1.5 mb-1 font-semibold">{f.title}</div>
      {f.cause && <p className="text-[13px] text-fg-2"><b>Likely cause:</b> {f.cause}</p>}
      {f.impact && <p className="text-[13px] text-fg-2"><b>Impact:</b> {f.impact}</p>}
      {f.recommendation && <p className="text-[13px]"><b>Recommendation:</b> {f.recommendation}</p>}
      {f.interventions.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-fg-3">Fixes the optimiser can test:</span>
          {f.interventions.map((i) => <span key={i} className="badge">{i.replaceAll("_", " ")}</span>)}
        </div>
      )}
      {rows.length > 0 && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-[13px] text-fg-2">Evidence</summary>
          <table className="tbl mt-1"><tbody>
            {rows.map(([k, v]) => (
              <tr key={k}><td className="text-fg-3">{k}</td><td className="font-mono">{typeof v === "number" ? +v.toFixed(4) : String(v)}</td></tr>
            ))}
          </tbody></table>
        </details>
      )}
    </div>
  );
}

export function FindingsList({ findings, empty = "No findings." }: { findings: Finding[]; empty?: string }) {
  const [filter, setFilter] = useState<Severity | "all">("all");
  if (!findings.length) return <Empty>{empty}</Empty>;
  const shown = filter === "all" ? findings : findings.filter((f) => f.severity === filter);
  const count = (s: Severity) => findings.filter((f) => f.severity === s).length;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {(["all", "critical", "warning", "info"] as const).map((s) => (
          <button key={s} className={cx("btn", filter === s && "btn-primary")} onClick={() => setFilter(s)}>
            {s === "all" ? `All (${findings.length})` : `${SEV[s].label} (${count(s)})`}
          </button>
        ))}
      </div>
      {shown.map((f, i) => <FindingCard key={`${f.check}-${i}`} f={f} />)}
    </div>
  );
}

export function Meter({ value, label, explanation }: { value: number | null; label: string; explanation?: string }) {
  return (
    <div title={explanation}>
      <div className="flex justify-between">
        <span className="capitalize text-fg-2">{label}</span>
        <b className="tabular-nums">{value ?? "not measured"}</b>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded bg-surface-2" role="meter" aria-label={label}
        aria-valuenow={value ?? 0} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-r bg-series-1" style={{ width: `${value ?? 0}%` }} />
      </div>
      {explanation && <div className="mt-0.5 text-xs text-fg-3">{explanation}</div>}
    </div>
  );
}

export function HealthPanel({ d }: { d: Diagnosis }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="card">
        <div className="text-xs text-fg-3">MODEL HEALTH{d.model ? ` · ${d.model.name}` : ""}</div>
        <div className="my-3 flex items-baseline gap-2">
          <span className="text-6xl font-bold tabular-nums leading-none">{d.health.overall ?? "–"}</span>
          <span className="text-fg-3">/ 100</span>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span><SeverityBadge severity="critical" /> {d.severity_counts.critical}</span>
          <span><SeverityBadge severity="warning" /> {d.severity_counts.warning}</span>
          <span><SeverityBadge severity="info" /> {d.severity_counts.info}</span>
        </div>
        <p className="mt-3 text-xs text-fg-3">A summary built from transparent sub-scores; it does not replace the underlying metrics.</p>
      </div>
      <div className="card space-y-3">
        {Object.entries(d.health.components).map(([k, c]) => <Meter key={k} label={k} value={c.value} explanation={c.explanation} />)}
      </div>
    </div>
  );
}

export function Section({ title, children, actions, className }: { title: string; children: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <section className={cx("card", className)}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2>{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card">
      <div className="text-xs text-fg-3">{label}</div>
      <div className="my-1 text-2xl font-bold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-fg-3">{sub}</div>}
    </div>
  );
}

export function StageBadge({ stage }: { stage: string }) {
  return <span className="badge">{stage === "production" ? "★ " : ""}{stage}</span>;
}

export function Crumbs({ children }: { children: ReactNode }) {
  return <nav className="mb-2 flex flex-wrap gap-1 text-[13px] text-fg-3">{children}</nav>;
}
