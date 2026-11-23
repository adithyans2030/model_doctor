import { Card } from "@/components/ui";
import { cx } from "@/lib/cx";
import type { Diagnosis } from "@/lib/types";
import { SeverityBadge } from "./severity";

const band = (v: number) => (v >= 80 ? "text-success" : v >= 60 ? "text-warning" : "text-danger");
const bandLabel = (v: number) => (v >= 80 ? "healthy" : v >= 60 ? "needs attention" : "unhealthy");

/** Ring gauge for a 0–100 score. The number is the content; the ring is decoration. */
export function ScoreRing({ value, size = 112, label }: { value: number | null; size?: number; label: string }) {
  const r = size / 2 - 7;
  const c = 2 * Math.PI * r;
  const v = value ?? 0;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} aria-hidden className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth="6" />
        {value != null && (
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round"
            strokeDasharray={`${(v / 100) * c} ${c}`} className={band(v)} />
        )}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="font-mono text-2xl font-medium leading-none tabular-nums">{value ?? "–"}</div>
          <div className="mt-1 text-xs text-fg-3">/ 100</div>
        </div>
      </div>
      <span className="sr-only">{label}: {value == null ? "not measured" : `${value} out of 100, ${bandLabel(v)}`}</span>
    </div>
  );
}

export function ComponentBar({ name, value, explanation }: { name: string; value: number | null; explanation: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="capitalize text-fg">{name}</span>
        <span className={cx("font-mono tabular-nums", value == null ? "text-fg-3" : "text-fg")}>{value ?? "not measured"}</span>
      </div>
      {/* Only a measured value is a meter; "not measured" is already in the text above. */}
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-3"
        {...(value != null ? { role: "meter", "aria-label": `${name} score`, "aria-valuenow": value, "aria-valuemin": 0, "aria-valuemax": 100 } : { "aria-hidden": true })}>
        {value != null && <div className={cx("h-full rounded-full", value >= 80 ? "bg-success" : value >= 60 ? "bg-warning" : "bg-danger")} style={{ width: `${value}%` }} />}
      </div>
      <p className="mt-1 text-xs text-fg-3">{explanation}</p>
    </div>
  );
}

/** Overall health, severity counts and the transparent component scores behind it. */
export function HealthPanel({ d }: { d: Diagnosis }) {
  return (
    <Card title="Model health" description={d.model ? `${d.model.name} · computed from transparent sub-scores` : "Computed from transparent sub-scores"}>
      <div className="grid gap-6 md:grid-cols-[auto_1fr]">
        <div className="flex items-center gap-5 md:flex-col md:items-start">
          <ScoreRing value={d.health.overall} label="Overall health" />
          <ul aria-label="Findings by severity" className="flex flex-wrap gap-1.5 md:flex-col md:items-start">
            {(["critical", "warning", "info"] as const).map((s) => (
              <li key={s}><SeverityBadge severity={s} count={d.severity_counts[s]} /></li>
            ))}
          </ul>
        </div>
        <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
          {Object.entries(d.health.components).map(([k, c]) => <ComponentBar key={k} name={k} value={c.value} explanation={c.explanation} />)}
        </div>
      </div>
    </Card>
  );
}
