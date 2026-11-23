import { CircleDot, Info, OctagonAlert, TriangleAlert, type LucideIcon } from "lucide-react";
import { Badge, type Tone } from "@/components/ui";
import type { Finding, Severity, Stage } from "@/lib/types";

export const SEVERITY: Record<Severity, { label: string; tone: Tone; icon: LucideIcon; rank: number; rail: string; text: string }> = {
  critical: { label: "Critical", tone: "danger", icon: OctagonAlert, rank: 0, rail: "bg-danger", text: "text-danger" },
  warning: { label: "Warning", tone: "warning", icon: TriangleAlert, rank: 1, rail: "bg-warning", text: "text-warning" },
  info: { label: "Info", tone: "info", icon: Info, rank: 2, rail: "bg-line-strong", text: "text-info" },
};

/** Status colour never carries meaning alone: each severity has its own icon shape and label. */
export function SeverityBadge({ severity, count }: { severity: Severity; count?: number }) {
  const s = SEVERITY[severity];
  const Icon = s.icon;
  return (
    <Badge tone={s.tone} icon={<Icon size={12} />}>
      {s.label}{count != null && <span className="font-mono">{count}</span>}
    </Badge>
  );
}

/** Stable sort: critical first, then warning, then info; API order within a severity. */
export function sortBySeverity<T extends Pick<Finding, "severity">>(items: T[]): T[] {
  return items.map((f, i) => [f, i] as const)
    .sort(([a, i], [b, j]) => SEVERITY[a.severity].rank - SEVERITY[b.severity].rank || i - j)
    .map(([f]) => f);
}

export const findingAnchor = (check: string) => `finding-${check.replace(/[^a-z0-9]+/gi, "-")}`;

const STAGE: Record<Stage, { tone: Tone; label: string }> = {
  production: { tone: "success", label: "Production" },
  approved: { tone: "info", label: "Approved" },
  candidate: { tone: "accent", label: "Candidate" },
  rejected: { tone: "danger", label: "Rejected" },
  archived: { tone: "neutral", label: "Archived" },
};

export function StageBadge({ stage }: { stage: Stage }) {
  const s = STAGE[stage] ?? { tone: "neutral" as Tone, label: stage };
  return <Badge tone={s.tone} icon={stage === "production" ? <CircleDot size={11} /> : undefined}>{s.label}</Badge>;
}
