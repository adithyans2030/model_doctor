import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import type { ReactNode } from "react";
import { cx } from "@/lib/cx";

export interface MetricDelta {
  /** Signed change (after − before). */
  value: number;
  format?: (v: number) => string;
  /** Whether an increase is an improvement (true for recall/F1/AUC, false for RMSE). */
  higherIsBetter?: boolean;
  /** Context for the change, e.g. "vs baseline". */
  label?: string;
}

/** Direction of a change relative to "better". Exported for tables that show deltas. */
export function deltaSentiment(value: number, higherIsBetter = true, epsilon = 1e-9): "better" | "worse" | "flat" {
  if (Math.abs(value) < epsilon) return "flat";
  return value > 0 === higherIsBetter ? "better" : "worse";
}

export function DeltaBadge({ delta }: { delta: MetricDelta }) {
  const fmt = delta.format ?? ((v: number) => v.toFixed(3));
  const sentiment = deltaSentiment(delta.value, delta.higherIsBetter ?? true);
  const Icon = sentiment === "flat" ? Minus : delta.value > 0 ? ArrowUpRight : ArrowDownRight;
  const sign = delta.value > 0 ? "+" : delta.value < 0 ? "−" : "±";
  return (
    <span className={cx(
      "inline-flex items-center gap-0.5 rounded-sm px-1 font-mono text-xs font-medium",
      sentiment === "better" ? "bg-success-subtle text-success" : sentiment === "worse" ? "bg-danger-subtle text-danger" : "bg-surface-2 text-fg-3",
    )}>
      <Icon size={12} aria-hidden />
      {sign}{fmt(Math.abs(delta.value))}
      <span className="sr-only"> ({sentiment === "flat" ? "no change" : sentiment}{delta.label ? `, ${delta.label}` : ""})</span>
    </span>
  );
}

/** A labelled metric: big monospace value, optional before value and signed delta. */
export function MetricTile({ label, value, before, delta, hint, icon, emphasis, className }: {
  label: ReactNode;
  value: ReactNode;
  /** Previous value, shown muted before an arrow ("0.52 → 0.65"). */
  before?: ReactNode;
  delta?: MetricDelta;
  hint?: ReactNode;
  icon?: ReactNode;
  emphasis?: boolean;
  className?: string;
}) {
  return (
    <div className={cx("min-w-0 rounded-lg border bg-surface p-4", emphasis ? "border-accent/50" : "border-line", className)}>
      <div className="flex items-center gap-1.5 text-xs font-medium text-fg-3">
        {icon && <span aria-hidden className="inline-flex">{icon}</span>}
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {before != null && (
          <>
            <span className="font-mono text-md tabular-nums text-fg-3"><span className="sr-only">from </span>{before}</span>
            <span aria-hidden className="text-fg-3">→</span>
            <span className="sr-only">to</span>
          </>
        )}
        <span className="font-mono text-xl font-medium tabular-nums tracking-tight text-fg">{value}</span>
        {delta && <DeltaBadge delta={delta} />}
      </div>
      {hint && <div className="mt-1.5 text-xs text-fg-3">{hint}</div>}
    </div>
  );
}
