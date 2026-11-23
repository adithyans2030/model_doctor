import type { ReactNode } from "react";
import { cx } from "@/lib/cx";

/** A placeholder block. Decorative; wrap groups in <LoadingRegion> for screen readers. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx("animate-shimmer rounded-md bg-surface-3", className)} />;
}

/** Announces loading once ("Loading models…") and renders the skeleton layout. */
export function LoadingRegion({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div role="status" aria-live="polite" aria-busy="true" className={className}>
      <span className="sr-only">{label}…</span>
      {children}
    </div>
  );
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cx("space-y-2", className)}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className={cx("h-3", i === lines - 1 ? "w-3/5" : "w-full")} />
      ))}
    </div>
  );
}

export function MetricTileSkeleton() {
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-7 w-20" />
      <Skeleton className="mt-3 h-3 w-32" />
    </div>
  );
}

export function TableSkeleton({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-0">
      <div className="flex gap-4 border-b border-line px-3 py-2.5">
        {Array.from({ length: cols }, (_, i) => <Skeleton key={i} className="h-3 flex-1" />)}
      </div>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="flex gap-4 border-b border-line px-3 py-3 last:border-b-0">
          {Array.from({ length: cols }, (_, i) => <Skeleton key={i} className={cx("h-3 flex-1", i === 0 && "max-w-40")} />)}
        </div>
      ))}
    </div>
  );
}

export function ChartSkeleton({ height = 180 }: { height?: number }) {
  return (
    <div aria-hidden className="flex items-end gap-1.5" style={{ height }}>
      {[45, 70, 55, 88, 62, 40, 75, 52, 66, 35].map((h, i) => (
        <div key={i} className="flex-1 animate-shimmer rounded-t-sm bg-surface-3" style={{ height: `${h}%` }} />
      ))}
    </div>
  );
}

export function CardSkeleton({ lines = 4, title = true }: { lines?: number; title?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-surface">
      {title && <div className="border-b border-line px-4 py-3.5"><Skeleton className="h-3.5 w-40" /></div>}
      <div className="p-4"><SkeletonText lines={lines} /></div>
    </div>
  );
}
