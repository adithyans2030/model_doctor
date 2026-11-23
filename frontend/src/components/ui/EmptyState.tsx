import { AlertTriangle, RotateCw, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cx } from "@/lib/cx";
import { Button } from "./Button";

/** Nothing to show yet: say why, and what to do next. */
export function EmptyState({ icon: Icon, title, description, action, compact, className }: {
  icon?: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
  className?: string;
}) {
  return (
    <div className={cx("flex flex-col items-center text-center", compact ? "px-4 py-6" : "px-6 py-12", className)}>
      {Icon && (
        <div className="mb-3 grid size-9 place-items-center rounded-lg border border-line bg-surface-2 text-fg-3">
          <Icon size={18} aria-hidden />
        </div>
      )}
      <p className="text-sm font-medium text-fg">{title}</p>
      {description && <p className="mt-1 max-w-sm text-sm text-fg-3">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/** A failed request: the message, and a retry when the caller can refetch. */
export function ErrorState({ title = "Couldn't load this", error, onRetry, compact, className }: {
  title?: ReactNode;
  error: unknown;
  onRetry?: () => void;
  compact?: boolean;
  className?: string;
}) {
  const message = error instanceof Error ? error.message : typeof error === "string" ? error : "Unknown error";
  return (
    <div role="alert" className={cx("flex flex-col items-center text-center", compact ? "px-4 py-6" : "px-6 py-12", className)}>
      <div className="mb-3 grid size-9 place-items-center rounded-lg bg-danger-subtle text-danger">
        <AlertTriangle size={18} aria-hidden />
      </div>
      <p className="text-sm font-medium text-fg">{title}</p>
      <p className="mt-1 max-w-md break-words font-mono text-xs text-fg-2">{message}</p>
      {onRetry && <Button className="mt-4" size="sm" icon={<RotateCw size={14} />} onClick={onRetry}>Try again</Button>}
    </div>
  );
}

/** Inline error for a form or action (not a whole view). */
export function InlineError({ error, className }: { error: unknown; className?: string }) {
  if (!error) return null;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <p role="alert" className={cx("flex items-start gap-1.5 rounded-md bg-danger-subtle px-2.5 py-2 text-sm text-danger", className)}>
      <AlertTriangle size={14} aria-hidden className="mt-[3px] shrink-0" />
      <span className="min-w-0 break-words">{message}</span>
    </p>
  );
}
