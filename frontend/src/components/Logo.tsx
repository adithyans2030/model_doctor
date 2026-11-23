import { cx } from "@/lib/cx";

/** Wordmark: a pulse line in a rounded square, then the name. */
export function Logo({ className, withName = true }: { className?: string; withName?: boolean }) {
  return (
    <span className={cx("inline-flex items-center gap-2 font-semibold tracking-tight text-fg", className)}>
      <svg viewBox="0 0 24 24" className="size-6 shrink-0" aria-hidden>
        <rect width="24" height="24" rx="6" fill="var(--accent)" />
        <path d="M4.5 12.5h3.2l1.8-4.5 3 9 2.2-6 1.3 1.5h3.5" fill="none" stroke="var(--accent-ink)" strokeWidth="1.8"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {withName && <span>ModelDoctor</span>}
    </span>
  );
}
