import { cx } from "@/lib/cx";

/** Decorative spinner. Pair it with visible text or an aria-busy parent. */
export function Spinner({ size = 14, className }: { size?: number; className?: string }) {
  return (
    <span
      aria-hidden
      className={cx("inline-block shrink-0 animate-spin rounded-full border-2 border-line-strong border-t-accent-fg", className)}
      style={{ width: size, height: size }}
    />
  );
}
