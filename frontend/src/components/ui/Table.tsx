import type { HTMLAttributes, ReactNode, TdHTMLAttributes, ThHTMLAttributes } from "react";
import { cx } from "@/lib/cx";

/**
 * A table inside a horizontally scrollable region. The region is focusable and labelled
 * so keyboard users can scroll wide tables.
 */
export function Table({ label, caption, children, className, dense }: {
  /** Accessible name for the scroll region and table. */
  label: string;
  /** Visible caption; defaults to an sr-only caption with `label`. */
  caption?: ReactNode;
  children: ReactNode;
  className?: string;
  dense?: boolean;
}) {
  return (
    <div role="region" aria-label={label} tabIndex={0} className={cx("relative max-w-full overflow-x-auto", className)}>
      <table className={cx("w-full border-collapse text-sm", dense && "[&_td]:py-1 [&_th]:py-1")}>
        <caption className={caption ? "pb-2 text-left text-xs text-fg-3" : "sr-only"}>{caption ?? label}</caption>
        {children}
      </table>
    </div>
  );
}

export function THead({ children }: { children: ReactNode }) {
  return <thead className="[&_th]:sticky [&_th]:top-0">{children}</thead>;
}

export function TBody({ children }: { children: ReactNode }) {
  return <tbody className="[&>tr:last-child>td]:border-b-0">{children}</tbody>;
}

export function TR({ className, ...rest }: HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={cx("transition-colors hover:bg-surface-2/60", className)} {...rest} />;
}

export function TH({ numeric, className, scope = "col", ...rest }: ThHTMLAttributes<HTMLTableCellElement> & { numeric?: boolean }) {
  return (
    <th
      scope={scope}
      className={cx(
        "whitespace-nowrap border-b border-line bg-surface px-3 py-2 text-left align-bottom text-xs font-medium text-fg-3",
        numeric && "text-right", className,
      )}
      {...rest}
    />
  );
}

export function TD({ numeric, className, ...rest }: TdHTMLAttributes<HTMLTableCellElement> & { numeric?: boolean }) {
  return (
    <td
      className={cx("border-b border-line px-3 py-2 align-top", numeric && "text-right font-mono tabular-nums", className)}
      {...rest}
    />
  );
}
