import { useId, type ReactNode } from "react";
import { cx } from "@/lib/cx";

export interface CardProps {
  title?: ReactNode;
  description?: ReactNode;
  /** Controls shown at the right of the header (buttons, filters). */
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClassName?: string;
  /** Heading level for the title; 2 by default. */
  level?: 2 | 3;
  /** Remove body padding (for full-bleed tables and charts). */
  flush?: boolean;
  id?: string;
}

/** A bordered surface. With a title it renders as a labelled <section>. */
export function Card({ title, description, actions, children, className, bodyClassName, level = 2, flush, id }: CardProps) {
  const headingId = useId();
  const Heading = level === 2 ? "h2" : "h3";
  const hasHeader = title || actions;
  const Tag = title ? "section" : "div";
  return (
    <Tag
      id={id}
      aria-labelledby={title ? headingId : undefined}
      className={cx("min-w-0 rounded-lg border border-line bg-surface shadow-sm", className)}
    >
      {hasHeader && (
        <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {title && <Heading id={headingId} className="text-sm font-semibold text-fg">{title}</Heading>}
            {description && <p className="mt-0.5 text-xs text-fg-3">{description}</p>}
          </div>
          {actions && <div className="flex min-w-0 max-w-full flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx(!flush && "p-4", bodyClassName)}>{children}</div>
    </Tag>
  );
}
