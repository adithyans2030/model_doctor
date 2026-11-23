import { ChevronRight } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import { Link } from "react-router-dom";

export interface Crumb { label: string; to?: string }

export function PageHeader({ crumbs, title, meta, actions }: {
  crumbs?: Crumb[];
  title: ReactNode;
  /** A row of facts under the title (rows, target, stage…). */
  meta?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-5">
      {crumbs && crumbs.length > 0 && (
        <nav aria-label="Breadcrumb" className="mb-2">
          <ol className="flex flex-wrap items-center gap-1 text-xs text-fg-3">
            {crumbs.map((c, i) => (
              <Fragment key={i}>
                {i > 0 && <li aria-hidden><ChevronRight size={12} /></li>}
                <li className="max-w-56 truncate">
                  {c.to ? <Link to={c.to} className="text-fg-3 hover:text-fg">{c.label}</Link> : <span aria-current="page" className="text-fg-2">{c.label}</span>}
                </li>
              </Fragment>
            ))}
          </ol>
        </nav>
      )}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="break-words text-xl font-semibold tracking-tight">{title}</h1>
          {meta && <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-sm text-fg-2">{meta}</div>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}

/** A "label value" pair for PageHeader meta rows. */
export function Meta({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-fg-3">{label}</span>
      <span className="font-mono text-fg">{children}</span>
    </span>
  );
}
