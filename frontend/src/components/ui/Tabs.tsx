import { useRef, type KeyboardEvent, type ReactNode } from "react";
import { cx } from "@/lib/cx";

export interface TabItem<T extends string> {
  value: T;
  label: ReactNode;
  icon?: ReactNode;
  /** Small count shown after the label (e.g. number of findings). */
  count?: number;
}

export const tabId = (base: string, value: string) => `${base}-tab-${value}`;
export const panelId = (base: string, value: string) => `${base}-panel-${value}`;

/**
 * WAI-ARIA tablist with automatic activation: arrow keys / Home / End move focus and
 * select; only the selected tab is in the tab order (roving tabindex).
 */
export function Tabs<T extends string>({ items, value, onChange, label, idBase, className }: {
  items: TabItem<T>[];
  value: T;
  onChange: (value: T) => void;
  /** Accessible name for the tab list. */
  label: string;
  /** Prefix for tab / panel ids; pair with <TabPanel idBase=…>. */
  idBase: string;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  function onKeyDown(e: KeyboardEvent, index: number) {
    const last = items.length - 1;
    const next = e.key === "ArrowRight" ? (index === last ? 0 : index + 1)
      : e.key === "ArrowLeft" ? (index === 0 ? last : index - 1)
      : e.key === "Home" ? 0 : e.key === "End" ? last : null;
    if (next === null) return;
    e.preventDefault();
    refs.current[next]?.focus();
    onChange(items[next].value);
  }

  return (
    <div
      role="tablist"
      aria-label={label}
      className={cx("flex gap-1 overflow-x-auto border-b border-line [scrollbar-width:none]", className)}
    >
      {items.map((item, i) => {
        const selected = item.value === value;
        return (
          <button
            key={item.value}
            ref={(el) => { refs.current[i] = el; }}
            type="button"
            role="tab"
            id={tabId(idBase, item.value)}
            aria-selected={selected}
            aria-controls={panelId(idBase, item.value)}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.value)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cx(
              "relative -mb-px inline-flex h-9 cursor-pointer items-center gap-1.5 whitespace-nowrap border-b-2 px-2.5 text-sm transition-colors",
              selected ? "border-accent-fg font-medium text-fg" : "border-transparent text-fg-2 hover:text-fg",
            )}
          >
            {item.icon && <span aria-hidden className="inline-flex text-fg-3">{item.icon}</span>}
            {item.label}
            {item.count != null && (
              <span className="rounded-sm bg-surface-2 px-1 font-mono text-xs text-fg-3">{item.count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({ idBase, value, children, className }: {
  idBase: string; value: string; children: ReactNode; className?: string;
}) {
  return (
    <div role="tabpanel" id={panelId(idBase, value)} aria-labelledby={tabId(idBase, value)} tabIndex={-1}
      className={cx("pt-5 outline-none", className)}>
      {children}
    </div>
  );
}
