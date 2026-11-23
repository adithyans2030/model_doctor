import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { cx } from "@/lib/cx";

type ToastTone = "success" | "danger" | "info";
export interface ToastInput { title: string; description?: string; tone?: ToastTone; duration?: number }
interface ToastItem extends Required<Omit<ToastInput, "description">> { id: number; description?: string }

const ToastContext = createContext<((t: ToastInput) => void) | null>(null);

const ICONS = { success: CheckCircle2, danger: AlertTriangle, info: Info } as const;
const ICON_TONE = { success: "text-success", danger: "text-danger", info: "text-info" } as const;

/** Provides `useToast()`. Toasts are announced through a polite live region. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => setItems((all) => all.filter((t) => t.id !== id)), []);
  const push = useCallback((t: ToastInput) => {
    const id = next.current++;
    const item: ToastItem = { id, title: t.title, description: t.description, tone: t.tone ?? "info", duration: t.duration ?? 5000 };
    setItems((all) => [...all.slice(-3), item]);
    if (item.duration > 0) setTimeout(() => dismiss(id), item.duration);
  }, [dismiss]);

  const value = useMemo(() => push, [push]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        role="region"
        aria-label="Notifications"
        className="pointer-events-none fixed inset-x-4 bottom-4 z-[60] flex flex-col items-end gap-2 sm:left-auto sm:w-96"
      >
        <div aria-live="polite" className="contents">
          {items.map((t) => {
            const Icon = ICONS[t.tone];
            return (
              <div key={t.id} role="status"
                className="pointer-events-auto flex w-full items-start gap-2.5 rounded-lg border border-line bg-surface-2 p-3 shadow-lg">
                <Icon size={16} aria-hidden className={cx("mt-0.5 shrink-0", ICON_TONE[t.tone])} />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{t.title}</p>
                  {t.description && <p className="mt-0.5 text-sm text-fg-2">{t.description}</p>}
                </div>
                <button type="button" aria-label="Dismiss notification" onClick={() => dismiss(t.id)}
                  className="-m-1 cursor-pointer rounded p-1 text-fg-3 hover:bg-surface-3 hover:text-fg">
                  <X size={14} aria-hidden />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
