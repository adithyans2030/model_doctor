import { X } from "lucide-react";
import { useEffect, useId, useRef, type ReactNode } from "react";
import { cx } from "@/lib/cx";
import { Button } from "./Button";

/**
 * Modal dialog on the native <dialog> element: showModal() gives a focus trap, inert
 * background and Escape-to-close for free. Focus returns to the opener on close.
 */
export function Modal({ open, onClose, title, description, children, footer, size = "md" }: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  size?: "sm" | "md" | "lg";
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const opener = useRef<Element | null>(null);
  const titleId = useId();
  const descId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      opener.current = document.activeElement;
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", ""); // jsdom and very old browsers
    } else if (!open && dialog.open) {
      if (typeof dialog.close === "function") dialog.close();
      else dialog.removeAttribute("open");
      if (opener.current instanceof HTMLElement) opener.current.focus();
    }
  }, [open]);

  return (
    // Backdrop click closes; the keyboard equivalent is Escape, handled natively via onCancel.
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      onCancel={(e) => { e.preventDefault(); onClose(); }}
      onClick={(e) => { if (e.target === ref.current) onClose(); }}
      className={cx(
        "m-auto w-[calc(100%-32px)] rounded-xl border border-line bg-surface p-0 text-fg shadow-lg",
        "backdrop:bg-black/60 backdrop:backdrop-blur-[2px]",
        size === "sm" ? "max-w-sm" : size === "lg" ? "max-w-2xl" : "max-w-lg",
      )}
    >
      {open && (
        <div className="flex max-h-[85vh] flex-col">
          <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
            <div>
              <h2 id={titleId} className="text-md font-semibold">{title}</h2>
              {description && <p id={descId} className="mt-1 text-sm text-fg-2">{description}</p>}
            </div>
            <Button variant="ghost" size="sm" aria-label="Close dialog" onClick={onClose} className="-mr-1.5 -mt-0.5 px-1.5">
              <X size={16} aria-hidden />
            </Button>
          </div>
          {children && <div className="overflow-y-auto px-5 py-4">{children}</div>}
          {footer && <div className="flex flex-wrap justify-end gap-2 border-t border-line bg-surface-2/50 px-5 py-3">{footer}</div>}
        </div>
      )}
    </dialog>
  );
}

/** Confirmation for destructive or consequential actions (replaces window.confirm). */
export function ConfirmDialog({ open, title, description, confirmLabel, tone = "danger", busy, onConfirm, onCancel, children }: {
  open: boolean;
  title: ReactNode;
  description?: ReactNode;
  confirmLabel: string;
  tone?: "danger" | "primary";
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  children?: ReactNode;
}) {
  return (
    <Modal open={open} onClose={onCancel} title={title} description={description} size="sm"
      footer={
        <>
          <Button onClick={onCancel} autoFocus>Cancel</Button>
          <Button variant={tone} loading={busy} onClick={onConfirm}>{confirmLabel}</Button>
        </>
      }>
      {children}
    </Modal>
  );
}
