import { FileUp, X } from "lucide-react";
import { useId, useRef, useState, type DragEvent } from "react";
import { cx } from "@/lib/cx";

/**
 * Drag-and-drop or click-to-browse file picker. The native file input stays in the tab
 * order (visually hidden), so keyboard and screen-reader users get the platform picker.
 */
export function FileDrop({ accept, file, onFile, hint, label = "Choose a file", disabled }: {
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  hint?: string;
  label?: string;
  disabled?: boolean;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f && !disabled) onFile(f);
  };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={cx(
        "relative rounded-lg border border-dashed px-4 py-5 text-center transition-colors focus-within:border-ring",
        over ? "border-accent-fg bg-accent-subtle" : "border-line-strong bg-surface-2/40 hover:bg-surface-2",
        disabled && "opacity-60",
      )}
    >
      <input ref={input} id={id} type="file" accept={accept} disabled={disabled} className="peer sr-only"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)} aria-describedby={hint ? `${id}-hint` : undefined} />
      {file ? (
        <div className="flex items-center justify-center gap-2 text-sm">
          <FileUp size={16} aria-hidden className="text-accent-fg" />
          <span className="max-w-64 truncate font-mono">{file.name}</span>
          <span className="text-fg-3">{(file.size / 1024).toFixed(0)} KB</span>
          <button type="button" aria-label="Remove selected file" disabled={disabled}
            onClick={() => { onFile(null); if (input.current) input.current.value = ""; }}
            className="cursor-pointer rounded p-0.5 text-fg-3 hover:bg-surface-3 hover:text-fg">
            <X size={14} aria-hidden />
          </button>
        </div>
      ) : (
        <label htmlFor={id} className="cursor-pointer text-sm">
          <FileUp size={18} aria-hidden className="mx-auto mb-1.5 text-fg-3" />
          <span className="font-medium text-accent-fg">{label}</span> <span className="text-fg-2">or drag it here</span>
        </label>
      )}
      {hint && <p id={`${id}-hint`} className="mt-1.5 text-xs text-fg-3">{hint}</p>}
    </div>
  );
}
