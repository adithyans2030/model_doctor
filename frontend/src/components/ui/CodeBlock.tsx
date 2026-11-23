import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";
import { cx } from "@/lib/cx";

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** Monospace block with a copy button. Lines starting with "#" are shown as comments. */
export function CodeBlock({ code, label, className }: {
  code: string;
  /** Shown in the header and used to name the copy button, e.g. "Terminal". */
  label?: string;
  className?: string;
}) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const t = setTimeout(() => setState("idle"), 2000);
    return () => clearTimeout(t);
  }, [state]);

  return (
    <div className={cx("overflow-hidden rounded-lg border border-line bg-surface-2", className)}>
      <div className="flex h-9 items-center justify-between border-b border-line pl-3.5 pr-1.5">
        <span className="text-xs text-fg-3">{label ?? "Code"}</span>
        <button
          type="button"
          onClick={async () => setState((await copyText(code)) ? "copied" : "failed")}
          aria-label={`Copy ${label ? label.toLowerCase() : "code"} to clipboard`}
          className="inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-md px-2 text-xs text-fg-2 transition-colors hover:bg-surface-3 hover:text-fg"
        >
          {state === "copied" ? <Check size={14} aria-hidden className="text-success" /> : <Copy size={14} aria-hidden />}
          <span aria-hidden>{state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}</span>
        </button>
        <span className="sr-only" aria-live="polite">{state === "copied" ? "Copied to clipboard" : state === "failed" ? "Copy failed" : ""}</span>
      </div>
      <pre tabIndex={0} className="overflow-x-auto px-3.5 py-3 font-mono text-sm leading-6">
        <code>
          {code.split("\n").map((line, i) => (
            <span key={i} className={cx("block", line.trimStart().startsWith("#") ? "text-fg-3" : "text-fg")}>
              {line || " "}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
