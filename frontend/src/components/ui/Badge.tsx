import type { HTMLAttributes, ReactNode } from "react";
import { cx } from "@/lib/cx";

export type Tone = "neutral" | "accent" | "success" | "warning" | "danger" | "info" | "ai";

const TONES: Record<Tone, string> = {
  neutral: "border-line bg-surface-2 text-fg-2",
  accent: "border-transparent bg-accent-subtle text-accent-fg",
  success: "border-transparent bg-success-subtle text-success",
  warning: "border-transparent bg-warning-subtle text-warning",
  danger: "border-transparent bg-danger-subtle text-danger",
  info: "border-transparent bg-info-subtle text-info",
  ai: "border-ai-border bg-ai-subtle text-ai",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  mono?: boolean;
  icon?: ReactNode;
}

export function Badge({ tone = "neutral", mono, icon, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cx(
        "inline-flex h-5 shrink-0 items-center gap-1 whitespace-nowrap rounded-sm border px-1.5 text-xs font-medium",
        mono && "font-mono font-normal",
        TONES[tone], className,
      )}
      {...rest}
    >
      {icon && <span aria-hidden className="inline-flex">{icon}</span>}
      {children}
    </span>
  );
}
