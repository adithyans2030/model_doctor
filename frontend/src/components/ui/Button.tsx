import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";
import { cx } from "@/lib/cx";
import { Spinner } from "./Spinner";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md" | "lg";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-ink shadow-sm hover:bg-accent-hover",
  secondary: "border border-line bg-surface-2 text-fg hover:border-line-strong hover:bg-surface-3",
  ghost: "text-fg-2 hover:bg-surface-2 hover:text-fg",
  danger: "border border-line bg-surface-2 text-danger hover:border-danger/40 hover:bg-danger-subtle",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-7 gap-1 px-2.5 text-xs",
  md: "h-8 gap-1.5 px-3 text-sm",
  lg: "h-10 gap-2 px-4 text-base",
};

/** Classes for anything that should look like a button (e.g. a router <Link>). */
export function buttonClasses({ variant = "secondary", size = "md", className }: {
  variant?: ButtonVariant; size?: ButtonSize; className?: string;
} = {}): string {
  return cx(
    "inline-flex cursor-pointer select-none items-center justify-center whitespace-nowrap rounded-md font-medium",
    "no-underline transition-colors hover:no-underline disabled:pointer-events-none disabled:opacity-50",
    "aria-disabled:pointer-events-none aria-disabled:opacity-50",
    VARIANTS[variant], SIZES[size], className,
  );
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner, disables the button and sets aria-busy. */
  loading?: boolean;
  /** Leading icon (decorative; the label carries the meaning). */
  icon?: ReactNode;
  ref?: Ref<HTMLButtonElement>;
}

export function Button({
  variant = "secondary", size = "md", loading = false, icon, className, children, disabled, type = "button", ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={buttonClasses({ variant, size, className })}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size={size === "lg" ? 16 : 14} /> : icon && <span aria-hidden className="inline-flex shrink-0">{icon}</span>}
      {children}
    </button>
  );
}
