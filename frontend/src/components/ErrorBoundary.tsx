import { AlertTriangle, RotateCw } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode; label?: string };
type State = { error: Error | null };

/** Contains a crash to the section that failed instead of blanking the whole app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ModelDoctor UI error:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="rounded-lg border border-danger/40 bg-surface p-5" role="alert">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <AlertTriangle size={16} aria-hidden className="text-danger" />
          Something went wrong{this.props.label ? ` in ${this.props.label}` : ""}
        </h2>
        <p className="mt-1.5 break-words font-mono text-xs text-fg-2">{this.state.error.message}</p>
        <button type="button" onClick={() => this.setState({ error: null })}
          className="mt-3 inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-surface-2 px-3 text-sm hover:bg-surface-3">
          <RotateCw size={14} aria-hidden /> Try again
        </button>
      </div>
    );
  }
}
