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
      <div className="card border-critical" role="alert">
        <h2>Something went wrong{this.props.label ? ` in ${this.props.label}` : ""}</h2>
        <p className="mt-1 font-mono text-xs text-fg-2">{this.state.error.message}</p>
        <button className="btn mt-3" onClick={() => this.setState({ error: null })}>Try again</button>
      </div>
    );
  }
}
