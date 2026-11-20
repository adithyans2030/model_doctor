import { useEffect, useRef, useState } from "react";
import { api, streamNdjson } from "@/lib/api";
import { cx, Spinner } from "./ui";

type Msg = { role: "user" | "assistant"; content: string; meta?: string; pending?: boolean };
type Providers = {
  provider: "ollama" | "anthropic" | "rules";
  ollama_available: boolean;
  ollama_url: string;
  default_model: string;
  local_models: { name: string; parameters: string | null; size_gb: number }[];
};

const SUGGESTIONS = [
  "Why is my recall so low?",
  "What should I fix first?",
  "What did the experiment find?",
  "Which features drive the model?",
  "Compare model version 1 and version 2",
];
const MODEL_KEY = "md_local_model";

export function Assistant({ datasetId, modelId }: { datasetId: string; modelId?: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [providers, setProviders] = useState<Providers | null>(null);
  const [localModel, setLocalModel] = useState<string>(() => {
    try { return localStorage.getItem(MODEL_KEY) || ""; } catch { return ""; }
  });
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api<Providers>("/assistant/providers").then(setProviders).catch(() => setProviders(null));
    return () => abort.current?.abort();
  }, []);
  useEffect(() => end.current?.scrollIntoView({ block: "nearest" }), [messages]);

  const model = localModel || providers?.default_model || "";
  function pickModel(name: string) {
    setLocalModel(name);
    try { localStorage.setItem(MODEL_KEY, name); } catch { /* ignore */ }
  }

  async function send(question: string) {
    if (!question.trim() || busy) return;
    const history = messages.filter((m) => !m.pending).map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", pending: true }]);
    setInput("");
    setBusy(true);
    abort.current = new AbortController();
    const update = (fn: (m: Msg) => Msg) =>
      setMessages((all) => all.map((m, i) => (i === all.length - 1 ? fn(m) : m)));
    let mode = "", used: string | null = null;
    try {
      await streamNdjson("/assistant/ask/stream", {
        dataset_id: datasetId, model_id: modelId, question, history,
        llm_model: providers?.provider === "ollama" ? model : undefined,
      }, (e) => {
        if (e.type === "meta") { mode = e.mode; used = e.model; }
        else if (e.type === "delta") update((m) => ({ ...m, content: m.content + e.text }));
        else if (e.type === "done") {
          const label = mode === "ollama" ? `local model (${used})` : mode === "anthropic" ? `Claude (${used})` : "rule-based";
          const meta = [label, e.sources?.length ? `cites ${e.sources.join(", ")}` : "", e.note || ""].filter(Boolean).join(" · ");
          update((m) => ({ ...m, pending: false, meta }));
        }
      }, abort.current.signal);
    } catch (err) {
      update(() => ({ role: "assistant", content: (err as Error).message, meta: "error" }));
    } finally {
      update((m) => ({ ...m, pending: false }));
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2>Ask ModelDoctor</h2>
          <p className="mt-1 text-[13px] text-fg-3">Answers are built from this dataset's diagnosis, models, experiments and drift reports. The assistant explains results; it doesn't invent them.</p>
        </div>
        <ProviderPicker providers={providers} model={model} onPick={pickModel} disabled={busy} />
      </div>

      <div className="mt-3 flex max-h-[520px] flex-col gap-2.5 overflow-y-auto p-1" aria-live="polite">
        {messages.length === 0 && (
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((s) => <button key={s} className="btn" onClick={() => send(s)}>{s}</button>)}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={cx("max-w-[88%] whitespace-pre-wrap rounded-xl px-3 py-2.5",
            m.role === "user" ? "self-end bg-accent text-accent-ink" : "self-start bg-surface-2")}>
            {m.content || (m.pending && (
              <span className="text-fg-3"><Spinner /> {providers?.provider === "ollama" ? `${model} is thinking (local models can take a minute)…` : "thinking…"}</span>
            ))}
            {m.pending && m.content && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-fg-3 align-[-2px]" />}
            {m.meta && <div className="mt-1.5 text-xs opacity-70">{m.meta}</div>}
          </div>
        ))}
        <div ref={end} />
      </div>

      <form className="mt-3 flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input className="field min-w-52 flex-1" value={input} onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about recall, fixes, drift, experiments, model versions…" maxLength={2000} />
        {busy
          ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Stop</button>
          : <button className="btn btn-primary" disabled={!input.trim()}>Ask</button>}
      </form>
    </section>
  );
}

function ProviderPicker({ providers, model, onPick, disabled }: {
  providers: Providers | null; model: string; onPick: (m: string) => void; disabled: boolean;
}) {
  if (!providers) return <span className="badge">assistant status unknown</span>;
  if (providers.provider === "ollama") {
    return (
      <label className="flex items-center gap-2 text-[13px] text-fg-2">
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-good" aria-hidden />Local model</span>
        <select className="field" value={model} onChange={(e) => onPick(e.target.value)} disabled={disabled}>
          {providers.local_models.map((m) => (
            <option key={m.name} value={m.name}>{m.name}{m.parameters ? ` · ${m.parameters}` : ""}</option>
          ))}
        </select>
      </label>
    );
  }
  if (providers.provider === "anthropic") return <span className="badge">Claude API</span>;
  return (
    <span className="badge" title={`Start Ollama (${providers.ollama_url}) to use a local model`}>
      rule-based answers · Ollama not running
    </span>
  );
}
