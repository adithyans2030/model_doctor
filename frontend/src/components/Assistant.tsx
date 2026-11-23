import { ArrowUp, Link2, Sparkles, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge, Button, Spinner } from "@/components/ui";
import { api, streamNdjson } from "@/lib/api";
import { cx } from "@/lib/cx";
import { useProviders } from "@/lib/queries";
import type { AssistantEvent, Providers } from "@/lib/types";
import { findingAnchor } from "./domain/severity";
import { Markdown } from "./Markdown";

type Msg = { role: "user" | "assistant"; content: string; provider?: string; sources?: string[]; note?: string; error?: boolean; pending?: boolean };
type Warm = "idle" | "loading" | "ready" | "failed";

const SUGGESTIONS = [
  "Why is my recall so low?",
  "What should I fix first?",
  "What did the experiment find?",
  "Which features drive the model?",
  "Compare model version 1 and version 2",
];
const MODEL_KEY = "md_local_model";

/**
 * The AI panel. It explains results that were already computed deterministically
 * (findings, metrics, experiments); it never produces a diagnosis of its own. Each
 * answer lists the finding ids it cites; on pages that show those findings, the
 * citations link to them.
 */
export function Assistant({ datasetId, modelId, compact, linkCitations, suggestions = SUGGESTIONS }: {
  datasetId: string;
  modelId?: string;
  compact?: boolean;
  /** Link cited finding ids to finding cards on the same page. */
  linkCitations?: boolean;
  suggestions?: string[];
}) {
  const providersQuery = useProviders();
  const providers = providersQuery.data ?? null;
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [localModel, setLocalModel] = useState<string>(() => {
    try { return localStorage.getItem(MODEL_KEY) || ""; } catch { return ""; }
  });
  const [warmState, setWarmState] = useState<{ model: string; status: Warm } | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);

  const model = localModel || providers?.default_model || "";
  const usesOllama = providers?.provider === "ollama" && !!model;
  const warm: Warm = !usesOllama ? "idle" : warmState?.model === model ? warmState.status : "loading";
  const elapsed = startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;

  useEffect(() => () => abort.current?.abort(), []);
  // Braces matter: scrollIntoView() returns a Promise in newer browsers, and an effect
  // must return nothing or a cleanup function.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);

  // Load the selected local model as soon as the panel opens, so the first answer does
  // not also pay the (up to a minute) model-loading time on a CPU-only machine.
  useEffect(() => {
    if (!usesOllama) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const started = Date.now();
    const done = (status: Warm) => { if (alive) setWarmState({ model, status }); };
    const poll = async () => {
      try {
        const p = await api<Providers>("/assistant/providers");
        if (!alive) return;
        if (p.loaded_models.includes(model)) return done("ready");
      } catch { /* keep polling */ }
      if (Date.now() - started > 180_000) return done("failed");
      timer = setTimeout(poll, 3000);
    };
    api<{ status: string }>("/assistant/warmup", { json: { llm_model: model } })
      .then((r) => (r.status === "loaded" ? done("ready") : r.status === "loading" ? poll() : done("failed")))
      .catch(() => done("failed"));
    return () => { alive = false; clearTimeout(timer); };
  }, [usesOllama, model]);

  // Running clock while waiting for an answer.
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [busy]);

  function pickModel(name: string) {
    setLocalModel(name);
    try { localStorage.setItem(MODEL_KEY, name); } catch { /* ignore */ }
  }

  async function send(question: string) {
    if (!question.trim() || busy) return;
    const history = messages.filter((m) => !m.pending && !m.error).map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", pending: true }]);
    setInput("");
    setBusy(true);
    setStartedAt(Date.now());
    setNow(Date.now());
    abort.current = new AbortController();
    const update = (fn: (m: Msg) => Msg) => setMessages((all) => all.map((m, i) => (i === all.length - 1 ? fn(m) : m)));
    let mode = "", used: string | null = null;
    try {
      await streamNdjson<AssistantEvent>("/assistant/ask/stream", {
        dataset_id: datasetId, model_id: modelId, question, history,
        llm_model: providers?.provider === "ollama" ? model : undefined,
      }, (e) => {
        if (e.type === "meta") { mode = e.mode; used = e.model; }
        else if (e.type === "delta") update((m) => ({ ...m, content: m.content + e.text }));
        else if (e.type === "done") {
          const provider = mode === "ollama" ? `local model (${used})` : mode === "anthropic" ? `Claude (${used})` : "rule-based";
          update((m) => ({ ...m, pending: false, provider, sources: e.sources ?? [], note: e.note }));
        }
      }, abort.current.signal);
    } catch (err) {
      update(() => ({ role: "assistant", content: (err as Error).message, error: true }));
    } finally {
      update((m) => ({ ...m, pending: false }));
      setBusy(false);
      setStartedAt(null);
    }
  }

  return (
    <section aria-labelledby="ai-panel-title" className="rounded-lg border border-ai-border bg-ai-subtle">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-ai-border px-4 py-3">
        <div className="min-w-0">
          <h2 id="ai-panel-title" className="flex items-center gap-2 text-sm font-semibold">
            <Sparkles size={15} aria-hidden className="text-ai" />
            AI explanation
            <Badge tone="ai">explains deterministic results</Badge>
          </h2>
          <p className="mt-1 max-w-2xl text-xs text-fg-2">
            Every number and finding below was computed by ModelDoctor's statistical checks. The assistant only puts them
            into words, and each answer lists the findings it cites.
          </p>
        </div>
        <ProviderPicker providers={providers} unknown={providersQuery.isError} model={model} onPick={pickModel} disabled={busy} warm={warm} />
      </header>

      <div className={cx("flex flex-col gap-3 overflow-y-auto px-4 py-4", compact ? "max-h-[420px]" : "max-h-[560px]")}
        role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && (
          <div>
            <p className="mb-2 text-xs text-fg-3">Try asking</p>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((s) => (
                <button key={s} type="button" onClick={() => send(s)}
                  className="h-8 cursor-pointer rounded-full border border-ai-border bg-surface px-3 text-sm text-fg-2 transition-colors hover:border-ai hover:text-fg">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} data-role={m.role}
            className={cx("max-w-[92%] rounded-lg px-3.5 py-2.5 text-sm",
              m.role === "user" ? "self-end bg-surface-3 text-fg" : "self-start border border-line bg-surface",
              m.error && "border-danger/40 text-danger")}>
            {m.content ? (m.role === "assistant" && !m.error ? <Markdown text={m.content} /> : <span className="whitespace-pre-wrap">{m.content}</span>) : (m.pending && (
              <span className="flex items-center gap-2 text-fg-3">
                <Spinner /> {usesOllama ? `${model} is ${warm === "ready" ? "thinking" : "loading and thinking"}… ${elapsed}s` : `Thinking… ${elapsed}s`}
                {usesOllama && elapsed >= 15 && (
                  <span className="block text-xs">Local models run on your CPU; 8B models can take about a minute. gemma2:2b is faster.</span>
                )}
              </span>
            ))}
            {m.pending && m.content && <span aria-hidden className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-fg-3 align-[-2px]" />}
            {m.provider && (
              <div className="mt-2.5 flex flex-wrap items-center gap-1.5 border-t border-line pt-2 text-xs text-fg-3">
                <span>{m.provider}</span>
                {m.sources && m.sources.length > 0 ? (
                  <>
                    <span aria-hidden>·</span><span>cites</span>
                    {m.sources.map((s) => linkCitations ? (
                      <a key={s} href={`#${findingAnchor(s)}`} className="no-underline hover:no-underline">
                        <Badge mono tone="ai" icon={<Link2 size={11} />}>{s}</Badge>
                      </a>
                    ) : <Badge key={s} mono tone="ai">{s}</Badge>)}
                  </>
                ) : <><span aria-hidden>·</span><span>no findings cited</span></>}
                {m.note && <><span aria-hidden>·</span><span>{m.note}</span></>}
              </div>
            )}
          </div>
        ))}
        <div ref={end} />
      </div>

      <form className="flex gap-2 border-t border-ai-border px-4 py-3" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <label htmlFor={`ask-${datasetId}`} className="sr-only">Ask about this diagnosis</label>
        <input id={`ask-${datasetId}`} className="field h-9 min-w-0 flex-1" value={input} onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about recall, fixes, drift, experiments…" maxLength={2000} />
        {busy
          ? <Button size="lg" className="h-9" icon={<Square size={12} />} onClick={() => abort.current?.abort()}>Stop</Button>
          : <Button size="lg" type="submit" variant="primary" className="h-9" icon={<ArrowUp size={15} />} disabled={!input.trim()}>Ask</Button>}
      </form>
    </section>
  );
}

function ProviderPicker({ providers, unknown, model, onPick, disabled, warm }: {
  providers: Providers | null; unknown: boolean; model: string; onPick: (m: string) => void; disabled: boolean; warm: Warm;
}) {
  if (!providers) return <Badge>{unknown ? "assistant status unknown" : "checking assistant…"}</Badge>;
  if (providers.provider === "ollama") {
    return (
      <div className="flex flex-col items-end gap-1">
        <label className="flex items-center gap-2 text-xs text-fg-2">
          Local model
          <select className="field" value={model} onChange={(e) => onPick(e.target.value)} disabled={disabled}>
            {providers.local_models.map((m) => (
              <option key={m.name} value={m.name}>{m.name}{m.parameters ? ` · ${m.parameters}` : ""}</option>
            ))}
          </select>
        </label>
        <span className="inline-flex items-center gap-1.5 text-xs text-fg-2" aria-live="polite">
          {warm === "ready" && <><span className="size-2 rounded-full bg-success" aria-hidden />ready</>}
          {warm === "loading" && <><Spinner /> loading model into memory…</>}
          {warm === "failed" && <><span className="size-2 rounded-full bg-warning" aria-hidden />not preloaded (first answer will be slower)</>}
        </span>
      </div>
    );
  }
  if (providers.provider === "anthropic") return <Badge>Claude API</Badge>;
  return (
    <Badge title={`Start Ollama (${providers.ollama_url}) to use a local model`}>rule-based answers · Ollama not running</Badge>
  );
}
