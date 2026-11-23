import { CheckCircle2, CircleX, GitCompareArrows, History, Minus, Plus, ShieldAlert, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { QueryView } from "@/components/domain/QueryView";
import { StageBadge } from "@/components/domain/severity";
import {
  Badge, Button, Card, CardSkeleton, ConfirmDialog, DeltaBadge, EmptyState, InlineError, TBody, TD, TH, THead, TR, Table,
  TableSkeleton, Tooltip, useToast,
} from "@/components/ui";
import { fmt } from "@/lib/api";
import { isAdmin, useAuth } from "@/lib/auth";
import { cx } from "@/lib/cx";
import { useComparison, useDatasetModels, useModelEvents, useMoveStage } from "@/lib/queries";
import type { ModelComparison, ModelDetail, ModelSummary, Stage } from "@/lib/types";

const NEXT: Record<Stage, [Stage, string][]> = {
  candidate: [["approved", "Approve"], ["rejected", "Reject"], ["archived", "Archive"]],
  approved: [["production", "Promote to production"], ["rejected", "Reject"], ["candidate", "Back to candidate"]],
  production: [["archived", "Archive"]],
  rejected: [["candidate", "Reopen"], ["archived", "Archive"]],
  archived: [["candidate", "Restore"]],
};
const GATED: Stage[] = ["approved", "production"];

/** Default reference: the production model; otherwise the original baseline (lowest version). */
function defaultReference(m: ModelDetail, siblings: ModelSummary[]): string | null {
  const others = siblings.filter((s) => s.id !== m.id);
  return (others.find((s) => s.stage === "production") ?? [...others].sort((a, b) => a.version - b.version)[0])?.id ?? null;
}

export function ApprovalTab({ m }: { m: ModelDetail }) {
  const siblings = useDatasetModels(m.dataset_id);
  const [chosen, setChosen] = useState<string | null>(null);
  const reference = chosen ?? (siblings.data ? defaultReference(m, siblings.data) : null);
  const comparison = useComparison(reference, m.id);
  const others = (siblings.data ?? []).filter((s) => s.id !== m.id);

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <Card title="Diff against the current model" description="What changes if this model replaces the reference: metrics, configuration and findings."
        actions={others.length > 0 && (
          <label className="flex min-w-0 max-w-full flex-wrap items-center gap-2 text-xs text-fg-3">
            Compare with
            <select className="field min-w-0 max-w-64" value={reference ?? ""} onChange={(e) => setChosen(e.target.value)}>
              {others.map((s) => <option key={s.id} value={s.id}>v{s.version} · {s.name}{s.stage === "production" ? " (production)" : ""}</option>)}
            </select>
          </label>
        )}>
        {siblings.isPending ? <TableSkeleton rows={6} cols={4} /> : !reference ? (
          <EmptyState compact icon={GitCompareArrows} title="Nothing to compare with" description="This is the only model for this dataset." />
        ) : (
          <QueryView query={comparison} label="Comparing models" skeleton={<TableSkeleton rows={8} cols={4} />}>
            {(c) => <DiffView c={c} />}
          </QueryView>
        )}
      </Card>
      <div className="space-y-5">
        <DecisionCard m={m} />
        <HistoryCard m={m} />
      </div>
    </div>
  );
}

function display(v: unknown): string {
  if (v === null || v === undefined) return "none";
  if (Array.isArray(v)) return v.length ? `[${v.join(", ")}]` : "[]";
  if (typeof v === "object") {
    const e = Object.entries(v as Record<string, unknown>);
    return e.length ? `{${e.map(([k, x]) => `${k}: ${typeof x === "number" ? +x.toPrecision(4) : String(x)}`).join(", ")}}` : "{}";
  }
  return typeof v === "number" ? String(+v.toPrecision(4)) : String(v);
}

function DiffView({ c }: { c: ModelComparison }) {
  const fmtMetric = (k: string, v: number | null) => (v == null ? "–" : k.startsWith("recall") ? fmt.pct(v, 1) : fmt.num(v));
  const config = Object.entries(c.config_diff);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="rounded-md bg-surface-2 px-2 py-1 font-mono text-xs">{c.a.name}</span>
        <StageBadge stage={c.a.stage} />
        <span aria-label="compared with" className="text-fg-3">→</span>
        <span className="rounded-md border border-accent/50 bg-accent-subtle px-2 py-1 font-mono text-xs">{c.b.name}</span>
        <StageBadge stage={c.b.stage} />
      </div>
      {!c.same_test_split && (
        <p className="flex items-start gap-1.5 rounded-md bg-warning-subtle px-3 py-2 text-sm">
          <TriangleAlert size={14} aria-hidden className="mt-[3px] shrink-0 text-warning" />
          These models were evaluated on different test splits, so the metric deltas are indicative only.
        </p>
      )}

      <section aria-labelledby="diff-metrics">
        <h3 id="diff-metrics" className="mb-2 text-xs font-medium text-fg-3">Metrics (held-out test set)</h3>
        <div className="-mx-4 border-y border-line">
          <Table label="Metric comparison" dense>
            <THead><TR><TH>Metric</TH><TH numeric>v{c.a.version}</TH><TH numeric>v{c.b.version}</TH><TH numeric>Change</TH></TR></THead>
            <TBody>
              {c.metrics.map((r) => (
                <TR key={r.metric}>
                  <TD className="font-mono text-xs">{r.metric}</TD>
                  <TD numeric className={r.better === "a" ? "text-fg" : "text-fg-3"}>{fmtMetric(r.metric, r.a)}</TD>
                  <TD numeric className={r.better === "b" ? "text-fg" : "text-fg-3"}>{fmtMetric(r.metric, r.b)}</TD>
                  <TD numeric>
                    {r.delta == null ? "–" : (
                      <DeltaBadge delta={{
                        value: r.delta,
                        // The API says which side is better; colour the change accordingly.
                        higherIsBetter: r.better === "b" ? r.delta > 0 : r.better === "a" ? r.delta < 0 : true,
                        format: r.metric.startsWith("recall") ? (v) => `${(v * 100).toFixed(1)} pts` : (v) => v.toFixed(3),
                        label: r.better === "b" ? `v${c.b.version} better` : r.better === "a" ? `v${c.a.version} better` : undefined,
                      }} />
                    )}
                  </TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </div>
      </section>

      <section aria-labelledby="diff-config">
        <h3 id="diff-config" className="mb-2 text-xs font-medium text-fg-3">Configuration</h3>
        {config.length === 0 ? <p className="text-sm text-fg-3">Same configuration.</p> : (
          <div className="overflow-x-auto rounded-md border border-line font-mono text-xs leading-6">
            {config.map(([k, v]) => (
              <div key={k}>
                <div className="flex gap-2 bg-danger-subtle px-3 text-fg">
                  <span aria-hidden className="select-none text-danger">−</span>
                  <span className="sr-only">v{c.a.version}:</span>
                  <span className="whitespace-pre">{k}: {display(v.a)}</span>
                </div>
                <div className="flex gap-2 bg-success-subtle px-3 text-fg">
                  <span aria-hidden className="select-none text-success">+</span>
                  <span className="sr-only">v{c.b.version}:</span>
                  <span className="whitespace-pre">{k}: {display(v.b)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section aria-labelledby="diff-findings">
        <h3 id="diff-findings" className="mb-2 text-xs font-medium text-fg-3">Findings</h3>
        <ul className="space-y-1.5 text-sm">
          {c.findings.resolved_in_b.map((f) => (
            <li key={`r-${f}`} className="flex items-center gap-2"><CheckCircle2 size={14} aria-hidden className="text-success" /><span className="sr-only">Resolved:</span><Badge mono>{f}</Badge><span className="text-xs text-fg-3">resolved</span></li>
          ))}
          {c.findings.new_in_b.map((f) => (
            <li key={`n-${f}`} className="flex items-center gap-2"><Plus size={14} aria-hidden className="text-danger" /><span className="sr-only">New:</span><Badge mono>{f}</Badge><span className="text-xs text-fg-3">new in v{c.b.version}</span></li>
          ))}
          {c.findings.shared.map((f) => (
            <li key={`s-${f}`} className="flex items-center gap-2"><Minus size={14} aria-hidden className="text-fg-3" /><span className="sr-only">Unchanged:</span><Badge mono>{f}</Badge><span className="text-xs text-fg-3">in both</span></li>
          ))}
          {!c.findings.resolved_in_b.length && !c.findings.new_in_b.length && !c.findings.shared.length && <li className="text-fg-3">Neither model has findings.</li>}
        </ul>
      </section>

      {c.summary.length > 0 && (
        <ul className="space-y-1 border-t border-line pt-4 text-xs text-fg-2">
          {c.summary.map((s) => <li key={s}>{s}</li>)}
        </ul>
      )}
    </div>
  );
}

function DecisionCard({ m }: { m: ModelDetail }) {
  const { user, status } = useAuth();
  const admin = isAdmin(user, status);
  const toast = useToast();
  const move = useMoveStage(m);
  const [note, setNote] = useState("");
  const [actor, setActor] = useState("");
  const [ack, setAck] = useState(false);
  const [pending, setPending] = useState<[Stage, string] | null>(null);
  const critical = m.findings.filter((f) => f.severity === "critical" && !f.resolved);
  const needsNote = (s: Stage) => ["approved", "rejected", "production"].includes(s);

  async function confirm() {
    if (!pending) return;
    const [stage, label] = pending;
    const ok = await move.mutateAsync({ stage, note: note.trim() || null, actor: actor.trim() || null, acknowledge_risks: ack }).then(() => true, () => false);
    if (ok) {
      toast({ title: `${label}: ${m.name}`, description: `Stage is now ${stage}.`, tone: "success" });
      setNote(""); setAck(false); setPending(null);
    }
  }

  return (
    <Card title="Approval gate" description="Experiments only produce candidates. A person approves, then promotes; promoting archives the previous production model.">
      <div className="space-y-4">
        <p className="flex items-center gap-2 text-sm">Current stage <StageBadge stage={m.stage} /></p>
        {critical.length > 0 && (
          <div className="rounded-md border border-danger/40 bg-danger-subtle px-3 py-2.5">
            <p className="flex items-center gap-2 text-sm font-medium"><ShieldAlert size={15} aria-hidden className="text-danger" />{critical.length} unresolved critical finding{critical.length > 1 ? "s" : ""}</p>
            <ul className="mt-1.5 list-disc space-y-1 pl-6 text-sm text-fg">{critical.map((f, i) => <li key={i}>{f.title}</li>)}</ul>
            <label className="mt-2.5 flex items-center gap-2 text-sm">
              <input type="checkbox" className="size-4 accent-[var(--accent)]" checked={ack} onChange={(e) => setAck(e.target.checked)} />
              I acknowledge these risks
            </label>
          </div>
        )}
        {!user && (
          <div>
            <label htmlFor="actor" className="mb-1 block text-xs font-medium text-fg-3">Your name</label>
            <input id="actor" className="field w-full" placeholder="Recorded in the audit trail" value={actor} onChange={(e) => setActor(e.target.value)} />
          </div>
        )}
        <div>
          <label htmlFor="stage-note" className="mb-1 block text-xs font-medium text-fg-3">Decision note <span className="font-normal">(required to approve, reject or promote)</span></label>
          <textarea id="stage-note" className="field w-full resize-y" rows={3} placeholder="Why is this the right call?" value={note} onChange={(e) => setNote(e.target.value)} />
        </div>
        <div className="flex flex-wrap gap-2">
          {NEXT[m.stage].map(([s, label]) => {
            const gated = GATED.includes(s) && !admin;
            const missingNote = needsNote(s) && !note.trim();
            // Gated buttons stay focusable (aria-disabled, not disabled) so the tooltip can explain why.
            const btn = (
              <Button key={s} variant={s === "rejected" ? "danger" : GATED.includes(s) ? "primary" : "secondary"}
                icon={s === "rejected" ? <CircleX size={14} /> : GATED.includes(s) ? <CheckCircle2 size={14} /> : undefined}
                disabled={!gated && missingNote} aria-disabled={gated || undefined}
                className={gated ? "cursor-not-allowed opacity-50" : undefined}
                onClick={() => { if (!gated) setPending([s, label]); }}>
                {label}
              </Button>
            );
            return gated ? <Tooltip key={s} content="Only admins can approve or promote models">{btn}</Tooltip> : btn;
          })}
        </div>
      </div>
      <ConfirmDialog open={!!pending} tone={pending?.[0] === "rejected" ? "danger" : "primary"} busy={move.isPending}
        title={pending ? `${pending[1]} ${m.name}?` : ""} confirmLabel={pending?.[1] ?? "Confirm"}
        description={pending?.[0] === "production" ? "It replaces the current production model, which is archived." : "This is recorded in the audit trail with your note."}
        onCancel={() => { setPending(null); move.reset(); }} onConfirm={confirm}>
        {note.trim() && <blockquote className="border-l-2 border-line-strong pl-3 text-sm text-fg-2">“{note.trim()}”</blockquote>}
        <InlineError error={move.error} className="mt-3" />
      </ConfirmDialog>
    </Card>
  );
}

function HistoryCard({ m }: { m: ModelDetail }) {
  const events = useModelEvents(m.id);
  return (
    <Card title="History">
      <QueryView query={events} label="Loading history" skeleton={<CardSkeleton title={false} lines={3} />}
        isEmpty={(e) => e.length === 0} empty={<EmptyState compact icon={History} title="No stage changes yet" />}>
        {(list) => (
          <ol className="relative space-y-4 border-l border-line pl-5">
            {list.map((e) => (
              <li key={e.id} className="relative">
                <span aria-hidden className={cx("absolute top-1.5 -left-[24.5px] size-2 rounded-full border-2 border-surface", e.to_stage === "production" ? "bg-success" : "bg-fg-3")} />
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  {e.from_stage && <><StageBadge stage={e.from_stage} /><span aria-label="to" className="text-fg-3">→</span></>}
                  <StageBadge stage={e.to_stage} />
                </div>
                <p className="mt-1 text-xs text-fg-3">{e.actor} · <time dateTime={e.created_at}>{fmt.date(e.created_at)}</time></p>
                {e.note && <p className="mt-1 text-sm text-fg-2">“{e.note}”</p>}
              </li>
            ))}
          </ol>
        )}
      </QueryView>
    </Card>
  );
}
