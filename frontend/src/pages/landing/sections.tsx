import {
  Activity, ArrowDown, ArrowRight, Bot, Check, Database, FlaskConical, GitBranch, ListChecks, Lock, Minus, ScanSearch, ShieldCheck, Stethoscope,
  type LucideIcon,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { GithubMark } from "@/components/GithubMark";
import { Logo } from "@/components/Logo";
import { CodeBlock, TabPanel, Tabs, buttonClasses } from "@/components/ui";
import { cx } from "@/lib/cx";
import { AUTHOR, DEMO_SHAP, DEMO_TRIALS, DOCS, GITHUB, LICENSE_URL, PROOF } from "./content";
import { Reveal } from "./Reveal";

const Container = ({ children, className }: { children: ReactNode; className?: string }) => (
  <div className={cx("mx-auto max-w-[1200px] px-4 sm:px-6", className)}>{children}</div>
);

function SectionHeading({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="max-w-2xl">
      <p className="font-mono text-xs uppercase tracking-[0.12em] text-accent-fg">{eyebrow}</p>
      <h2 id={id} className="mt-3 text-2xl font-semibold tracking-[-0.02em] text-balance sm:text-3xl">{title}</h2>
      {children && <p className="mt-4 text-md text-fg-2">{children}</p>}
    </div>
  );
}

/* 3. Proof strip */
export function ProofStrip() {
  return (
    <section aria-label="Project facts" className="border-y border-line bg-surface/60">
      <Container>
        <dl className="grid grid-cols-2 divide-line lg:grid-cols-4 lg:divide-x">
          {PROOF.map((p, i) => (
            <div key={p.label} className={cx("px-2 py-7 sm:px-6", i % 2 === 1 && "border-l border-line lg:border-l-0", i >= 2 && "border-t border-line lg:border-t-0")}>
              <dt className="sr-only">{p.label}</dt>
              <dd>
                <span className="block font-mono text-2xl font-medium tracking-tight sm:text-3xl">{p.value}</span>
                <span className="mt-1 block text-sm text-fg-2">{p.label}</span>
                <span className="mt-1 block text-xs text-fg-3">{p.source}</span>
              </dd>
            </div>
          ))}
        </dl>
      </Container>
    </section>
  );
}

/* 4. Problem */
const PROBLEMS = [
  { tag: "Leakage", title: "98% accurate, useless in production.", body: "The demo baseline leans on cancellation_request, a column that only exists after a customer has already left." },
  { tag: "Imbalance", title: "90% accuracy, 5% recall.", body: "Remove the leak and the model almost never predicts churn. Accuracy hides it, because 90% of customers stay." },
  { tag: "Drift", title: "The data moves after you ship.", body: "Feature distributions and the prediction mix shift over time, and nobody looks until a business metric drops." },
];

export function Problem() {
  return (
    <section aria-labelledby="problem-title" className="py-24 sm:py-32">
      <Container>
        <Reveal><SectionHeading id="problem-title" eyebrow="The problem" title="Models fail quietly." /></Reveal>
        <ol className="mt-14 grid gap-10 md:grid-cols-3 md:gap-0 md:divide-x md:divide-line">
          {PROBLEMS.map((p, i) => (
            <li key={p.tag} className="md:px-8 md:first:pl-0 md:last:pr-0">
              <Reveal delay={i * 0.08}>
                <span className="font-mono text-sm text-fg-3">0{i + 1} · {p.tag}</span>
                <h3 className="mt-3 text-lg font-semibold tracking-tight">{p.title}</h3>
                <p className="mt-2 text-fg-2">{p.body}</p>
              </Reveal>
            </li>
          ))}
        </ol>
      </Container>
    </section>
  );
}

/* 5. How it works */
const STEPS: { icon: LucideIcon; title: string; body: string; out: string }[] = [
  { icon: Stethoscope, title: "Diagnose", body: "Statistical checks on the data, the model and its predictions produce findings with evidence.", out: "findings[]" },
  { icon: ListChecks, title: "Recommend", body: "Each finding names fixes from a fixed catalogue: class weights, SMOTE, threshold tuning, feature drops.", out: "plan" },
  { icon: FlaskConical, title: "Experiment", body: "Fixes are applied one at a time. A step is kept only if the cross-validated score improves.", out: "trials[]" },
  { icon: ShieldCheck, title: "Approve", body: "The winner becomes a candidate. A person approves it, acknowledging any critical findings.", out: "stage: approved" },
  { icon: Activity, title: "Monitor", body: "Production batches are checked for feature, prediction and concept drift.", out: "PSI per feature" },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" aria-labelledby="how-title" className="scroll-mt-16 border-t border-line bg-surface/40 py-24 sm:py-32">
      <Container>
        <Reveal><SectionHeading id="how-title" eyebrow="How it works" title="From a CSV to an approved model, in five steps." /></Reveal>
        <ol className="relative mt-14 grid gap-4 lg:grid-cols-5 lg:gap-3">
          {STEPS.map((s, i) => (
            <li key={s.title} className="relative">
              <Reveal delay={i * 0.06} className="relative flex h-full gap-4 rounded-lg border border-line bg-surface p-5 lg:flex-col lg:gap-0">
                <div className="flex items-center gap-3 lg:mb-4">
                  <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-line bg-surface-2 text-accent-fg"><s.icon size={17} aria-hidden /></span>
                  <span className="hidden font-mono text-xs text-fg-3 lg:inline">0{i + 1}</span>
                </div>
                <div className="min-w-0">
                  <h3 className="text-base font-semibold"><span className="font-mono text-xs font-normal text-fg-3 lg:hidden">0{i + 1} </span>{s.title}</h3>
                  <p className="mt-1.5 text-sm text-fg-2">{s.body}</p>
                  <p className="mt-3 inline-block rounded-sm bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-fg-3">→ {s.out}</p>
                </div>
                {i < STEPS.length - 1 && (
                  <span aria-hidden className="absolute top-1/2 -right-[11px] z-10 hidden size-5 -translate-y-1/2 place-items-center rounded-full border border-line bg-bg text-fg-3 lg:grid">
                    <ArrowRight size={11} />
                  </span>
                )}
              </Reveal>
            </li>
          ))}
        </ol>
      </Container>
    </section>
  );
}

/* 6. Feature grid (bento: varied spans, not repeated equal rows) */
function Feature({ icon: Icon, title, body, children, className }: { icon: LucideIcon; title: string; body: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <Reveal className={cx("col-span-12", className)}>
      <article className="flex h-full flex-col rounded-xl border border-line bg-surface p-6">
        <Icon size={18} aria-hidden className="text-accent-fg" />
        <h3 className="mt-4 text-base font-semibold">{title}</h3>
        <p className="mt-1.5 text-sm text-fg-2">{body}</p>
        {children && <div className="mt-6 flex-1">{children}</div>}
      </article>
    </Reveal>
  );
}

export function Features() {
  const top = DEMO_SHAP[0].value;
  return (
    <section id="features" aria-labelledby="features-title" className="scroll-mt-16 py-24 sm:py-32">
      <Container>
        <Reveal><SectionHeading id="features-title" eyebrow="Features" title="Everything between “the metric looks off” and “this model is approved”." /></Reveal>
        <div className="mt-14 grid grid-cols-12 gap-4">
          <Feature className="md:col-span-7" icon={ScanSearch} title="Explainability you can check"
            body="TreeSHAP for tree models, exact SHAP for linear ones, permutation importance, and LIME or occlusion for single predictions. Error slices and group gaps show where the model fails.">
            <figure>
              <figcaption className="mb-2.5 text-xs text-fg-3">Mean |SHAP| on the demo baseline: one column carries the model.</figcaption>
              <div role="img" aria-label={`Mean absolute SHAP values: ${DEMO_SHAP.map((s) => `${s.feature} ${s.value}`).join(", ")}`} className="space-y-1.5">
                {DEMO_SHAP.map((s, i) => (
                  <div key={s.feature} aria-hidden className="grid grid-cols-[9.5rem_1fr_3rem] items-center gap-3 text-xs">
                    <span className="truncate font-mono text-fg-2">{s.feature}</span>
                    <span className="h-2 rounded-sm bg-surface-2"><span className={cx("block h-full rounded-sm", i === 0 ? "bg-div-neg" : "bg-series-1")} style={{ width: `${(s.value / top) * 100}%` }} /></span>
                    <span className="text-right font-mono text-fg-3">{s.value.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </figure>
          </Feature>
          <Feature className="md:col-span-5" icon={FlaskConical} title="Cross-validated experiments"
            body="A fix is kept only if it improves the cross-validated score (macro-F1 or R²) on the training split. The test set is used only to report results.">
            <ol aria-label="Demo experiment trials" className="space-y-1.5">
              {DEMO_TRIALS.map((t) => (
                <li key={t.name} className="flex items-center gap-2 text-xs">
                  <span className={cx("grid size-4 place-items-center rounded-full", t.kept ? "bg-success-subtle text-success" : "bg-surface-2 text-fg-3")}>
                    {t.kept ? <Check size={10} aria-hidden /> : <Minus size={10} aria-hidden />}
                  </span>
                  <span className={t.kept ? "text-fg" : "text-fg-3 line-through decoration-fg-3/50"}>{t.name}</span>
                  <span className="sr-only">{t.kept ? "kept" : "rejected"}</span>
                </li>
              ))}
            </ol>
          </Feature>
          <Feature className="md:col-span-5" icon={ShieldCheck} title="A human approval gate"
            body="Experiments only create candidates. Moving to approved or production needs a note and an explicit acknowledgement of critical findings; with sign-in on, it also needs an admin. Every change goes in the audit trail.">
            <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
              {["candidate", "approved", "production"].map((s, i) => (
                <span key={s} className="inline-flex items-center gap-2">
                  {i > 0 && <ArrowRight size={12} aria-hidden className="text-fg-3" />}
                  <span className={cx("rounded-sm border px-2 py-1", s === "production" ? "border-success/40 bg-success-subtle text-success" : "border-line bg-surface-2 text-fg-2")}>{s}</span>
                </span>
              ))}
            </div>
          </Feature>
          <Feature className="md:col-span-7" icon={GitBranch} title="Model registry, with an optional MLflow mirror"
            body="Every trained or optimised model is versioned with its config, metrics and findings. Set MD_MLFLOW_TRACKING_URI and runs are mirrored to MLflow; datasets can be versioned with DVC." />
          <Feature className="md:col-span-6" icon={Activity} title="Drift monitoring"
            body="Upload a production batch to get PSI plus KS or chi-square tests per feature, prediction-mix drift and, when labels are included, concept drift. Metrics are exported to Prometheus with a Grafana dashboard." />
          <Feature className="md:col-span-6" icon={Bot} title="A local AI assistant that cites its sources"
            body="Uses your Ollama models by default, so nothing leaves your machine. Answers name the finding ids they rely on, and fall back to rule-based answers when no model is running." />
        </div>
      </Container>
    </section>
  );
}

/* 7. Trust: deterministic diagnosis, AI explanation only */
function Node({ children, tone = "default", className }: { children: ReactNode; tone?: "default" | "accent" | "ai"; className?: string }) {
  return (
    <div className={cx(
      "rounded-lg border px-3.5 py-2.5 text-sm",
      tone === "accent" ? "border-accent/50 bg-accent-subtle" : tone === "ai" ? "border-dashed border-ai-border bg-ai-subtle" : "border-line bg-surface",
      className,
    )}>
      {children}
    </div>
  );
}

function FlowDiagram() {
  return (
    <figure aria-labelledby="flow-caption" className="rounded-xl border border-line bg-surface-2/50 p-5 sm:p-6">
      <div className="flex flex-col items-stretch gap-2">
        <Node><span className="flex items-center gap-2"><Database size={14} aria-hidden className="text-fg-3" />Dataset + trained models</span></Node>
        <ArrowDown size={14} aria-hidden className="mx-auto text-fg-3" />
        <Node>
          <span className="font-medium">Deterministic checks</span>
          <span className="mt-0.5 block text-xs text-fg-3">pandas · scikit-learn · SciPy · SHAP: data, model, predictions, drift</span>
        </Node>
        <ArrowDown size={14} aria-hidden className="mx-auto text-fg-3" />
        <Node tone="accent">
          <span className="font-medium">Findings</span>
          <span className="mt-0.5 block font-mono text-xs text-fg-2">check · severity · evidence · cause · fixes</span>
        </Node>
        <div className="grid grid-cols-1 gap-3 pt-2 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <span className="text-center font-mono text-[11px] text-fg-3">drives</span>
            <Node>Optimiser, registry, dashboard</Node>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-center font-mono text-[11px] text-ai">read-only</span>
            <Node tone="ai">
              <span className="flex items-center gap-2 font-medium"><Bot size={14} aria-hidden className="text-ai" />AI assistant</span>
              <span className="mt-0.5 block text-xs text-fg-2">explains, cites finding ids</span>
            </Node>
          </div>
        </div>
      </div>
      <figcaption id="flow-caption" className="mt-4 flex items-start gap-1.5 text-xs text-fg-3">
        <Lock size={12} aria-hidden className="mt-0.5 shrink-0" />
        Data flow: findings are computed first and drive every decision. The assistant receives them as input and has no path back to the findings, the optimiser or the registry.
      </figcaption>
    </figure>
  );
}

export function Trust() {
  return (
    <section aria-labelledby="trust-title" className="border-t border-line bg-surface/40 py-24 sm:py-32">
      <Container>
        <div className="grid grid-cols-12 items-center gap-y-12 lg:gap-x-16">
          <Reveal className="col-span-12 lg:col-span-6">
            <SectionHeading id="trust-title" eyebrow="Design principle" title="Deterministic diagnosis. AI explanation only." />
            <ul className="mt-8 space-y-5">
              {[
                ["Every finding is a statistic you can inspect.", "The evidence (AUC, ratios, p-values, counts) is shown next to each finding, and each score lists its formula."],
                ["The assistant explains; it doesn't decide.", "It gets the structured results, must cite finding ids, and its citations are checked against what it was given."],
                ["It still works without an LLM.", "With no Ollama model running, answers come from rules built on the same results."],
              ].map(([t, b]) => (
                <li key={t} className="flex gap-3">
                  <Check size={16} aria-hidden className="mt-1 shrink-0 text-success" />
                  <div><p className="font-medium">{t}</p><p className="mt-1 text-sm text-fg-2">{b}</p></div>
                </li>
              ))}
            </ul>
          </Reveal>
          <Reveal className="col-span-12 lg:col-span-6" delay={0.1}><FlowDiagram /></Reveal>
        </div>
      </Container>
    </section>
  );
}

/* 8. Demo walkthrough: tabbed screenshots */
const SHOTS = [
  { value: "diagnosis", label: "Diagnosis", src: "/landing/diagnosis.webp", alt: "Diagnosis report: model health 60 out of 100 with sub-scores, a narrative summary, and issues ranked by severity.", caption: "Health from transparent sub-scores, issues ranked by severity, and a fix plan linked to the findings." },
  { value: "experiments", label: "Experiments", src: "/landing/experiments.webp", alt: "Experiments view: churn recall 5% to 49%, macro-F1 0.521 to 0.647, and a table of trials marked kept or rejected.", caption: "Leak-free baseline vs the best configuration, and every trial with the reason it was kept or rejected." },
  { value: "approval", label: "Approval", src: "/landing/approval.webp", alt: "Approval gate: metric deltas, a configuration diff and two unresolved critical findings to acknowledge before approving.", caption: "A diff against the production model, and an approval that needs a note and risk acknowledgement." },
  { value: "monitoring", label: "Monitoring", src: "/landing/monitoring.webp", alt: "Monitoring view: drift score 92 out of 100, PSI per feature with moderate and major thresholds, and batch history.", caption: "Drift score over time, PSI per feature against the thresholds, and batch-level performance." },
] as const;
type Shot = (typeof SHOTS)[number]["value"];

export function DemoWalkthrough() {
  const [tab, setTab] = useState<Shot>("diagnosis");
  const shot = SHOTS.find((s) => s.value === tab)!;
  return (
    <section id="demo" aria-labelledby="demo-title" className="scroll-mt-16 py-24 sm:py-32">
      <Container>
        <Reveal>
          <SectionHeading id="demo-title" eyebrow="Demo" title="What it looks like on the demo dataset.">
            Screenshots from the dashboard after running the full workflow on sample_data/churn_demo.csv, a synthetic customer-churn dataset with 9 planted problems.
          </SectionHeading>
        </Reveal>
        <Reveal className="mt-10">
          <Tabs label="Dashboard screenshots" idBase="demo" value={tab} onChange={setTab} items={SHOTS.map(({ value, label }) => ({ value, label }))} />
          <TabPanel idBase="demo" value={tab} className="pt-6">
            <figure>
              <div className="overflow-hidden rounded-xl border border-line-strong bg-surface shadow-lg">
                <img src={shot.src} alt={shot.alt} width={1920} height={1200} loading="lazy" decoding="async" className="block h-auto w-full" />
              </div>
              <figcaption className="mt-3 text-sm text-fg-2">{shot.caption}</figcaption>
            </figure>
          </TabPanel>
        </Reveal>
      </Container>
    </section>
  );
}

/* 9. Comparison */
type Cell = { v: "yes" | "partial" | "no"; t: string };
const COMPARE: { row: string; cells: [Cell, Cell, Cell, Cell] }[] = [
  { row: "Explains why a model underperforms", cells: [{ v: "yes", t: "Findings with evidence and likely cause" }, { v: "no", t: "Optimises a score" }, { v: "no", t: "Not its job" }, { v: "no", t: "Shows the metric" }] },
  { row: "Tests fixes", cells: [{ v: "yes", t: "Targeted fixes from the diagnosis, kept by CV" }, { v: "partial", t: "Broad model and hyperparameter search" }, { v: "no", t: "Tracks the runs you do" }, { v: "no", t: "Not covered" }] },
  { row: "Experiment tracking and registry", cells: [{ v: "yes", t: "Built in; optional MLflow mirror" }, { v: "partial", t: "Varies by product" }, { v: "yes", t: "Its core strength" }, { v: "no", t: "Not covered" }] },
  { row: "Human approval before production", cells: [{ v: "yes", t: "Required note and risk acknowledgement; admin role when sign-in is on" }, { v: "no", t: "Usually not built in" }, { v: "partial", t: "Aliases and tags; sign-off is up to you" }, { v: "no", t: "Not covered" }] },
  { row: "Per-prediction explanations", cells: [{ v: "yes", t: "SHAP, LIME, occlusion" }, { v: "partial", t: "Some products" }, { v: "partial", t: "Logs artefacts you produce" }, { v: "partial", t: "If someone builds it" }] },
  { row: "Drift monitoring", cells: [{ v: "yes", t: "PSI, KS, chi-square, concept drift" }, { v: "partial", t: "Some products" }, { v: "no", t: "Not built in" }, { v: "partial", t: "Usually custom" }] },
  { row: "Maturity and scale", cells: [{ v: "partial", t: "Young project, single node, tabular data" }, { v: "yes", t: "Mature, managed options" }, { v: "yes", t: "Mature, widely adopted" }, { v: "yes", t: "Mature tooling" }] },
];

const CellIcon = ({ v }: { v: Cell["v"] }) => v === "yes"
  ? <Check size={14} aria-hidden className="mt-0.5 shrink-0 text-success" />
  : v === "partial" ? <span aria-hidden className="mt-1.5 size-2 shrink-0 rounded-full border border-warning" />
  : <Minus size={14} aria-hidden className="mt-0.5 shrink-0 text-fg-3" />;
const CELL_LABEL = { yes: "Yes", partial: "Partly", no: "No" } as const;

export function Comparison() {
  const heads = ["ModelDoctor", "AutoML tools", "MLflow", "Plain dashboards"];
  return (
    <section aria-labelledby="compare-title" className="border-t border-line bg-surface/40 py-24 sm:py-32">
      <Container>
        <Reveal>
          <SectionHeading id="compare-title" eyebrow="Comparison" title="Where it fits.">
            ModelDoctor is not a replacement for AutoML or MLflow. It uses MLflow as infrastructure and treats model search as one step of a diagnosis.
          </SectionHeading>
        </Reveal>
        <Reveal className="mt-10">
          <div role="region" aria-label="Comparison table" tabIndex={0} className="relative overflow-x-auto rounded-xl border border-line bg-surface">
            <table className="w-full min-w-[760px] border-collapse text-sm">
              <caption className="sr-only">ModelDoctor compared with AutoML tools, MLflow and plain dashboards</caption>
              <thead>
                <tr>
                  <th scope="col" className="w-[22%] border-b border-line px-4 py-3 text-left text-xs font-medium text-fg-3"><span className="sr-only">Capability</span></th>
                  {heads.map((h, i) => (
                    <th key={h} scope="col" className={cx("border-b border-line px-4 py-3 text-left text-sm font-semibold", i === 0 && "bg-accent-subtle text-fg")}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {COMPARE.map((r) => (
                  <tr key={r.row} className="border-b border-line last:border-b-0">
                    <th scope="row" className="px-4 py-3 text-left align-top font-medium">{r.row}</th>
                    {r.cells.map((c, i) => (
                      <td key={i} className={cx("px-4 py-3 align-top text-fg-2", i === 0 && "bg-accent-subtle/60 text-fg")}>
                        <span className="flex gap-2"><CellIcon v={c.v} /><span><span className="sr-only">{CELL_LABEL[c.v]}: </span>{c.t}</span></span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-fg-3">General characterisation; individual AutoML and dashboard products vary.</p>
        </Reveal>
      </Container>
    </section>
  );
}

/* 10. Tech stack */
const STACK = ["React 19", "TypeScript", "Tailwind CSS 4", "TanStack Query", "FastAPI", "Pydantic v2", "SQLAlchemy", "scikit-learn", "XGBoost", "LightGBM", "SHAP", "MLflow", "DVC", "Postgres", "Prometheus", "Grafana", "Docker", "Kubernetes", "Ollama"];

export function TechStack() {
  return (
    <section aria-labelledby="stack-title" className="py-20">
      <Container>
        <h2 id="stack-title" className="text-center text-sm font-medium text-fg-3">Built with</h2>
        <ul className="mx-auto mt-5 flex max-w-4xl flex-wrap justify-center gap-2">
          {STACK.map((s) => <li key={s} className="rounded-md border border-line bg-surface px-2.5 py-1 font-mono text-xs text-fg-2">{s}</li>)}
        </ul>
      </Container>
    </section>
  );
}

/* 11. Quick start */
const QUICKSTART = `git clone ${GITHUB}.git
cd model_doctor
cp .env.example .env
docker compose up --build
# Dashboard  http://localhost:3000
# API docs   http://localhost:8000/docs`;

export function QuickStart() {
  return (
    <section aria-labelledby="quickstart-title" className="border-t border-line py-24 sm:py-32">
      <Container>
        <div className="grid grid-cols-12 items-start gap-y-10 lg:gap-x-16">
          <Reveal className="col-span-12 lg:col-span-5">
            <SectionHeading id="quickstart-title" eyebrow="Quick start" title="Run it locally in one command.">
              Docker Compose starts the API, the dashboard, Postgres, MLflow, Prometheus and Grafana. Then click “Load demo dataset”.
            </SectionHeading>
            <p className="mt-4 text-sm text-fg-3">
              Prefer no Docker? The <a href={`${GITHUB}#quick-start-local`} target="_blank" rel="noreferrer">README</a> has a two-terminal setup with uvicorn and Vite.
            </p>
          </Reveal>
          <Reveal className="col-span-12 lg:col-span-7" delay={0.08}>
            <CodeBlock label="Terminal" code={QUICKSTART} />
          </Reveal>
        </div>
      </Container>
    </section>
  );
}

/* 12. Final CTA + footer */
export function FinalCta() {
  return (
    <section aria-labelledby="cta-title" className="border-t border-line bg-surface/40 py-24">
      <Container className="text-center">
        <Reveal>
          <h2 id="cta-title" className="mx-auto max-w-2xl text-2xl font-semibold tracking-[-0.02em] text-balance sm:text-3xl">See the whole workflow on the demo dataset.</h2>
          <p className="mx-auto mt-4 max-w-xl text-fg-2">Diagnose, test fixes, approve the winner. It takes a few minutes on a laptop.</p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link to="/app" className={buttonClasses({ variant: "primary", size: "lg" })}>Try the demo <ArrowRight size={16} aria-hidden /></Link>
            <a href={GITHUB} target="_blank" rel="noreferrer" className={buttonClasses({ size: "lg" })}><GithubMark size={16} /> View on GitHub</a>
          </div>
        </Reveal>
      </Container>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-line">
      <Container className="flex flex-col gap-6 py-10 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Logo />
          <p className="mt-2 text-sm text-fg-3">Built by {AUTHOR}.</p>
        </div>
        <nav aria-label="Footer">
          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <li><a href={GITHUB} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-fg">GitHub</a></li>
            <li><a href={DOCS} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-fg">Docs</a></li>
            <li><a href={LICENSE_URL} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-fg">MIT License</a></li>
            <li><Link to="/app" className="text-fg-2 hover:text-fg">Dashboard</Link></li>
          </ul>
        </nav>
      </Container>
    </footer>
  );
}
