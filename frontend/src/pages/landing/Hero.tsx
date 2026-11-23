import { ArrowRight, FlaskConical, Info, OctagonAlert, TriangleAlert } from "lucide-react";
import { m, useReducedMotion } from "motion/react";
import { Link } from "react-router-dom";
import { GithubMark } from "@/components/GithubMark";
import { buttonClasses } from "@/components/ui";
import { cx } from "@/lib/cx";
import { DEMO_DIAGNOSIS, DEMO_RESULT, GITHUB } from "./content";

export function Hero() {
  return (
    <section aria-labelledby="hero-title" className="relative overflow-hidden">
      {/* A faint grid, masked to fade out: texture without a gradient blob. */}
      <div aria-hidden className="pointer-events-none absolute inset-0 [background-image:linear-gradient(var(--grid)_1px,transparent_1px),linear-gradient(90deg,var(--grid)_1px,transparent_1px)] [background-size:48px_48px] [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)] opacity-60" />
      <div className="relative mx-auto max-w-[1200px] px-4 pt-16 pb-12 sm:px-6 sm:pt-24 lg:pt-28">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-9">
            <p className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 text-xs text-fg-2">
              <span className="size-1.5 rounded-full bg-success" aria-hidden /> Open source · MIT · runs on your machine
            </p>
            <h1 id="hero-title" className="text-3xl font-semibold leading-[1.08] tracking-[-0.03em] text-balance sm:text-4xl lg:text-5xl">
              Find out why your model underperforms. Then prove the fix.
            </h1>
            <p className="mt-5 max-w-2xl text-md text-fg-2 sm:text-lg">
              ModelDoctor runs statistical checks on your dataset and trained models, tests targeted fixes with cross-validation,
              and sends the best candidate to a person for approval. The AI assistant only explains the results.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link to="/app" className={buttonClasses({ variant: "primary", size: "lg" })}>
                Try the demo <ArrowRight size={16} aria-hidden />
              </Link>
              <a href={GITHUB} target="_blank" rel="noreferrer" className={buttonClasses({ size: "lg" })}>
                <GithubMark size={16} /> View on GitHub
              </a>
            </div>
          </div>
        </div>
        <div className="mt-14 sm:mt-16">
          <DiagnosisMock />
        </div>
      </div>
    </section>
  );
}

const SEV = {
  critical: { icon: OctagonAlert, text: "text-danger", bg: "bg-danger-subtle", rail: "bg-danger", label: "Critical" },
  warning: { icon: TriangleAlert, text: "text-warning", bg: "bg-warning-subtle", rail: "bg-warning", label: "Warning" },
  info: { icon: Info, text: "text-info", bg: "bg-info-subtle", rail: "bg-line-strong", label: "Info" },
} as const;

/** A static rendering of ModelDoctor's diagnosis report on the demo dataset, animated in. */
function DiagnosisMock() {
  const reduce = useReducedMotion();
  const d = DEMO_DIAGNOSIS;
  const r = 38, c = 2 * Math.PI * r;
  const item = (i: number) => (reduce ? {} : {
    initial: { opacity: 0, y: 8 },
    animate: { opacity: 1, y: 0 },
    transition: { delay: 0.35 + i * 0.12, duration: 0.4, ease: [0.22, 1, 0.36, 1] as const },
  });
  return (
    <figure className="relative">
      <div className="overflow-hidden rounded-xl border border-line-strong bg-surface shadow-lg">
        <div className="flex items-center gap-3 border-b border-line bg-surface-2 px-4 py-2.5">
          <span aria-hidden className="flex gap-1.5"><i className="size-2.5 rounded-full bg-surface-3" /><i className="size-2.5 rounded-full bg-surface-3" /><i className="size-2.5 rounded-full bg-surface-3" /></span>
          <span className="truncate font-mono text-xs text-fg-3">churn_demo.csv · Diagnosis · Logistic Regression v1</span>
        </div>
        <div className="grid gap-px bg-line lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div className="bg-surface p-5">
            <p className="text-xs font-medium text-fg-3">Model health</p>
            <div className="mt-4 flex items-center gap-5">
              <div className="relative size-24 shrink-0">
                <svg viewBox="0 0 96 96" className="size-24 -rotate-90" aria-hidden>
                  <circle cx="48" cy="48" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="7" />
                  <m.circle cx="48" cy="48" r={r} fill="none" stroke="var(--warning)" strokeWidth="7" strokeLinecap="round"
                    strokeDasharray={c} initial={reduce ? false : { strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - d.health / 100) }}
                    style={reduce ? { strokeDashoffset: c * (1 - d.health / 100) } : undefined}
                    transition={{ duration: 1, ease: [0.22, 1, 0.36, 1], delay: 0.2 }} />
                </svg>
                <div className="absolute inset-0 grid place-items-center">
                  <span className="font-mono text-2xl font-medium">{d.health}<span className="sr-only"> out of 100</span></span>
                </div>
              </div>
              <ul className="space-y-1.5 text-xs" aria-label="Findings by severity">
                {(["critical", "warning", "info"] as const).map((s) => {
                  const Icon = SEV[s].icon;
                  return (
                    <li key={s} className={cx("inline-flex w-full items-center gap-1.5 rounded-sm px-1.5 py-0.5", SEV[s].bg, SEV[s].text)}>
                      <Icon size={12} aria-hidden /> {SEV[s].label} <span className="ml-auto pl-3 font-mono">{d.counts[s]}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
            <div className="mt-5 space-y-3">
              {d.components.map((comp) => (
                <div key={comp.name}>
                  <div className="flex justify-between text-xs"><span className="text-fg-2">{comp.name}</span><span className="font-mono">{comp.value}</span></div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-3">
                    <div className={cx("h-full rounded-full", comp.value >= 80 ? "bg-success" : "bg-danger")} style={{ width: `${comp.value}%` }} />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-fg-3">95 on performance, because the model leans on a leaking column.</p>
          </div>
          <div className="bg-surface p-5">
            <p className="text-xs font-medium text-fg-3">Issues, ranked by severity</p>
            <ol className="mt-3 space-y-2">
              {d.findings.map((f, i) => {
                const s = SEV[f.severity];
                const Icon = s.icon;
                return (
                  <m.li key={f.check} {...item(i)} className="relative overflow-hidden rounded-lg border border-line bg-surface py-2.5 pl-4 pr-3">
                    <span aria-hidden className={cx("absolute inset-y-0 left-0 w-[3px]", s.rail)} />
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className={cx("inline-flex items-center gap-1 rounded-sm px-1.5 text-xs font-medium", s.bg, s.text)}><Icon size={11} aria-hidden />{s.label}</span>
                      <span className="text-sm font-medium text-fg">{f.title}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-3 font-mono text-xs text-fg-3"><span>{f.check}</span><span>{f.evidence}</span></div>
                  </m.li>
                );
              })}
            </ol>
          </div>
        </div>
        <m.div {...item(d.findings.length + 1)} className="flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-line bg-surface-2 px-5 py-3.5">
          <span className="inline-flex items-center gap-2 text-sm font-medium"><FlaskConical size={15} aria-hidden className="text-accent-fg" /> After cross-validated fixes</span>
          {DEMO_RESULT.map((x) => (
            <span key={x.label} className="inline-flex items-baseline gap-2 text-sm">
              <span className="text-fg-3">{x.label}</span>
              <span className="font-mono text-fg-3">{x.before}</span><span aria-hidden className="text-fg-3">→</span><span className="sr-only">to</span>
              <span className="font-mono font-medium text-success">{x.after}</span>
            </span>
          ))}
          <span className="ml-auto rounded-sm border border-line px-1.5 py-0.5 text-xs text-fg-3">candidate v6 · awaiting approval</span>
        </m.div>
      </div>
      <figcaption className="mt-3 text-center text-xs text-fg-3">
        Diagnosis of a baseline trained on the synthetic demo dataset (sample_data/churn_demo.csv). Values are from a real run.
      </figcaption>
    </figure>
  );
}
