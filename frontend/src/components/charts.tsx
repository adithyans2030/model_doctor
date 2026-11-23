import { useRef, useState, type MouseEvent, type ReactNode } from "react";
import { cx } from "@/lib/cx";

/* Hand-built SVG/HTML charts. Conventions: thin marks with rounded data-ends, recessive
   grid, colours only from tokens, a legend for 2+ series, text in text tokens (never in
   series colour), hover tooltips, and for accessibility every chart is a labelled image
   (role="img") backed by a visually hidden data table. */

function useTip() {
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null);
  const show = (e: MouseEvent, html: ReactNode) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const x = Math.min(e.clientX - box.left + 12, box.width - 160);
    setTip({ x: Math.max(0, x), y: e.clientY - box.top + 14, html });
  };
  const node = tip && (
    <div aria-hidden className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-line bg-surface-3 px-2 py-1.5 text-xs text-fg shadow-md"
      style={{ left: tip.x, top: tip.y }}>{tip.html}</div>
  );
  return { ref, show, hide: () => setTip(null), node };
}

/** Visually hidden table with the chart's data, for screen readers. */
export function SrTable({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    // sr-only goes on a wrapper: tables ignore the 1px width and would widen the page.
    <div className="sr-only">
      <table>
        <caption>{caption}</caption>
        <thead><tr>{head.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => (j === 0 ? <th key={j} scope="row">{c}</th> : <td key={j}>{c}</td>))}</tr>)}</tbody>
      </table>
    </div>
  );
}

export function Legend({ items }: { items: [string, string][] }) {
  return (
    <div aria-hidden className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-2">
      {items.map(([label, color]) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <i className="inline-block h-[3px] w-3 rounded-sm" style={{ background: color }} />{label}
        </span>
      ))}
    </div>
  );
}

export interface BarItem { label: string; value: number; note?: string; tone?: "default" | "highlight" | "muted" | "danger" }

/** Horizontal bar list (magnitude, one series). Optional vertical reference lines. */
export function Bars({ items, label, format = (v) => v.toFixed(3), max, markers = [] }: {
  items: BarItem[];
  /** Accessible chart name. */
  label: string;
  format?: (v: number) => string;
  max?: number;
  /** Reference lines in data units, e.g. PSI thresholds. */
  markers?: { value: number; label: string }[];
}) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  const top = max ?? Math.max(...items.map((i) => Math.abs(i.value)), ...markers.map((m) => m.value), 1e-9);
  const fill = (tone: BarItem["tone"]) =>
    tone === "highlight" ? "bg-series-2" : tone === "muted" ? "bg-series-muted" : tone === "danger" ? "bg-div-neg" : "bg-series-1";
  return (
    <div className="relative" ref={tipRef} onMouseLeave={hide}>
      <div role="img" aria-label={label}>
        {markers.length > 0 && (
          <div aria-hidden className="grid grid-cols-[minmax(84px,34%)_1fr_60px] gap-2 pb-1">
            <span />
            <div className="relative h-4 text-[11px] text-fg-3">
              {markers.map((m) => (
                <span key={m.label} className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: `${(m.value / top) * 100}%` }}>{m.label}</span>
              ))}
            </div>
            <span />
          </div>
        )}
        {items.map((it) => (
          <div key={it.label} aria-hidden className="grid grid-cols-[minmax(84px,34%)_1fr_60px] items-center gap-2 py-[3px]"
            onMouseMove={(e) => show(e, <><b>{it.label}</b><br />{format(it.value)}{it.note && <><br /><span className="text-fg-3">{it.note}</span></>}</>)}>
            <span className="truncate text-sm text-fg-2">{it.label}</span>
            <div className="relative h-3 rounded-sm bg-surface-2">
              <div className={cx("h-full rounded-r-sm", fill(it.tone))}
                style={{ width: `${Math.min(1, Math.max(0, it.value / top)) * 100}%`, minWidth: it.value > 0 ? 2 : 0 }} />
              {markers.map((m) => (
                <span key={m.label} className="absolute -top-1 -bottom-1 w-px bg-fg-3/60" style={{ left: `${(m.value / top) * 100}%` }} />
              ))}
            </div>
            <span className="text-right font-mono text-xs text-fg">{format(it.value)}</span>
          </div>
        ))}
      </div>
      <SrTable caption={label} head={["Item", "Value", "Note"]} rows={items.map((i) => [i.label, format(i.value), i.note ?? ""])} />
      {tipNode}
    </div>
  );
}

/** Signed contributions (polarity) on the diverging pair around a zero line. */
export function DivergingBars({ items, label, posLabel, negLabel }: {
  items: { label: string; value: number; detail?: string }[];
  label: string;
  posLabel: string;
  negLabel: string;
}) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  const top = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div className="relative" ref={tipRef} onMouseLeave={hide}>
      <Legend items={[[posLabel, "var(--div-pos)"], [negLabel, "var(--div-neg)"]]} />
      <div role="img" aria-label={label}>
        {items.map((it) => {
          const w = (Math.abs(it.value) / top) * 50;
          const pos = it.value >= 0;
          return (
            <div key={it.label} aria-hidden className="grid grid-cols-[minmax(100px,38%)_1fr_60px] items-center gap-2 py-[3px]"
              onMouseMove={(e) => show(e, <><b>{it.label}</b>{it.detail ? ` = ${it.detail}` : ""}<br />{pos ? "+" : ""}{it.value.toFixed(4)} ({pos ? posLabel : negLabel})</>)}>
              <span className="truncate text-sm text-fg-2">{it.label}{it.detail && <span className="text-fg-3"> = {it.detail}</span>}</span>
              <div className="relative h-3">
                <div className="absolute -top-0.5 -bottom-0.5 left-1/2 w-px bg-fg-3" />
                <div className={pos ? "absolute h-full rounded-r-sm bg-div-pos" : "absolute h-full rounded-l-sm bg-div-neg"}
                  style={{ width: `${w}%`, left: pos ? "50%" : `${50 - w}%` }} />
              </div>
              <span className="text-right font-mono text-xs">{pos ? "+" : "−"}{Math.abs(it.value).toFixed(3)}</span>
            </div>
          );
        })}
      </div>
      <SrTable caption={label} head={["Feature", "Contribution", "Direction"]}
        rows={items.map((i) => [i.detail ? `${i.label} = ${i.detail}` : i.label, i.value.toFixed(4), i.value >= 0 ? posLabel : negLabel])} />
      {tipNode}
    </div>
  );
}

/** One horizontal bar split into labelled segments (e.g. column types). */
export function SegmentedBar({ segments, label }: { segments: { label: string; value: number; color: string }[]; label: string }) {
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  return (
    <div>
      <div role="img" aria-label={`${label}: ${segments.map((s) => `${s.value} ${s.label}`).join(", ")}`}
        className="flex h-2.5 gap-0.5 overflow-hidden rounded-full">
        {segments.map((s) => <div key={s.label} style={{ width: `${(s.value / total) * 100}%`, background: s.color }} />)}
      </div>
      <ul aria-hidden className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-2">
        {segments.map((s) => (
          <li key={s.label} className="inline-flex items-center gap-1.5">
            <i className="inline-block size-2 rounded-full" style={{ background: s.color }} />
            {s.label} <span className="font-mono text-fg">{s.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Before → after on a shared 0–1 scale, one row per metric (dumbbell). */
export function Dumbbell({ rows, label, beforeLabel, afterLabel, format = (v) => v.toFixed(3) }: {
  rows: { metric: string; before: number | null; after: number | null; better?: "before" | "after" | null }[];
  label: string;
  beforeLabel: string;
  afterLabel: string;
  format?: (v: number) => string;
}) {
  const pct = (v: number) => `${Math.min(1, Math.max(0, v)) * 100}%`;
  return (
    <div>
      <Legend items={[[beforeLabel, "var(--series-muted)"], [afterLabel, "var(--series-1)"]]} />
      <div role="img" aria-label={label} className="space-y-3">
        {rows.map((r) => (
          <div key={r.metric} aria-hidden className="grid grid-cols-[minmax(96px,28%)_1fr] items-center gap-3">
            <span className="truncate text-sm text-fg-2">{r.metric}</span>
            <div className="relative h-6">
              <div className="absolute inset-x-0 top-1/2 h-px bg-grid" />
              {[0, 0.25, 0.5, 0.75, 1].map((g) => <div key={g} className="absolute top-1 bottom-1 w-px bg-grid" style={{ left: pct(g) }} />)}
              {r.before != null && r.after != null && (
                <div className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-series-1/35"
                  style={{ left: pct(Math.min(r.before, r.after)), width: `calc(${pct(Math.max(r.before, r.after))} - ${pct(Math.min(r.before, r.after))})` }} />
              )}
              {r.before != null && (
                <span className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface bg-series-muted" style={{ left: pct(r.before) }} />
              )}
              {r.after != null && (
                <span className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface bg-series-1" style={{ left: pct(r.after) }} />
              )}
            </div>
          </div>
        ))}
        <div aria-hidden className="grid grid-cols-[minmax(96px,28%)_1fr] gap-3">
          <span />
          <div className="relative h-3 font-mono text-[11px] text-fg-3">
            {[0, 0.5, 1].map((g) => <span key={g} className="absolute -translate-x-1/2" style={{ left: pct(g) }}>{g}</span>)}
          </div>
        </div>
      </div>
      <SrTable caption={label} head={["Metric", beforeLabel, afterLabel]}
        rows={rows.map((r) => [r.metric, r.before == null ? "–" : format(r.before), r.after == null ? "–" : format(r.after)])} />
    </div>
  );
}

/** Confusion matrix: one-hue sequential scale on the row-normalised share. */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  return (
    <div className="relative overflow-x-auto" ref={tipRef} onMouseLeave={hide}>
      <table className="border-separate border-spacing-0.5 text-sm">
        <caption className="sr-only">Confusion matrix: rows are actual classes, columns are predicted classes</caption>
        <thead>
          <tr>
            <th scope="col" className="px-2 py-1 text-left text-xs font-medium text-fg-3">actual ↓ predicted →</th>
            {labels.map((l) => <th key={l} scope="col" className="px-2 py-1 text-right font-mono text-xs font-medium text-fg-3">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th scope="row" className="px-2 py-1 text-left font-mono text-xs font-medium text-fg-3">{labels[i]}</th>
                {row.map((v, j) => {
                  const share = v / total;
                  return (
                    <td key={j} className="h-12 min-w-20 rounded-sm px-3 text-right font-mono tabular-nums"
                      style={{ background: `color-mix(in oklab, var(--seq-1) ${Math.round(share * 85)}%, var(--seq-0))`, color: share > 0.5 ? "var(--surface-1)" : "var(--text-primary)" }}
                      onMouseMove={(e) => show(e, <>actual <b>{labels[i]}</b> → predicted <b>{labels[j]}</b><br />{v} rows ({(share * 100).toFixed(1)}% of actual {labels[i]})</>)}>
                      {v}
                      <span className="sr-only"> rows, {(share * 100).toFixed(1)}% of actual {labels[i]}</span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {tipNode}
    </div>
  );
}

/** Precision / recall / F1 vs decision threshold, with a crosshair tooltip. */
export function ThresholdChart({ curve, current, best }: {
  curve: { threshold: number; precision: number; recall: number; f1: number }[];
  current: number | null;
  best: number;
}) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = 220, P = { l: 36, r: 12, t: 10, b: 28 };
  const x = (v: number) => P.l + v * (W - P.l - P.r);
  const y = (v: number) => H - P.b - v * (H - P.t - P.b);
  const series = [
    { key: "precision" as const, label: "Precision", color: "var(--series-1)" },
    { key: "recall" as const, label: "Recall", color: "var(--series-2)" },
    { key: "f1" as const, label: "F1", color: "var(--series-3)" },
  ];
  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const v = (((e.clientX - box.left) / box.width) * W - P.l) / (W - P.l - P.r);
    const p = curve.reduce((a, b) => (Math.abs(b.threshold - v) < Math.abs(a.threshold - v) ? b : a));
    setHover(p.threshold);
    show(e, <><b>threshold {p.threshold.toFixed(2)}</b>{series.map((s) => (
      <div key={s.key}><i className="mr-1.5 inline-block h-[3px] w-2.5 align-middle" style={{ background: s.color }} />{s.label} {p[s.key].toFixed(3)}</div>
    ))}</>);
  };
  const last = curve[curve.length - 1];
  const hp = hover == null ? null : curve.find((c) => c.threshold === hover);
  const sample = curve.filter((_, i) => i % Math.max(1, Math.round(curve.length / 10)) === 0);
  return (
    <div className="relative" ref={tipRef}>
      <Legend items={series.map((s) => [s.label, s.color])} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`Precision, recall and F1 by decision threshold. Best F1 at threshold ${best.toFixed(2)}${current != null ? `; this model uses ${current}` : ""}.`}
        onMouseMove={onMove} onMouseLeave={() => { setHover(null); hide(); }}>
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={P.l} x2={W - P.r} y1={y(g)} y2={y(g)} stroke="var(--grid)" />
            <text x={P.l - 6} y={y(g) + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">{g}</text>
            <text x={x(g)} y={H - 12} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{g}</text>
          </g>
        ))}
        <line x1={x(best)} x2={x(best)} y1={P.t} y2={H - P.b} stroke="var(--text-muted)" strokeDasharray="3 3" />
        <text x={x(best) + 4} y={P.t + 10} fontSize="10" fill="var(--text-secondary)">best F1 {best.toFixed(2)}</text>
        {current != null && current !== best && <line x1={x(current)} x2={x(current)} y1={P.t} y2={H - P.b} stroke="var(--text-secondary)" />}
        {series.map((s) => (
          <g key={s.key}>
            <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
              points={curve.map((p) => `${x(p.threshold)},${y(p[s.key])}`).join(" ")} />
            <text x={x(last.threshold) - 2} y={y(last[s.key]) - 4} fontSize="10" textAnchor="end" fill="var(--text-secondary)">{s.label}</text>
          </g>
        ))}
        {hp && <line x1={x(hp.threshold)} x2={x(hp.threshold)} y1={P.t} y2={H - P.b} stroke="var(--text-secondary)" strokeOpacity="0.5" />}
        {hp && series.map((s) => <circle key={s.key} cx={x(hp.threshold)} cy={y(hp[s.key])} r="4" fill={s.color} stroke="var(--surface-1)" strokeWidth="2" />)}
        <text x={W / 2} y={H} fontSize="10" textAnchor="middle" fill="var(--text-muted)">decision threshold</text>
      </svg>
      <SrTable caption="Threshold curve (sampled)" head={["Threshold", "Precision", "Recall", "F1"]}
        rows={sample.map((p) => [p.threshold.toFixed(2), p.precision.toFixed(3), p.recall.toFixed(3), p.f1.toFixed(3)])} />
      {tipNode}
    </div>
  );
}

/** Compact histogram (counts per bin). */
export function Histogram({ edges, counts, label }: { edges: number[]; counts: number[]; label: string }) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  const top = Math.max(...counts, 1);
  return (
    <div className="relative" ref={tipRef} onMouseLeave={hide}>
      <div role="img" aria-label={`${label}, from ${edges[0].toPrecision(3)} to ${edges[edges.length - 1].toPrecision(3)}`}
        className="flex h-14 items-end gap-0.5">
        {counts.map((c, i) => (
          <div key={i} aria-hidden className="flex-1 rounded-t-[2px] bg-series-1" style={{ height: `${(c / top) * 100}%`, minHeight: c ? 2 : 0 }}
            onMouseMove={(e) => show(e, <>{edges[i].toPrecision(4)} – {edges[i + 1].toPrecision(4)}<br />{c} rows</>)} />
        ))}
      </div>
      <div aria-hidden className="mt-1 flex justify-between font-mono text-[11px] text-fg-3"><span>{edges[0].toPrecision(3)}</span><span>{edges[edges.length - 1].toPrecision(3)}</span></div>
      <SrTable caption={label} head={["Bin", "Rows"]} rows={counts.map((c, i) => [`${edges[i].toPrecision(4)} – ${edges[i + 1].toPrecision(4)}`, c])} />
      {tipNode}
    </div>
  );
}

/** Predicted vs actual (regression), with the y = x "perfect prediction" line. */
export function ScatterChart({ points }: { points: { actual: number; predicted: number }[] }) {
  const { ref: tipRef, show, hide, node: tipNode } = useTip();
  const W = 420, H = 300, P = { l: 56, r: 12, t: 10, b: 36 };
  const vals = points.flatMap((p) => [p.actual, p.predicted]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.04 || 1;
  const min = lo - pad, max = hi + pad;
  const x = (v: number) => P.l + ((v - min) / (max - min)) * (W - P.l - P.r);
  const y = (v: number) => H - P.b - ((v - min) / (max - min)) * (H - P.t - P.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => min + f * (max - min));
  const label = (v: number) => (Math.abs(v) >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : Math.abs(v) >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : v.toPrecision(3));
  return (
    <div className="relative" ref={tipRef} onMouseLeave={hide}>
      <Legend items={[["test rows", "var(--series-1)"], ["perfect prediction (y = x)", "var(--text-muted)"]]} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Predicted versus actual values">
        {ticks.map((v) => (
          <g key={v}>
            <line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
            <text x={P.l - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">{label(v)}</text>
            <text x={x(v)} y={H - P.b + 14} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{label(v)}</text>
          </g>
        ))}
        <line x1={x(min)} y1={y(min)} x2={x(max)} y2={y(max)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        {points.map((p, i) => (
          <circle key={i} cx={x(p.actual)} cy={y(p.predicted)} r="4" fill="var(--series-1)" fillOpacity="0.55"
            stroke="var(--surface-1)" strokeWidth="1"
            onMouseMove={(e) => show(e, <>actual <b>{p.actual.toLocaleString()}</b><br />predicted <b>{Math.round(p.predicted).toLocaleString()}</b><br />error {Math.round(p.predicted - p.actual).toLocaleString()}</>)} />
        ))}
        <text x={(P.l + W - P.r) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">actual</text>
        <text x={12} y={(P.t + H - P.b) / 2} textAnchor="middle" fontSize="10" fill="var(--text-muted)" transform={`rotate(-90 12 ${(P.t + H - P.b) / 2})`}>predicted</text>
      </svg>
      {tipNode}
    </div>
  );
}
