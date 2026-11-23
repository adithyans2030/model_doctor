import { useRef, useState, type MouseEvent, type ReactNode } from "react";

/* Small hand-built SVG/HTML charts. Conventions (dataviz guidelines): thin marks with
   rounded data-ends, recessive grid, hover tooltips on every mark, a legend for 2+
   series, text in text tokens (never in series colour), and tables alongside. */

function useTip() {
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null);
  const show = (e: MouseEvent, html: ReactNode) => {
    const box = ref.current?.getBoundingClientRect();
    if (box) setTip({ x: e.clientX - box.left + 12, y: e.clientY - box.top + 12, html });
  };
  const node = tip && (
    <div className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-line bg-surface px-2 py-1.5 text-xs text-fg shadow-lg"
      style={{ left: tip.x, top: tip.y }}>{tip.html}</div>
  );
  return { ref, show, hide: () => setTip(null), node };
}

const rowGrid = "grid grid-cols-[minmax(90px,34%)_1fr_64px] items-center gap-2 py-[3px]";

/** Horizontal bar list (magnitude, one series). */
export function Bars({ items, format = (v) => v.toFixed(3), max }: {
  items: { label: string; value: number; note?: string }[];
  format?: (v: number) => string;
  max?: number;
}) {
  const t = useTip();
  const top = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div className="relative" ref={t.ref} onMouseLeave={t.hide}>
      {items.map((it) => (
        <div key={it.label} className={rowGrid}
          onMouseMove={(e) => t.show(e, <><b>{it.label}</b><br />{format(it.value)}{it.note && <><br /><span className="text-fg-3">{it.note}</span></>}</>)}>
          <span className="truncate text-[13px] text-fg-2">{it.label}</span>
          <div className="h-3.5 rounded bg-surface-2">
            <div className="h-full rounded-r bg-series-1" style={{ width: `${Math.max(0, it.value / top) * 100}%`, minWidth: it.value > 0 ? 2 : 0 }} />
          </div>
          <span className="text-right font-mono text-xs">{format(it.value)}</span>
        </div>
      ))}
      {t.node}
    </div>
  );
}

/** Signed contributions (polarity) on the diverging blue/red pair around a zero line. */
export function DivergingBars({ items, posLabel, negLabel }: {
  items: { label: string; value: number; detail?: string }[];
  posLabel: string;
  negLabel: string;
}) {
  const t = useTip();
  const top = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
  return (
    <div className="relative" ref={t.ref} onMouseLeave={t.hide}>
      <Legend items={[[posLabel, "var(--div-pos)"], [negLabel, "var(--div-neg)"]]} />
      {items.map((it) => {
        const w = (Math.abs(it.value) / top) * 50;
        const pos = it.value >= 0;
        return (
          <div key={it.label} className="grid grid-cols-[minmax(110px,36%)_1fr_64px] items-center gap-2 py-[3px]"
            onMouseMove={(e) => t.show(e, <><b>{it.label}</b>{it.detail ? ` = ${it.detail}` : ""}<br />{pos ? "+" : ""}{it.value.toFixed(4)} ({pos ? posLabel : negLabel})</>)}>
            <span className="truncate text-[13px] text-fg-2">{it.label}{it.detail && <span className="text-fg-3"> = {it.detail}</span>}</span>
            <div className="relative h-3.5">
              <div className="absolute -top-0.5 -bottom-0.5 left-1/2 w-px bg-fg-3" />
              <div className={pos ? "absolute h-full rounded-r bg-div-pos" : "absolute h-full rounded-l bg-div-neg"}
                style={{ width: `${w}%`, left: pos ? "50%" : `${50 - w}%` }} />
            </div>
            <span className="text-right font-mono text-xs">{pos ? "+" : ""}{it.value.toFixed(3)}</span>
          </div>
        );
      })}
      {t.node}
    </div>
  );
}

export function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="mb-1.5 flex flex-wrap gap-3.5 text-xs text-fg-2">
      {items.map(([label, color]) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <i className="inline-block h-[3px] w-3 rounded-sm" style={{ background: color }} />{label}
        </span>
      ))}
    </div>
  );
}

/** Confusion matrix: one-hue sequential scale on the row-normalised share. */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const t = useTip();
  return (
    <div className="relative overflow-x-auto" ref={t.ref} onMouseLeave={t.hide}>
      <table className="tbl w-auto">
        <thead><tr><th>actual \ predicted</th>{labels.map((l) => <th key={l} className="num">{l}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th>{labels[i]}</th>
                {row.map((v, j) => {
                  const share = v / total;
                  return (
                    <td key={j} className="num min-w-16 border-2! border-surface!"
                      style={{ background: `color-mix(in oklab, var(--seq-1) ${Math.round(share * 85)}%, var(--seq-0))`, color: share > 0.5 ? "#fff" : "var(--text-primary)" }}
                      onMouseMove={(e) => t.show(e, <>actual <b>{labels[i]}</b> → predicted <b>{labels[j]}</b><br />{v} rows ({(share * 100).toFixed(1)}% of actual {labels[i]})</>)}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {t.node}
    </div>
  );
}

/** Precision / recall / F1 vs decision threshold, with a crosshair tooltip. */
export function ThresholdChart({ curve, current, best }: {
  curve: { threshold: number; precision: number; recall: number; f1: number }[];
  current: number | null;
  best: number;
}) {
  const t = useTip();
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
    t.show(e, <><b>threshold {p.threshold.toFixed(2)}</b>{series.map((s) => (
      <div key={s.key}><i className="mr-1.5 inline-block h-[3px] w-2.5 align-middle" style={{ background: s.color }} />{s.label} {p[s.key].toFixed(3)}</div>
    ))}</>);
  };
  const last = curve[curve.length - 1];
  const hp = hover == null ? null : curve.find((c) => c.threshold === hover)!;
  return (
    <div className="relative" ref={t.ref}>
      <Legend items={series.map((s) => [s.label, s.color])} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Precision, recall and F1 by decision threshold"
        onMouseMove={onMove} onMouseLeave={() => { setHover(null); t.hide(); }}>
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
      {t.node}
    </div>
  );
}

/** Compact histogram (counts per bin). */
export function Histogram({ edges, counts }: { edges: number[]; counts: number[] }) {
  const t = useTip();
  const top = Math.max(...counts, 1);
  return (
    <div className="relative" ref={t.ref} onMouseLeave={t.hide}>
      <div className="flex h-14 items-end gap-0.5">
        {counts.map((c, i) => (
          <div key={i} className="flex-1 rounded-t-[3px] bg-series-1" style={{ height: `${(c / top) * 100}%`, minHeight: c ? 2 : 0 }}
            onMouseMove={(e) => t.show(e, <>{edges[i].toPrecision(4)} – {edges[i + 1].toPrecision(4)}<br />{c} rows</>)} />
        ))}
      </div>
      <div className="flex justify-between text-xs text-fg-3"><span>{edges[0].toPrecision(3)}</span><span>{edges[edges.length - 1].toPrecision(3)}</span></div>
      {t.node}
    </div>
  );
}

/** Predicted vs actual (regression), with the y = x "perfect prediction" line. */
export function ScatterChart({ points }: { points: { actual: number; predicted: number }[] }) {
  const t = useTip();
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
    <div className="relative" ref={t.ref} onMouseLeave={t.hide}>
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
            onMouseMove={(e) => t.show(e, <>actual <b>{p.actual.toLocaleString()}</b><br />predicted <b>{Math.round(p.predicted).toLocaleString()}</b><br />error {Math.round(p.predicted - p.actual).toLocaleString()}</>)} />
        ))}
        <text x={(P.l + W - P.r) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">actual</text>
        <text x={12} y={(P.t + H - P.b) / 2} textAnchor="middle" fontSize="10" fill="var(--text-muted)" transform={`rotate(-90 12 ${(P.t + H - P.b) / 2})`}>predicted</text>
      </svg>
      {t.node}
    </div>
  );
}
