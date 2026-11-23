import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SrTable } from "./charts";

export interface TrendPoint { x: string; y: number; detail?: string }

/** Line over time (Recharts), styled only with design tokens. */
export function TrendChart({ points, label, yLabel, domain = [0, 100], reference, height = 200 }: {
  points: TrendPoint[];
  label: string;
  yLabel: string;
  domain?: [number, number];
  reference?: { y: number; label: string };
  height?: number;
}) {
  return (
    <div>
      <div role="img" aria-label={`${label}: ${points.map((p) => `${p.x} ${p.y}`).join(", ")}`} style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="x" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <YAxis domain={domain} tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} width={44} />
            {reference && (
              <ReferenceLine y={reference.y} stroke="var(--text-muted)" strokeDasharray="4 3"
                label={{ value: reference.label, fill: "var(--text-muted)", fontSize: 11, position: "insideTopRight" }} />
            )}
            <Tooltip
              cursor={{ stroke: "var(--border-strong)" }}
              contentStyle={{ background: "var(--surface-3)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--text-primary)" }}
              labelStyle={{ color: "var(--text-secondary)" }}
              formatter={(v) => [String(v), yLabel]}
            />
            <Line type="monotone" dataKey="y" stroke="var(--series-1)" strokeWidth={2} isAnimationActive={false}
              dot={{ r: 3.5, fill: "var(--series-1)", stroke: "var(--surface-1)", strokeWidth: 2 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <SrTable caption={label} head={["When", yLabel, "Detail"]} rows={points.map((p) => [p.x, p.y, p.detail ?? ""])} />
    </div>
  );
}
