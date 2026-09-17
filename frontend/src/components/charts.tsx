import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ReactNode } from "react";
import { factorColor, factorName } from "../lib/factors";
import { num, pct } from "../lib/format";

const GRID = "var(--line)";
// Recharts' own draw animation stalls at frame 0 in this setup, so marks are painted
// final-state; the single orchestrated motion on mount is the KPI count-up.
const animate = false;

function Frame({ label, height, children }: { label: string; height: number; children: ReactNode }) {
  return (
    <div role="img" aria-label={label} style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        {children as any}
      </ResponsiveContainer>
    </div>
  );
}

function Box({ head, rows }: { head: string; rows: [string, string, string?][] }) {
  return (
    <div className="tooltip">
      <div className="t-head">{head}</div>
      {rows.map(([k, v, c]) => (
        <div className="t-row" key={k}>
          <span style={{ color: c ?? "var(--text-muted)" }}>{k}</span>
          <span>{v}</span>
        </div>
      ))}
    </div>
  );
}

/** Zero-centred horizontal factor bars with value + t-stat. */
export function FactorBars({
  data, ariaLabel,
}: { data: { factor: string; beta: number; tstat: number }[]; ariaLabel: string }) {
  const max = Math.max(...data.map((d) => Math.abs(d.beta)), 0.001) * 1.15;
  return (
    <div className="fbars" role="img" aria-label={ariaLabel}>
      {data.map((d) => {
        const w = (Math.abs(d.beta) / max) * 50;
        const reliable = Math.abs(d.tstat) >= 2;
        return (
          <div className="fbar" key={d.factor}>
            <div className="name">
              {factorName(d.factor)} <em>{d.factor}</em>
            </div>
            <div className="track">
              <span className="zero" style={{ left: "50%" }} />
              <span
                className="fill"
                style={{
                  background: factorColor(d.factor),
                  width: `${w}%`,
                  left: d.beta >= 0 ? "50%" : `${50 - w}%`,
                }}
              />
            </div>
            <div className="meta">
              <span>{d.beta >= 0 ? "+" : "−"}{Math.abs(d.beta).toFixed(3)}</span>
              <span className={`tag ${reliable ? "on" : ""}`}>
                t {d.tstat >= 0 ? "+" : "−"}{Math.abs(d.tstat).toFixed(1)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Stacked 100% bar — reads as a share breakdown without a donut's label crowding. */
export function ShareBar({
  data, highlight, ariaLabel,
}: { data: { key: string; share: number; color: string }[]; highlight?: string; ariaLabel: string }) {
  const total = data.reduce((s, d) => s + Math.abs(d.share), 0) || 1;
  return (
    <div role="img" aria-label={ariaLabel}>
      <div style={{ display: "flex", height: 34, borderRadius: 4, overflow: "hidden", gap: 1 }}>
        {data.map((d) => (
          <div
            key={d.key}
            title={`${d.key}: ${pct(d.share)}`}
            style={{
              width: `${(Math.abs(d.share) / total) * 100}%`,
              background: d.color,
              opacity: highlight && highlight !== d.key ? 0.32 : 1,
              display: "grid", placeItems: "center",
            }}
          >
            {Math.abs(d.share) / total > 0.09 && (
              <span className="mono" style={{ fontSize: 11, color: "#0E141B" }}>{pct(d.share, 0)}</span>
            )}
          </div>
        ))}
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        {data.map((d) => (
          <span key={d.key}>
            <i style={{ background: d.color }} />
            {d.key} {pct(d.share)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function MultiLine({
  dates, series, colorOf, emphasize, height = 300, ariaLabel, yFormat = (v: number) => num(v, 2),
}: {
  dates: string[]; series: Record<string, (number | null)[]>;
  colorOf: (k: string) => string; emphasize?: string; height?: number;
  ariaLabel: string; yFormat?: (v: number) => string;
}) {
  const keys = Object.keys(series);
  const rows = dates.map((d, i) => {
    const row: Record<string, number | string | null> = { date: d };
    keys.forEach((k) => (row[k] = series[k][i]));
    return row;
  });
  return (
    <Frame label={ariaLabel} height={height}>
      <LineChart data={rows} margin={{ top: 6, right: 10, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(0, 7)} minTickGap={44} />
        <YAxis tickFormatter={yFormat} width={54} />
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <Box
                head={String(label)}
                rows={payload.map((p) => [String(p.name), yFormat(Number(p.value)), String(p.color)])}
              />
            ) : null
          }
        />
        {keys.map((k) => (
          <Line
            key={k} type="monotone" dataKey={k} stroke={colorOf(k)} dot={false}
            strokeWidth={emphasize === k ? 2.4 : 1.3}
            strokeOpacity={emphasize && emphasize !== k ? 0.45 : 1}
            isAnimationActive={animate}
          />
        ))}
      </LineChart>
    </Frame>
  );
}

export function AreaCurve({
  dates, values, color, height = 260, ariaLabel, yFormat = (v: number) => pct(v, 0),
}: {
  dates: string[]; values: (number | null)[]; color: string; height?: number;
  ariaLabel: string; yFormat?: (v: number) => string;
}) {
  const rows = dates.map((d, i) => ({ date: d, v: values[i] }));
  return (
    <Frame label={ariaLabel} height={height}>
      <AreaChart data={rows} margin={{ top: 6, right: 10, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" tickFormatter={(d: string) => d.slice(0, 7)} minTickGap={44} />
        <YAxis tickFormatter={yFormat} width={54} />
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <Box head={String(label)} rows={[["value", yFormat(Number(payload[0].value))]]} />
            ) : null
          }
        />
        <Area type="monotone" dataKey="v" stroke={color} fill={color} fillOpacity={0.22}
              strokeWidth={1} isAnimationActive={animate} />
      </AreaChart>
    </Frame>
  );
}

export function Histogram({
  bins, counts, markers = [], height = 300, ariaLabel,
}: {
  bins: number[]; counts: number[]; height?: number; ariaLabel: string;
  markers?: { value: number; label: string; color: string }[];
}) {
  const rows = bins.map((b, i) => ({ bin: b, count: counts[i] }));
  return (
    <Frame label={ariaLabel} height={height}>
      <BarChart data={rows} margin={{ top: 16, right: 10, left: -14, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="bin" tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} minTickGap={28} />
        <YAxis width={46} />
        <Tooltip
          content={({ active, payload }) =>
            active && payload?.length ? (
              <Box
                head={`${(Number(payload[0].payload.bin) * 100).toFixed(2)}% daily`}
                rows={[["days", String(payload[0].value)]]}
              />
            ) : null
          }
        />
        <Bar dataKey="count" fill="var(--accent)" fillOpacity={0.55} isAnimationActive={animate} />
        {markers.map((m, i) => (
          <ReferenceLine
            key={m.label} x={bins.reduce((a, b) => (Math.abs(b - m.value) < Math.abs(a - m.value) ? b : a), bins[0])}
            stroke={m.color} strokeDasharray="4 3"
            /* thresholds sit within a few basis points of each other, so stagger the labels */
            label={{ value: `${m.label} ${(m.value * 100).toFixed(2)}%`, position: "top",
                     fill: m.color, fontSize: 10, dy: i * 13 }}
          />
        ))}
      </BarChart>
    </Frame>
  );
}

export function GroupedBars({
  rows, keys, colorOf, height = 300, ariaLabel, yFormat = (v: number) => pct(v, 0),
}: {
  rows: Record<string, string | number>[]; keys: string[]; colorOf: (k: string) => string;
  height?: number; ariaLabel: string; yFormat?: (v: number) => string;
}) {
  return (
    <Frame label={ariaLabel} height={height}>
      <BarChart data={rows} margin={{ top: 6, right: 10, left: -14, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="name" />
        <YAxis tickFormatter={yFormat} width={54} />
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <Box
                head={String(label)}
                rows={payload.map((p) => [String(p.name), yFormat(Number(p.value)), String(p.color)])}
              />
            ) : null
          }
        />
        {keys.map((k) => (
          <Bar key={k} dataKey={k} fill={colorOf(k)} isAnimationActive={animate} radius={[2, 2, 0, 0]} />
        ))}
      </BarChart>
    </Frame>
  );
}

/** Diverging correlation heatmap, centred at 0. */
export function Heatmap({
  labels, matrix, ariaLabel,
}: { labels: string[]; matrix: number[][]; ariaLabel: string }) {
  const cell = (v: number) => {
    const a = Math.min(1, Math.abs(v));
    const hue = v >= 0 ? "76, 120, 168" : "228, 87, 86";
    return `rgba(${hue}, ${0.12 + a * 0.72})`;
  };
  return (
    <div
      className="heat" role="img" aria-label={ariaLabel}
      style={{ gridTemplateColumns: `minmax(48px, auto) repeat(${labels.length}, minmax(0, 1fr))` }}
    >
      <div />
      {labels.map((l) => <div className="hlabel" key={`h${l}`}>{l}</div>)}
      {matrix.map((row, i) => (
        <>
          <div className="vlabel" key={`v${labels[i]}`}>{labels[i]}</div>
          {row.map((v, j) => (
            <div className="cell" key={`${i}-${j}`} style={{ background: cell(v) }}
                 title={`${labels[i]} / ${labels[j]}: ${v.toFixed(2)}`}>
              {v.toFixed(2)}
            </div>
          ))}
        </>
      ))}
    </div>
  );
}
