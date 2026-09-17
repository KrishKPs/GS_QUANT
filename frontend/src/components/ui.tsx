import { useEffect, useRef, useState, type ReactNode } from "react";

export const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function Panel({
  title, note, actions, children, className = "", hero = false,
}: {
  title?: string; note?: string; actions?: ReactNode; children: ReactNode;
  className?: string; hero?: boolean;
}) {
  return (
    <section className={`panel ${hero ? "hero" : ""} ${className}`}>
      {(title || actions) && (
        <header>
          {title && <h3>{title}</h3>}
          {actions}
        </header>
      )}
      {note && <p className="note">{note}</p>}
      {children}
    </section>
  );
}

/** Counts up once on mount; static under reduced-motion. */
function useCountUp(target: number, run: boolean) {
  const [v, setV] = useState(run ? 0 : target);
  const done = useRef(false);
  useEffect(() => {
    if (!run || done.current || !isFinite(target)) { setV(target); return; }
    done.current = true;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / 550);
      setV(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, run]);
  return v;
}

export function StatTile({
  label, value, format, sub, hint, animate = false,
}: {
  label: string; value: number; format: (n: number) => string;
  sub?: string; hint?: string; animate?: boolean;
}) {
  const shown = useCountUp(value, animate && !prefersReducedMotion());
  return (
    <div className="stat">
      <div className="label">
        {label}
        {hint && <span className="hint" title={hint} aria-label={hint}>?</span>}
      </div>
      <div className="value">{format(shown)}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Toggle<T extends string>({
  options, value, onChange, label,
}: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div className="toggle" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} aria-pressed={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function DataTable({
  columns, rows, ariaLabel,
}: { columns: string[]; rows: (string | number | ReactNode)[][]; ariaLabel: string }) {
  return (
    <div className="scroll">
      <table aria-label={ariaLabel}>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}><i style={{ background: it.color }} />{it.label}</span>
      ))}
    </div>
  );
}

export function TestCard({
  name, stat, pvalue, verdict, licenses, flagged,
}: { name: string; stat: number; pvalue: number; verdict: string; licenses: string; flagged: boolean }) {
  return (
    <div className="testcard">
      <h3>{name}</h3>
      <div className="stat-line">
        <span className="n">{stat.toFixed(2)}</span>
        <span className="mono" style={{ color: "var(--text-dim)", fontSize: 11.5 }}>
          p = {pvalue < 0.0001 ? "<0.0001" : pvalue.toFixed(4)}
        </span>
        <span className={`tag ${flagged ? "flagged" : "on"}`}>{verdict}</span>
      </div>
      <p className="licenses">{licenses}</p>
    </div>
  );
}
