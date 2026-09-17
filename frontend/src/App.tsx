import { useEffect, useMemo, useRef, useState } from "react";
import { NavLink, Outlet, Route, Routes, useSearchParams } from "react-router-dom";
import { useAnalysis } from "./api/client";
import type { AnalyzeRequest, Holding } from "./api/types";
import { Panel } from "./components/ui";
import Overview from "./views/Overview";
import Attribution from "./views/Attribution";
import RiskPerformance from "./views/RiskPerformance";
import TailRisk from "./views/TailRisk";
import Covariance from "./views/Covariance";
import Construction from "./views/Construction";
import Structure from "./views/Structure";
import Diagnostics from "./views/Diagnostics";

const VIEWS = [
  { path: "", label: "Overview" },
  { path: "attribution", label: "Attribution" },
  { path: "risk", label: "Risk & Performance" },
  { path: "tail", label: "Tail Risk" },
  { path: "covariance", label: "Covariance" },
  { path: "construction", label: "Construction" },
  { path: "structure", label: "Structure" },
  { path: "diagnostics", label: "Diagnostics" },
];

const SAMPLE = "AAPL:0.25,MSFT:0.20,JPM:0.20,XOM:0.20,JNJ:0.15";
const DEFAULT_START = "2018-01-01";
const DEFAULT_END = "2025-06-30";

const parseHoldings = (s: string): Holding[] =>
  s.split(",").filter(Boolean).map((part) => {
    const [ticker, weight] = part.split(":");
    return { ticker: ticker.toUpperCase(), weight: Number(weight) };
  }).filter((h) => h.ticker && isFinite(h.weight));

const encodeHoldings = (h: Holding[]) =>
  h.map((x) => `${x.ticker}:${x.weight}`).join(",");

export default function App() {
  const [params, setParams] = useSearchParams();
  const holdings = parseHoldings(params.get("p") ?? "");
  const start = params.get("start") ?? DEFAULT_START;
  const end = params.get("end") ?? DEFAULT_END;
  const useMock = params.get("mock") === "1" || import.meta.env.VITE_MOCK === "1";

  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem("theme") as "dark" | "light") ?? "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  const [draftStart, setDraftStart] = useState(start);
  const [draftEnd, setDraftEnd] = useState(end);
  const [draft, setDraft] = useState<Holding[]>(holdings);
  const [ticker, setTicker] = useState("");
  const [weight, setWeight] = useState("");

  const request: AnalyzeRequest | null = useMemo(
    () => (holdings.length ? { holdings, start, end } : null),
    [params.toString()],
  );
  const { data, isFetching, error } = useAnalysis(request, useMock);

  const dirty =
    encodeHoldings(draft) !== encodeHoldings(holdings) || draftStart !== start || draftEnd !== end;

  const run = (next = draft, s = draftStart, e = draftEnd) => {
    const p = new URLSearchParams(params);
    p.set("p", encodeHoldings(next));
    p.set("start", s);
    p.set("end", e);
    setParams(p);
  };

  const loadSample = () => {
    const next = parseHoldings(SAMPLE);
    setDraft(next);
    run(next, DEFAULT_START, DEFAULT_END);
  };

  const addHolding = () => {
    const t = ticker.trim().toUpperCase();
    const w = Number(weight);
    if (!t || !isFinite(w) || w <= 0) return;
    setDraft([...draft.filter((h) => h.ticker !== t), { ticker: t, weight: w }]);
    setTicker("");
    setWeight("");
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <b>Factor Risk Terminal</b>
          <span>gs-quant engine</span>
        </div>

        <div className="chips">
          {draft.map((h) => (
            <span className="chip" key={h.ticker}>
              {h.ticker} <span className="w">{(h.weight * 100).toFixed(0)}%</span>
              <button onClick={() => setDraft(draft.filter((x) => x.ticker !== h.ticker))}
                      aria-label={`Remove ${h.ticker}`}>×</button>
            </span>
          ))}
          <input
            aria-label="Ticker" placeholder="TICKER" value={ticker} size={7}
            onChange={(e) => setTicker(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addHolding()}
          />
          <input
            aria-label="Weight" placeholder="0.20" value={weight} size={5}
            onChange={(e) => setWeight(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addHolding()}
          />
          <button className="btn quiet" onClick={addHolding}>Add</button>
        </div>

        <div className="field">
          <label htmlFor="start">From</label>
          <input id="start" type="date" value={draftStart} onChange={(e) => setDraftStart(e.target.value)} />
          <label htmlFor="end">To</label>
          <input id="end" type="date" value={draftEnd} onChange={(e) => setDraftEnd(e.target.value)} />
        </div>

        <button className="btn primary" onClick={() => run()} disabled={!draft.length || isFetching}>
          {isFetching ? "Computing" : dirty ? "Run" : "Rerun"}
        </button>
        <button className="btn quiet" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                aria-label="Switch theme">
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        <span className="powered">Powered by gs-quant</span>
      </header>

      <div className="body">
        <Rail />
        <main className="stage">
          <div className="stage-inner">
            {!request && <Empty onSample={loadSample} />}
            {request && isFetching && <Loading />}
            {request && !isFetching && error && <ErrorState message={(error as Error).message} />}
            {request && !isFetching && data && (
              <Routes>
                <Route element={<Outlet context={data} />}>
                  <Route index element={<Overview />} />
                  <Route path="attribution" element={<Attribution />} />
                  <Route path="risk" element={<RiskPerformance />} />
                  <Route path="tail" element={<TailRisk />} />
                  <Route path="covariance" element={<Covariance />} />
                  <Route path="construction" element={<Construction />} />
                  <Route path="structure" element={<Structure />} />
                  <Route path="diagnostics" element={<Diagnostics />} />
                </Route>
              </Routes>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

function Rail() {
  const [collapsed, setCollapsed] = useState(false);
  const ref = useRef<HTMLElement>(null);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const links = Array.from(ref.current?.querySelectorAll("a") ?? []);
    const i = links.indexOf(document.activeElement as HTMLAnchorElement);
    if (i < 0) return;
    e.preventDefault();
    links[(i + (e.key === "ArrowDown" ? 1 : links.length - 1)) % links.length].focus();
  };

  return (
    <nav ref={ref} className={`rail ${collapsed ? "collapsed" : ""}`} aria-label="Analysis views"
         onKeyDown={onKeyDown}>
      {VIEWS.map((v, i) => (
        <NavLink key={v.path} to={v.path ? `/${v.path}${location.search}` : `/${location.search}`}
                 end={v.path === ""} className={({ isActive }) => (isActive ? "active" : "")}>
          <span className="idx">{i + 1}</span>
          <span className="txt">{v.label}</span>
        </NavLink>
      ))}
      <button className="btn quiet collapse" onClick={() => setCollapsed(!collapsed)}
              aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}>
        {collapsed ? "»" : "« Collapse"}
      </button>
    </nav>
  );
}

function Empty({ onSample }: { onSample: () => void }) {
  return (
    <div className="center">
      <div>
        <h1>Add holdings to see where this portfolio's risk actually lives.</h1>
        <p className="prose">
          Enter tickers and weights above, then run the analysis. Every number is measured on
          realised history with gs-quant — factor attribution, risk decomposition, tail risk.
        </p>
        <button className="btn primary" onClick={onSample}>Load a sample portfolio</button>
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div>
      <div className="computing" style={{ marginBottom: 16 }}>Computing on gs-quant…</div>
      <div className="skeleton" style={{ height: 210, marginBottom: 16 }} />
      <div className="kpis">
        {Array.from({ length: 6 }).map((_, i) => (
          <div className="stat" key={i}>
            <div className="skeleton" style={{ height: 11, width: "50%" }} />
            <div className="skeleton" style={{ height: 24, marginTop: 8 }} />
          </div>
        ))}
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <div className="skeleton" style={{ height: 260 }} />
        <div className="skeleton" style={{ height: 260 }} />
      </div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="center">
      <Panel className="errorbox">
        <h3>That analysis didn't run</h3>
        <p className="prose">{message}</p>
      </Panel>
    </div>
  );
}
