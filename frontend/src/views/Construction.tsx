import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { MultiLine, ShareBar } from "../components/charts";
import { Panel } from "../components/ui";
import { assetColor } from "../lib/factors";
import { num, pct } from "../lib/format";

const PATH_COLOR: Record<string, string> = {
  Current: "var(--accent)",
  "Min-Variance": "var(--f-smb)",
  "Risk-Parity": "var(--f-rmw)",
  "Max-Diversification": "var(--f-mom)",
};

export default function Construction() {
  const d = useOutletContext<Analysis>();
  const assets = Object.keys(d.weights);

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Construction</h2>
        <p className="caption">
          Comparison portfolios. Each needs only the covariance matrix, so given Σ the solution is a
          deterministic convex program — there is no expected-return input and nothing is forecast.
          <span className="measured">deterministic, not predicted</span>
        </p>
      </div>

      <div className="grid g2">
        {d.construction.portfolios.map((p) => (
          <Panel key={p.name} title={p.name} className={p.name === "Current" ? "keep-mobile" : ""}
                 note={`Annualised volatility ${pct(p.vol_ann, 1)}, diversification ratio ${num(p.diversification_ratio, 2)}, largest single risk share ${pct(p.max_risk_share, 0)}.`}>
            <ShareBar
              data={assets.map((a, i) => ({ key: a, share: p.risk_contrib[a] ?? 0, color: assetColor(i) }))}
              ariaLabel={`Risk contributions for the ${p.name} portfolio`}
            />
            <table style={{ marginTop: 14 }} aria-label={`${p.name} weights and risk contributions`}>
              <thead><tr><th>Holding</th><th>Weight</th><th>Risk</th></tr></thead>
              <tbody>
                {assets.map((a) => (
                  <tr key={a}>
                    <td>{a}</td>
                    <td>{pct(p.weights[a] ?? 0, 1)}</td>
                    <td>{pct(p.risk_contrib[a] ?? 0, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        ))}
      </div>

      <Panel title="Realised paths, same weights held throughout"
             note="Each book's wealth path computed on the same realised return history with fixed weights. This is a historical comparison of constructions, not a live backtest and not a projection: no trading costs, no rebalancing schedule.">
        <MultiLine
          dates={d.construction.backtest.dates}
          series={d.construction.backtest.paths}
          colorOf={(k) => PATH_COLOR[k] ?? "var(--text-dim)"}
          height={330}
          ariaLabel="Realised wealth paths of the current, minimum-variance, risk-parity and maximum-diversification portfolios"
          yFormat={(v) => num(v, 1)}
        />
      </Panel>

      <div className="callout">
        <b>Risk parity equalises the same Euler contributions</b> shown in the Attribution view.
        That identity — the parts summing exactly to total volatility — is what makes the problem
        well posed in the first place.
      </div>
    </div>
  );
}
