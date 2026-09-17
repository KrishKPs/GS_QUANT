import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { MirrorChart } from "../components/Mirror";
import { Panel, StatTile } from "../components/ui";
import { AreaCurve } from "../components/charts";
import { findDisagreement, insightSentence } from "../lib/insight";
import { num, pct } from "../lib/format";

export default function Overview() {
  const d = useOutletContext<Analysis>();
  const insight = findDisagreement(d);
  const sentence = insightSentence(d, insight);
  const [lead, rest] = sentence.split(", but ");

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Overview</h2>
        <p className="caption">
          The finding first: how the portfolio's bets compare with where its risk actually sits.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <Panel hero className="keep-mobile" >
        <div className="hero-grid">
          <div>
            <p className="finding">
              <b>Hidden concentration.</b>{" "}
              {lead}, but <span className="hl">{rest}</span>
            </p>
            <p className="finding-sub">
              Bet size and risk contribution are different questions. A beta says how big the
              position is; the risk share says how much of the portfolio's variance that position
              actually drives, once volatility and factor correlation are accounted for.
              Across {d.metrics.n_assets} holdings this book holds{" "}
              {d.metrics.effective_bets.toFixed(1)} effective independent bets.
            </p>
          </div>
          <MirrorChart exposures={d.exposures} shares={d.risk_shares} highlight={insight.factor} />
        </div>
      </Panel>

      <div className="kpis keep-mobile" style={{ marginTop: 16 }}>
        <StatTile animate label="R²" value={d.metrics.r2} format={(v) => pct(v, 1)}
                  hint="Share of the portfolio's realised return variation explained by the factor model." />
        <StatTile animate label="Alpha (ann.)" value={d.metrics.alpha_ann} format={(v) => pct(v, 2)}
                  sub={`t ${d.metrics.alpha_tstat >= 0 ? "+" : "−"}${Math.abs(d.metrics.alpha_tstat).toFixed(2)}`}
                  hint="Average realised return not explained by the factors, annualised. HAC t-stat shown." />
        <StatTile animate label="Volatility (ann.)" value={d.metrics.vol_ann} format={(v) => pct(v, 1)}
                  hint="Realised standard deviation of daily returns, scaled by the square root of 252." />
        <StatTile animate label="Sharpe" value={d.metrics.sharpe} format={(v) => num(v, 2)}
                  hint="Realised excess return per unit of volatility, annualised. Ex-post only." />
        <StatTile animate label="VaR 95% (1d)" value={d.metrics.var95_1d} format={(v) => pct(v, 2)}
                  sub={`ES ${pct(d.metrics.es95_1d, 2)}`}
                  hint="Cornish–Fisher one-day loss threshold at 95%, adjusted for skew and fat tails." />
        <StatTile animate label="Effective bets" value={d.metrics.effective_bets} format={(v) => num(v, 2)}
                  sub={`of ${d.metrics.n_assets} names`}
                  hint="Entropy of the covariance eigenvalue spectrum: how many independent risk directions the book really holds." />
      </div>

      <div className="grid g-2-1" style={{ marginTop: 16 }}>
        <Panel title="Realised wealth path"
               note={`${d.meta.trading_days} trading days, ${d.meta.start} to ${d.meta.end}. Growth of 1 unit, dividends reinvested.`}>
          <AreaCurve dates={d.performance.wealth_curve.dates} values={d.performance.wealth_curve.values}
                     color="var(--accent)" ariaLabel="Cumulative realised wealth path of the portfolio"
                     yFormat={(v) => num(v, 1)} />
        </Panel>
        <Panel title="Weight is not risk"
               note="Each holding's share of capital against its share of portfolio volatility.">
          <table aria-label="Weight versus risk share by holding">
            <thead>
              <tr><th>Holding</th><th>Weight</th><th>Risk</th><th>Gap</th></tr>
            </thead>
            <tbody>
              {d.asset_risk.map((a) => {
                const gap = a.share - a.weight;
                return (
                  <tr key={a.asset}>
                    <td>{a.asset}</td>
                    <td>{pct(a.weight, 1)}</td>
                    <td>{pct(a.share, 1)}</td>
                    <td style={{ color: Math.abs(gap) > 0.03 ? "var(--text)" : "var(--text-dim)" }}>
                      {gap >= 0 ? "+" : "−"}{pct(Math.abs(gap), 1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>
      <p className="small-screen-note caption" style={{ marginTop: 16 }}>
        The denser views are best on a larger screen.
      </p>
    </div>
  );
}
