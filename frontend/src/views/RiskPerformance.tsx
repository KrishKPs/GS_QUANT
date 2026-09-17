import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { AreaCurve, Histogram } from "../components/charts";
import { Panel, StatTile } from "../components/ui";
import { num, pct } from "../lib/format";

export default function RiskPerformance() {
  const d = useOutletContext<Analysis>();
  const p = d.performance;

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Risk &amp; Performance</h2>
        <p className="caption">
          Ex-post statistics on realised returns. Ratios are annualised by √252, never by 252.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <Panel className="keep-mobile">
        <div className="statgrid">
          <StatTile label="Sharpe" value={d.metrics.sharpe} format={(v) => num(v, 2)}
                    hint="Excess return per unit of total volatility." />
          <StatTile label="Sortino" value={p.sortino} format={(v) => num(v, 2)}
                    hint="Excess return per unit of downside deviation only." />
          <StatTile label="Information ratio" value={p.information_ratio} format={(v) => num(v, 2)}
                    hint="Active return over the market factor, divided by tracking error." />
          <StatTile label="Calmar" value={p.calmar} format={(v) => num(v, 2)}
                    hint="Annualised return divided by the absolute maximum drawdown." />
          <StatTile label="Ulcer index" value={p.ulcer} format={(v) => num(v, 3)}
                    hint="Root-mean-square drawdown: depth and duration of losses in one number." />
          <StatTile label="Treynor" value={p.treynor} format={(v) => pct(v, 1)}
                    hint="Excess return per unit of market beta." />
        </div>
      </Panel>

      <div className="grid g-2-1" style={{ marginTop: 16 }}>
        <Panel title="Drawdown"
               note={`Worst peak-to-trough loss ${pct(p.max_drawdown, 1)}; longest stretch under a prior peak ${p.drawdown_days} trading days.`}>
          <AreaCurve dates={p.drawdown_curve.dates} values={p.drawdown_curve.values}
                     color="var(--neg)" height={280}
                     ariaLabel={`Drawdown curve, maximum drawdown ${pct(p.max_drawdown, 1)}`} />
        </Panel>
        <Panel title="Shape of the distribution"
               note="Sharpe assumes normality. These two numbers are why that assumption is doing work it can't support.">
          <div className="statgrid">
            <StatTile label="Skew" value={p.skew} format={(v) => num(v, 2)}
                      sub={p.skew < 0 ? "left tail heavier" : "right tail heavier"}
                      hint="Negative skew means losses are the longer tail." />
            <StatTile label="Excess kurtosis" value={p.kurtosis} format={(v) => num(v, 2)}
                      sub={p.kurtosis > 0 ? "fatter than normal" : "thinner than normal"}
                      hint="Zero is Gaussian. Positive means extreme days happen more often than a normal curve allows." />
            <StatTile label="Annual return" value={p.annual_return} format={(v) => pct(v, 1)}
                      sub="geometric, realised" hint="Compound annual growth rate of the realised wealth path." />
            <StatTile label="Volatility" value={d.metrics.vol_ann} format={(v) => pct(v, 1)}
                      sub="annualised" hint="Realised daily standard deviation scaled by √252." />
          </div>
        </Panel>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Panel title="Rolling 126-day volatility"
               note="Descriptive view of regime shifts, computed with gs-quant.">
          <AreaCurve dates={p.rolling_vol.dates} values={p.rolling_vol.values}
                     color="var(--f-mkt)" height={260}
                     ariaLabel="Rolling 126-day annualised volatility"
                     yFormat={(v) => pct(v, 0)} />
        </Panel>
        <Panel title="Daily return distribution"
               note="The empirical distribution the tail-risk numbers are computed from.">
          <Histogram bins={p.return_hist.bins} counts={p.return_hist.counts} height={260}
                     ariaLabel="Histogram of daily portfolio returns" />
        </Panel>
      </div>
    </div>
  );
}
