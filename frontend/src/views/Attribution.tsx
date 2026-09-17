import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { FactorBars, MultiLine, ShareBar } from "../components/charts";
import { Panel } from "../components/ui";
import { assetColor, factorColor, factorName } from "../lib/factors";
import { num, pct } from "../lib/format";

export default function Attribution() {
  const d = useOutletContext<Analysis>();
  const rolling = d.rolling_betas;
  const drift = Object.entries(rolling.series)
    .map(([k, v]) => {
      const xs = v.filter((x): x is number => x != null);
      return { k, range: xs.length ? Math.max(...xs) - Math.min(...xs) : 0 };
    })
    .sort((a, b) => b.range - a.range)[0];

  const shares = d.risk_shares.map((s) => ({
    key: s.factor, share: s.share, color: factorColor(s.factor),
  }));
  const assetShares = d.asset_risk.map((a, i) => ({
    key: a.asset, share: a.share, color: assetColor(i),
  }));

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Attribution</h2>
        <p className="caption">
          Betas are bet size; risk contributions are how much each bet drives variance. t-stats use
          HAC (Newey–West) standard errors at lag {d.meta.hac_lag}, because daily residuals are
          autocorrelated and heteroskedastic.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <Panel
        className="keep-mobile"
        title="Factor exposures"
        note={`The model explains ${pct(d.metrics.r2)} of return variation (adjusted ${pct(d.metrics.adj_r2)}) over ${d.meta.trading_days} days. A "reliable" tag marks |t| ≥ 2.`}
      >
        <FactorBars data={d.exposures} ariaLabel="Factor exposures with HAC t-statistics" />
      </Panel>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Panel title="Variance decomposition"
               note="Factor contributions to variance, CCV_i = β_i (Σ_F β)_i, plus what the factors leave unexplained. Cross-terms carry factor correlation, so this is not squared betas.">
          <ShareBar data={shares} ariaLabel="Share of portfolio variance by factor" />
        </Panel>
        <Panel title="Volatility decomposition"
               note="Euler contributions by holding: σ_p is homogeneous of degree 1 in the weights, so the parts sum exactly to total volatility.">
          <ShareBar data={assetShares} ariaLabel="Share of portfolio volatility by holding" />
        </Panel>
      </div>

      {rolling.dates.length > 0 && (
        <Panel title="Rolling betas, 252-day window"
               note={`Exposures drift — one full-sample beta hides that. ${drift ? `${factorName(drift.k)} moves most across the window.` : ""}`}
               >
          <MultiLine
            dates={rolling.dates} series={rolling.series} colorOf={factorColor}
            emphasize={drift?.k} height={320}
            ariaLabel="Rolling 252-day factor betas over time"
            yFormat={(v) => num(v, 2)}
          />
        </Panel>
      )}

      <Panel title="Exposure table" className="keep-mobile">
        <table aria-label="Factor exposures, t-statistics and variance shares">
          <thead>
            <tr><th>Factor</th><th>Beta</th><th>HAC t</th><th>Reliable</th><th>Variance share</th></tr>
          </thead>
          <tbody>
            {d.exposures.map((e) => {
              const share = d.risk_shares.find((s) => s.factor === e.factor)?.share ?? 0;
              return (
                <tr key={e.factor}>
                  <td>{factorName(e.factor)} <span className="mono" style={{ color: "var(--text-dim)" }}>{e.factor}</span></td>
                  <td>{e.beta >= 0 ? "+" : "−"}{Math.abs(e.beta).toFixed(3)}</td>
                  <td>{e.tstat >= 0 ? "+" : "−"}{Math.abs(e.tstat).toFixed(2)}</td>
                  <td>{Math.abs(e.tstat) >= 2 ? <span className="tag on">reliable</span> : <span className="tag">noise</span>}</td>
                  <td>{pct(share, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
