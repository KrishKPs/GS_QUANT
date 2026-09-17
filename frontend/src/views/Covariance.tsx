import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { Heatmap, ShareBar } from "../components/charts";
import { Panel, StatTile, Toggle } from "../components/ui";
import { assetColor } from "../lib/factors";
import { num, pct } from "../lib/format";

const LABELS: Record<string, string> = {
  sample: "Sample",
  ledoit_wolf: "Ledoit–Wolf",
  ewma: "EWMA λ=0.94",
};

const NOTES: Record<string, string> = {
  sample: "The plain sample covariance: unbiased, but noisy when the number of names approaches the number of observations.",
  ledoit_wolf: "Shrinks the sample matrix toward a structured target with the closed-form intensity δ*. A little bias for a large variance reduction, and a guaranteed-invertible matrix.",
  ewma: "Exponentially weighted, so recent structure dominates. It describes the present regime; it does not extrapolate it.",
};

export default function Covariance() {
  const d = useOutletContext<Analysis>();
  const [est, setEst] = useState(d.meta.estimator);
  const cur = d.covariance.by_estimator[est];
  const base = d.covariance.by_estimator[d.meta.estimator];
  const shift = cur.risk_shares.map((r) => {
    const before = base.risk_shares.find((b) => b.asset === r.asset)?.share ?? 0;
    return { asset: r.asset, delta: r.share - before };
  }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0];

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Covariance</h2>
        <p className="caption">
          One matrix drives the risk decomposition, the VaR and every optimiser, so how it is
          estimated matters. All three are estimators of present structure.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <Panel
        className="keep-mobile"
        title="Estimator"
        note={NOTES[est]}
        actions={
          <Toggle
            label="Covariance estimator"
            value={est}
            onChange={setEst}
            options={d.covariance.estimators.map((e) => ({ value: e, label: LABELS[e] ?? e }))}
          />
        }
      >
        <div className="statgrid">
          <StatTile label="Portfolio volatility" value={cur.vol_ann} format={(v) => pct(v, 1)}
                    sub="annualised" hint="√(wᵀΣw), annualised by √252, under this estimator." />
          <StatTile label="Condition number" value={cur.condition_number} format={(v) => num(v, 1)}
                    sub={cur.condition_number > 1e10 ? "ill-conditioned" : "well conditioned"}
                    hint="Ratio of largest to smallest eigenvalue. Above 1e10 the matrix is effectively singular." />
          <StatTile label="Effective bets" value={cur.effective_bets} format={(v) => num(v, 2)}
                    sub={`of ${d.metrics.n_assets}`} hint="Entropy of this matrix's eigenvalue spectrum." />
          <StatTile label="Largest risk share" value={Math.max(...cur.risk_shares.map((r) => r.share))}
                    format={(v) => pct(v, 1)} hint="The single holding contributing most volatility under this estimator." />
        </div>
      </Panel>

      <div className="grid g-1-2" style={{ marginTop: 16 }}>
        <Panel title="Risk shares under this estimator"
               note={shift && Math.abs(shift.delta) > 0.001
                 ? `Switching from ${LABELS[d.meta.estimator]} moves ${shift.asset} by ${(shift.delta * 100).toFixed(1)} points.`
                 : "Shares are stable across estimators for this book."}>
          <ShareBar
            data={cur.risk_shares.map((r, i) => ({ key: r.asset, share: r.share, color: assetColor(i) }))}
            ariaLabel="Share of portfolio volatility by holding under the selected estimator"
          />
        </Panel>
        <Panel title="Asset correlation"
               note="Diverging scale centred at zero; blue is positive, red negative. Values are labelled, so colour is never the only carrier of meaning.">
          <Heatmap labels={d.covariance.labels} matrix={cur.correlation}
                   ariaLabel="Correlation matrix of holdings under the selected estimator" />
        </Panel>
      </div>
    </div>
  );
}
