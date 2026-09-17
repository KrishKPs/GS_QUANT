import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { GroupedBars, Histogram } from "../components/charts";
import { Panel, StatTile } from "../components/ui";
import { pct } from "../lib/format";

const METHOD_COLOR: Record<string, string> = {
  historical: "var(--f-smb)",
  gaussian: "var(--f-mom)",
  cornish_fisher: "var(--f-hml)",
};

export default function TailRisk() {
  const d = useOutletContext<Analysis>();
  const nonNormal = d.var.jarque_bera_p < 0.05;
  const l95 = d.var.levels.find((l) => l.alpha === 0.05)!;

  const rows = d.var.levels.map((l) => ({
    name: `${l.label} confidence`,
    historical: l.historical,
    gaussian: l.gaussian,
    cornish_fisher: l.cornish_fisher,
  }));

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Tail Risk</h2>
        <p className="caption">
          One-day horizon, losses shown as positive numbers. These are quantiles and moments of the
          realised distribution — no distribution is fitted and no loss is forecast.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <div className="callout keep-mobile" style={{ marginBottom: 16 }}>
        <b>Jarque–Bera p {d.var.jarque_bera_p < 0.0001 ? "< 0.0001" : `= ${d.var.jarque_bera_p.toFixed(4)}`}.</b>{" "}
        {nonNormal
          ? "returns are not normal, so Gaussian VaR understates the left tail. Trust the historical and Cornish–Fisher numbers."
          : "normality is not rejected here, so the Gaussian number is defensible."}{" "}
        Expected Shortfall is reported alongside because it is coherent (sub-additive) where VaR is not.
      </div>

      <div className="grid g-2-1">
        <Panel title="VaR by method"
               note="Historical is assumption-free but bounded by the sample. Gaussian assumes normality. Cornish–Fisher folds skew and excess kurtosis back into the quantile.">
          <GroupedBars rows={rows} keys={["historical", "gaussian", "cornish_fisher"]}
                       colorOf={(k) => METHOD_COLOR[k]} height={280}
                       ariaLabel="Value at Risk by method and confidence level"
                       yFormat={(v) => pct(v, 1)} />
        </Panel>
        <Panel title="At 95% confidence">
          <div className="statgrid">
            <StatTile label="VaR historical" value={l95.historical} format={(v) => pct(v, 2)}
                      hint="Empirical 5th percentile of daily returns." />
            <StatTile label="VaR Cornish–Fisher" value={l95.cornish_fisher} format={(v) => pct(v, 2)}
                      hint="Gaussian quantile adjusted for the sample's skew and excess kurtosis." />
            <StatTile label="ES historical" value={l95.es_historical} format={(v) => pct(v, 2)}
                      hint="Mean loss on the days worse than the VaR threshold." />
            <StatTile label="ES Gaussian" value={l95.es_gaussian} format={(v) => pct(v, 2)}
                      hint="Expected shortfall under the normal assumption." />
          </div>
        </Panel>
      </div>

      <Panel title="Where the thresholds sit"
             note="The same distribution as the histogram, with each method's 95% threshold marked.">
        <Histogram
          bins={d.performance.return_hist.bins} counts={d.performance.return_hist.counts} height={330}
          ariaLabel={`Daily return distribution with 95 percent Value at Risk thresholds: historical ${pct(l95.historical, 2)}, Gaussian ${pct(l95.gaussian, 2)}, Cornish-Fisher ${pct(l95.cornish_fisher, 2)}`}
          markers={[
            { value: -l95.historical, label: "Historical", color: METHOD_COLOR.historical },
            { value: -l95.gaussian, label: "Gaussian", color: METHOD_COLOR.gaussian },
            { value: -l95.cornish_fisher, label: "Cornish–Fisher", color: METHOD_COLOR.cornish_fisher },
          ]}
        />
      </Panel>

      <Panel title="All levels" className="keep-mobile">
        <table aria-label="VaR and Expected Shortfall by confidence level">
          <thead>
            <tr><th>Confidence</th><th>VaR hist.</th><th>VaR Gaussian</th><th>VaR C–F</th><th>ES hist.</th><th>ES Gaussian</th></tr>
          </thead>
          <tbody>
            {d.var.levels.map((l) => (
              <tr key={l.label}>
                <td>{l.label}</td>
                <td>{pct(l.historical, 2)}</td>
                <td>{pct(l.gaussian, 2)}</td>
                <td>{pct(l.cornish_fisher, 2)}</td>
                <td>{pct(l.es_historical, 2)}</td>
                <td>{pct(l.es_gaussian, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
