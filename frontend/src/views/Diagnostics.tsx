import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { Panel, TestCard } from "../components/ui";

const FLAG_WORDS = ["non-normal", "autocorrelated", "unit root"];

export default function Diagnostics() {
  const d = useOutletContext<Analysis>();
  const residual = d.diagnostics.filter((t) => t.name.includes("residuals"));
  const adf = d.diagnostics.filter((t) => t.name.startsWith("ADF"));

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Diagnostics</h2>
        <p className="caption">
          Classical hypothesis tests on the sample. Each result licenses a modelling choice made
          elsewhere in this app — that is the only reason they are here.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <div className="grid g2">
        {residual.map((t) => (
          <TestCard key={t.name} name={t.name} stat={t.stat} pvalue={t.pvalue}
                    verdict={t.verdict} licenses={t.licenses}
                    flagged={FLAG_WORDS.some((w) => t.verdict.includes(w))} />
        ))}
      </div>

      <Panel title="Stationarity of the factor series" className="keep-mobile"
             note="Augmented Dickey–Fuller on each factor. A unit root would make the betas spurious, so this is what licenses using them as regressors at all.">
        <table aria-label="Augmented Dickey-Fuller test results by factor">
          <thead><tr><th>Series</th><th>Statistic</th><th>p</th><th>Verdict</th></tr></thead>
          <tbody>
            {adf.map((t) => (
              <tr key={t.name}>
                <td>{t.name.replace("ADF ", "").replace(/[()]/g, "")}</td>
                <td>{t.stat.toFixed(2)}</td>
                <td>{t.pvalue < 0.0001 ? "<0.0001" : t.pvalue.toFixed(4)}</td>
                <td>{t.verdict === "stationary"
                  ? <span className="tag on">stationary</span>
                  : <span className="tag flagged">unit root</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel title="What this analysis will not tell you">
        <p className="prose">
          Every number here is estimated on a historical window, so all of it is backward-looking.
          Exposures drift — the rolling betas in the Attribution view show it directly — and a
          shorter window makes every estimate noisier. The engine refuses any regression with fewer
          than 60 observations; this one ran on {d.meta.trading_days} trading days from{" "}
          {d.meta.start} to {d.meta.end}, with HAC standard errors at lag {d.meta.hac_lag}.
        </p>
      </Panel>
    </div>
  );
}
