import { useOutletContext } from "react-router-dom";
import type { Analysis } from "../api/types";
import { GroupedBars, Heatmap, MultiLine } from "../components/charts";
import { Panel, StatTile } from "../components/ui";
import { factorName } from "../lib/factors";
import { num, pct } from "../lib/format";

export default function Structure() {
  const s = useOutletContext<Analysis>().structure;
  const d = useOutletContext<Analysis>();
  const scree = s.components.map((c, i) => ({ name: c, explained: s.pca_scree[i] }));
  const pcsFor90 = s.pca_cumulative.findIndex((v) => v >= 0.9) + 1;
  const flagged = s.vif.filter((v) => v.vif > 5);

  return (
    <div className="dense">
      <div className="view-head">
        <h2>Structure</h2>
        <p className="caption">
          Pure linear algebra on structure that already exists in the sample: how many independent
          directions of risk the book really holds, and how far the factors overlap.
          <span className="measured">measured, not predicted</span>
        </p>
      </div>

      <div className="grid g-1-2">
        <Panel className="keep-mobile" title="Effective number of bets"
               note="Entropy of the eigenvalue spectrum, exp(−Σ pᵢ ln pᵢ). Bounded between 1 and the number of holdings by construction.">
          <div className="statgrid">
            <StatTile animate label="N effective" value={s.effective_bets} format={(v) => num(v, 2)}
                      sub={`of ${s.n_assets} holdings`}
                      hint="One means every holding is the same bet; n means they are mutually independent." />
            <StatTile label="First component" value={s.pca_scree[0]} format={(v) => pct(v, 1)}
                      sub="of total variance" hint="Share of variance along the dominant direction." />
            <StatTile label="Components for 90%" value={pcsFor90} format={(v) => String(Math.round(v))}
                      sub={`of ${s.n_assets}`} hint="How many independent directions it takes to span 90% of the variance." />
          </div>
        </Panel>
        <Panel title="Principal component spectrum"
               note="Each bar is the variance along one independent direction; the line is the cumulative share.">
          <GroupedBars rows={scree} keys={["explained"]} colorOf={() => "var(--accent)"} height={230}
                       ariaLabel="Scree plot of principal component explained variance" />
          <MultiLine dates={s.components}
                     series={{ cumulative: s.pca_cumulative }}
                     colorOf={() => "var(--f-rmw)"} height={140}
                     ariaLabel="Cumulative explained variance by principal component"
                     yFormat={(v) => pct(v, 0)} />
        </Panel>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Panel title="Factor multicollinearity"
               note="VIF_j = 1/(1 − R²_j) from regressing each factor on the others. Above 5, individual betas get unstable and should be read as a pair.">
          <table aria-label="Variance inflation factors by factor">
            <thead><tr><th>Factor</th><th>VIF</th><th>Reading</th></tr></thead>
            <tbody>
              {s.vif.map((v) => (
                <tr key={v.factor}>
                  <td>{factorName(v.factor)} <span className="mono" style={{ color: "var(--text-dim)" }}>{v.factor}</span></td>
                  <td>{num(v.vif, 2)}</td>
                  <td>{v.vif > 5
                    ? <span className="tag flagged">unstable</span>
                    : <span className="tag on">independent enough</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {flagged.length > 0 && (
            <p className="note" style={{ marginTop: 12 }}>
              {flagged.map((f) => factorName(f.factor)).join(" and ")} overlap enough that their
              individual betas trade off against each other.
            </p>
          )}
        </Panel>
        <Panel title="Factor correlation"
               note="How much the factor bets already overlap before any portfolio is applied.">
          <Heatmap labels={s.factor_correlation.labels} matrix={s.factor_correlation.matrix}
                   ariaLabel="Correlation matrix of the factor returns" />
        </Panel>
      </div>

      <Panel title="Component loadings"
             note={`Which holdings move together along each direction. Over ${d.meta.trading_days} trading days.`}>
        <Heatmap labels={s.loadings.assets}
                 matrix={s.loadings.matrix}
                 ariaLabel="Principal component loadings by holding" />
      </Panel>
    </div>
  );
}
