import { factorColor, factorName } from "../lib/factors";
import { pct } from "../lib/format";

/**
 * Exposures on the left, risk share on the right, sharing factor rows — the eye
 * lands on the row where bet size and risk contribution disagree.
 */
export function MirrorChart({
  exposures, shares, highlight,
}: {
  exposures: { factor: string; beta: number }[];
  shares: { factor: string; share: number }[];
  highlight?: string;
}) {
  const maxBeta = Math.max(...exposures.map((e) => Math.abs(e.beta)), 0.01);
  const maxShare = Math.max(...shares.map((s) => Math.abs(s.share)), 0.01);
  const shareOf = (f: string) => shares.find((s) => s.factor === f)?.share ?? 0;
  const rows = [...exposures].sort((a, b) => shareOf(b.factor) - shareOf(a.factor));

  return (
    <div
      className="mirror" role="img"
      aria-label={`Factor exposures compared with each factor's share of portfolio risk. ${rows
        .map((r) => `${factorName(r.factor)}: beta ${r.beta.toFixed(2)}, ${pct(shareOf(r.factor))} of risk`)
        .join("; ")}`}
    >
      <div className="mirror-head">
        <div className="l">Bet size (beta)</div>
        <div style={{ textAlign: "center" }}>Factor</div>
        <div>Share of risk</div>
      </div>
      {rows.map((e) => {
        const share = shareOf(e.factor);
        const on = highlight === e.factor;
        return (
          <div className={`mirror-row ${on ? "flag" : ""}`} key={e.factor}>
            <div className="mbar left">
              <span className="v">{e.beta >= 0 ? "+" : "−"}{Math.abs(e.beta).toFixed(2)}</span>
              <i
                style={{
                  width: `${(Math.abs(e.beta) / maxBeta) * 82}%`,
                  background: factorColor(e.factor),
                  opacity: on ? 1 : 0.65,
                }}
              />
            </div>
            <div className="mirror-name">
              {factorName(e.factor)}
              <span className="code">{e.factor}</span>
            </div>
            <div className="mbar">
              <i
                style={{
                  width: `${(Math.abs(share) / maxShare) * 82}%`,
                  background: factorColor(e.factor),
                  opacity: on ? 1 : 0.65,
                }}
              />
              <span className="v">{pct(share)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
