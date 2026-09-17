import type { Analysis } from "../api/types";
import { factorName } from "./factors";
import { ordinal } from "./format";

export interface Insight {
  factor: string;
  name: string;
  share: number;
  betaRank: number;
  shareRank: number;
  disagrees: boolean;
}

/**
 * The thesis of the tool: find the factor whose share of risk outruns its rank
 * as a bet. Ranks are computed on |beta| (bet size) vs share of variance.
 */
export function findDisagreement(data: Analysis): Insight {
  const shares = data.risk_shares.filter((s) => s.factor !== "IDIO");
  const byShare = [...shares].sort((a, b) => b.share - a.share);
  const byBeta = [...data.exposures].sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta));
  const betaRank = (f: string) => byBeta.findIndex((e) => e.factor === f) + 1;

  const scored = byShare.map((s) => ({
    factor: s.factor,
    name: factorName(s.factor),
    share: s.share,
    betaRank: betaRank(s.factor),
    shareRank: byShare.findIndex((x) => x.factor === s.factor) + 1,
  }));
  // biggest positive gap between how much risk it drives and how big the bet looks
  const best = [...scored].sort(
    (a, b) => (b.betaRank - b.shareRank) * b.share - (a.betaRank - a.shareRank) * a.share,
  )[0];
  const top = scored[0];
  const chosen = best && best.betaRank > best.shareRank ? best : top;
  return { ...chosen, disagrees: chosen.betaRank > chosen.shareRank };
}

export function insightSentence(data: Analysis, i: Insight) {
  const n = data.metrics.n_assets;
  return i.disagrees
    ? `This portfolio looks diversified across ${n} holdings, but ${i.name} drives ${(i.share * 100).toFixed(0)}% of its risk while being only its ${ordinal(i.betaRank)}-largest bet.`
    : `This portfolio looks diversified across ${n} holdings, but ${i.name} alone drives ${(i.share * 100).toFixed(0)}% of its risk — the ${n} names amount to ${data.metrics.effective_bets.toFixed(1)} independent bets.`;
}
