export const pct = (x: number | null | undefined, dp = 1) =>
  x == null || !isFinite(x) ? "—" : `${(x * 100).toFixed(dp)}%`;

export const num = (x: number | null | undefined, dp = 2) =>
  x == null || !isFinite(x) ? "—" : x.toFixed(dp);

/** Sign is carried by a glyph, never by colour alone. */
export const signed = (x: number | null | undefined, dp = 2) =>
  x == null || !isFinite(x) ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(dp)}`;

export const signedPct = (x: number | null | undefined, dp = 1) =>
  x == null || !isFinite(x) ? "—" : `${x >= 0 ? "+" : "−"}${(Math.abs(x) * 100).toFixed(dp)}%`;

export const ordinal = (n: number) => {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

export const compactDate = (iso: string) => iso.slice(0, 7);
