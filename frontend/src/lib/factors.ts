/** Fixed semantic factor mapping — a colour always means the same factor. */
export const FACTOR_COLOR: Record<string, string> = {
  MKT: "var(--f-mkt)",
  SMB: "var(--f-smb)",
  HML: "var(--f-hml)",
  RMW: "var(--f-rmw)",
  MOM: "var(--f-mom)",
  IDIO: "var(--f-idio)",
  IDIOSYNCRATIC: "var(--f-idio)",
};

export const FACTOR_NAME: Record<string, string> = {
  MKT: "Market",
  SMB: "Size",
  HML: "Value",
  RMW: "Quality",
  MOM: "Momentum",
  IDIO: "Stock-specific",
};

export const factorColor = (code: string) => FACTOR_COLOR[code] ?? "var(--text-dim)";
export const factorName = (code: string) => FACTOR_NAME[code] ?? code;

/** Assets are not factors: a neutral structural ramp, never the factor hues. */
const ASSET_RAMP = ["#5B8DEF", "#8FB4F5", "#3E5C8A", "#B9C7DC", "#6C7A94", "#2E4468", "#DCE4F0"];
export const assetColor = (i: number) => ASSET_RAMP[i % ASSET_RAMP.length];
