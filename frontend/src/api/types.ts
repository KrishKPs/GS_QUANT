/** Mirrors the FastAPI response contract. Codes only — colours are a frontend concern. */
export interface Exposure { factor: string; beta: number; tstat: number }
export interface RiskShare { factor: string; share: number }
export interface AssetRisk { asset: string; weight: number; share: number; mcr: number }
export interface Curve { dates: string[]; values: (number | null)[] }

export interface Analysis {
  meta: { trading_days: number; start: string; end: string; skipped: string[];
          hac_lag: number; estimator: string };
  metrics: { r2: number; adj_r2: number; alpha_ann: number; alpha_tstat: number;
             vol_ann: number; sharpe: number; var95_1d: number; es95_1d: number;
             effective_bets: number; n_assets: number };
  weights: Record<string, number>;
  exposures: Exposure[];
  risk_shares: RiskShare[];
  asset_risk: AssetRisk[];
  rolling_betas: { dates: string[]; series: Record<string, (number | null)[]> };
  performance: {
    annual_return: number; sortino: number; calmar: number; ulcer: number;
    information_ratio: number; treynor: number; max_drawdown: number;
    drawdown_days: number; skew: number; kurtosis: number;
    drawdown_curve: Curve; wealth_curve: Curve; rolling_vol: Curve;
    return_hist: { bins: number[]; counts: number[] };
  };
  var: {
    levels: { alpha: number; label: string; historical: number; gaussian: number;
              cornish_fisher: number; es_historical: number; es_gaussian: number }[];
    historical: number; gaussian: number; cornish_fisher: number; jarque_bera_p: number;
  };
  covariance: {
    estimators: string[];
    labels: string[];
    by_estimator: Record<string, {
      correlation: number[][]; condition_number: number; vol_ann: number;
      effective_bets: number; risk_shares: { asset: string; share: number }[];
    }>;
  };
  construction: {
    portfolios: { name: string; weights: Record<string, number>;
                  risk_contrib: Record<string, number>; vol_ann: number;
                  diversification_ratio: number; max_risk_share: number }[];
    backtest: { dates: string[]; paths: Record<string, (number | null)[]> };
  };
  structure: {
    pca_scree: number[]; pca_cumulative: number[]; components: string[];
    loadings: { assets: string[]; matrix: number[][] };
    effective_bets: number; n_assets: number;
    vif: { factor: string; vif: number }[];
    factor_correlation: { labels: string[]; matrix: number[][] };
  };
  diagnostics: { name: string; stat: number; pvalue: number; verdict: string; licenses: string }[];
}

export interface Holding { ticker: string; weight: number }
export interface AnalyzeRequest {
  holdings: Holding[];
  start: string;
  end: string;
  estimator?: string;
}
