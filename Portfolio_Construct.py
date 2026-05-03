from pathlib import Path

from Defensive import get_defensive_signal
from Carry     import get_carry_signal
from Momentum  import get_momentum_signal
from Value     import get_value_signal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_DIR         = Path("Data")
FUNDAMENTALS_FILE = DATA_DIR / "sp500_fundamentals.csv"
STOCKS_FILE       = DATA_DIR / "sp500_stocks.csv"


#EPO uses shrinkage to reduce noise (relative to MVO)

def _apply_epo_shrinkage(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """
    Shrink sample correlation matrix toward identity:
      C_EPO = (1 - w) * C_sample + w * I
    then reconstruct: Σ_EPO = diag(σ) · C_EPO · diag(σ)
    """
    stds = np.sqrt(np.diag(cov))
    stds = np.where(stds == 0, 1e-10, stds)
    D_inv = np.diag(1.0 / stds)
    corr  = D_inv @ cov @ D_inv
    corr_shrunk = (1 - shrinkage) * corr + shrinkage * np.eye(len(stds))
    D = np.diag(stds)
    return D @ corr_shrunk @ D

#find optimal portfolio weights

def _optimize(
    mu: np.ndarray,
    cov_epo: np.ndarray,
    lam: float,
    max_pos: float,
    dollar_neutral: bool,
    tc_cost: float,
    x_prev: np.ndarray,
) -> np.ndarray:
    """
    Solve:  max  xᵀμ  -  (λ/2) xᵀ Σ_EPO x  -  cᵀ|x - x_prev|
    subject to: Σx = 0 (dollar-neutral) or Σx = 1 (long-only)
                |x_i| ≤ max_pos
    Transaction-cost term uses a smooth L1 approximation (√(t² + ε²) - ε).
    """
    n   = len(mu)
    eps = 1e-6  # smoothing for |x - x_prev|

    def obj(x):
        tc   = tc_cost * np.sum(np.sqrt((x - x_prev) ** 2 + eps) - eps)
        return -x @ mu + (lam / 2) * (x @ cov_epo @ x) + tc

    def jac(x):
        dtc = tc_cost * (x - x_prev) / np.sqrt((x - x_prev) ** 2 + eps)
        return -mu + lam * (cov_epo @ x) + dtc

    bounds      = [(-max_pos, max_pos)] * n
    sum_target  = 0.0 if dollar_neutral else 1.0
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - sum_target,
                    "jac": lambda _: np.ones(n)}]

    x0  = np.zeros(n)
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-12, "maxiter": 2000})
    return res.x

#lambda work

def _calibrate_lambda(mu: np.ndarray, cov_epo: np.ndarray, target_vol: float) -> float:
    """
    Derive λ so that the unconstrained EPO solution has volatility ≈ target_vol.
    Closed form: x* = (1/λ) Σ⁻¹μ  →  σ(x*) = (1/λ)√(μᵀΣ⁻¹μ)  →  λ = √(μᵀΣ⁻¹μ) / σ_target
    """
    try:
        cov_inv = np.linalg.inv(cov_epo)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_epo)
    ir_sq = float(mu @ cov_inv @ mu)
    return np.sqrt(max(ir_sq, 1e-12)) / target_vol


#find exposure to each factor signal

def _factor_attribution(weights: np.ndarray, signals: pd.DataFrame) -> pd.Series:
    """
    Net exposure of the portfolio to each factor signal:
      exposure_k = xᵀ s_k / ‖s_k‖
    """
    exposures = {}
    for col in signals.columns:
        s = signals[col].to_numpy(dtype=float)
        norm = np.linalg.norm(s)
        exposures[col] = float(weights @ s) / norm if norm > 0 else 0.0
    return pd.Series(exposures)


#create actual portfolio

def create_portfolio(
    stocks_CSV,
    fundamentals_CSV,
    shrinkage:      float = 0.75,
    target_vol:     float = 0.10,
    max_pos:        float = 0.05,
    lookback_days:  int   = 756,
    dollar_neutral: bool  = True,
    factor_weights: dict | None = None,
    tc_cost:        float = 0.001,
    x_prev:         pd.Series | None = None,
):
    # factor signals
    signals_raw = {
        "value":     get_value_signal(fundamentals_CSV),
        "carry":     get_carry_signal(fundamentals_CSV),
        "defensive": get_defensive_signal(fundamentals_CSV),
        "momentum":  get_momentum_signal(stocks_CSV),
    }
    signals = pd.DataFrame(signals_raw)          # rows = symbols, cols = factors
    signals = signals.dropna(how="all")

    # factor -> alpha vector
    if factor_weights is None:
        factor_weights = {k: 0.25 for k in signals.columns}

    # weighted sum of z-scored signals; re-z-score the composite
    alpha: pd.Series = pd.Series(0.0, index=signals.index)
    for k in factor_weights:
        if k in signals.columns:
            alpha = alpha.add(factor_weights[k] * signals[k], fill_value=0.0)
    mu_raw = alpha.dropna()
    # re-standardise composite signal (cross-sectional)
    sd = mu_raw.std()
    if sd > 0:
        mu_raw = (mu_raw - mu_raw.mean()) / sd

    # estimate cov from rolling returns
    prices = (
        pd.read_csv(stocks_CSV, parse_dates=["Date"])
        .dropna(subset=["Symbol", "Adj Close"])
        .sort_values(["Symbol", "Date"])
    )
    price_wide = prices.pivot_table(index="Date", columns="Symbol", values="Adj Close")
    price_wide = price_wide.iloc[-lookback_days:].dropna(axis=1)

    # intersect: stocks that have both a signal and full price history
    common = mu_raw.index.intersection(price_wide.columns)
    if len(common) < 10:
        raise ValueError(
            f"Only {len(common)} stocks overlap signals and price history — "
            "check data or widen lookback."
        )

    price_wide    = price_wide[common]
    daily_returns = price_wide.pct_change().dropna()

    mu  = mu_raw[common].to_numpy(dtype=float)
    cov = daily_returns.cov().to_numpy(dtype=float) * 252
    symbols = list(common)

    # ── Step 5: EPO shrinkage ──────────────────────────────────────────────────
    cov_epo = _apply_epo_shrinkage(cov, shrinkage)

    # ── Step 6: calibrate λ and optimise ──────────────────────────────────────
    lam = _calibrate_lambda(mu, cov_epo, target_vol)

    prev_w = (
        x_prev.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
        if x_prev is not None
        else np.zeros(len(symbols))
    )

    weights = _optimize(mu, cov_epo, lam, max_pos, dollar_neutral, tc_cost, prev_w)

    # ── Step 7: factor attribution ─────────────────────────────────────────────
    signal_matrix = signals.reindex(symbols).fillna(0.0)
    exposures = _factor_attribution(weights, signal_matrix)

    # ── Assemble output ────────────────────────────────────────────────────────
    portfolio = (
        pd.DataFrame({"Symbol": symbols, "Weight": weights})
        .assign(Side=lambda df: df["Weight"].map(lambda w: "Long" if w > 1e-6 else ("Short" if w < -1e-6 else "Flat")))
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )

    port_return = float(weights @ mu)
    port_vol    = float(np.sqrt(weights @ cov @ weights))
    sharpe      = port_return / port_vol if port_vol > 0 else np.nan
    gross_exp   = float(np.abs(weights).sum())
    n_long      = int((weights >  1e-6).sum())
    n_short     = int((weights < -1e-6).sum())

    print(f"\n=== EPO Portfolio  (shrinkage={shrinkage}, target_vol={target_vol:.0%}) ===")
    print(f"  Universe          : {len(symbols)} stocks")
    print(f"  Long / Short      : {n_long} / {n_short}")
    print(f"  Gross exposure    : {gross_exp:.2f}x")
    print(f"  Ex-ante return    : {port_return:.2%}")
    print(f"  Ex-ante vol       : {port_vol:.2%}")
    print(f"  Sharpe ratio      : {sharpe:.3f}")
    print(f"\nFactor exposures:")
    print(exposures.to_string())
    print(f"\nTop 10 longs:")
    print(portfolio[portfolio["Weight"] >  1e-6].head(10).to_string(index=False))
    print(f"\nTop 10 shorts:")
    print(portfolio[portfolio["Weight"] < -1e-6].head(10).to_string(index=False))

    stats = {
        "return": port_return, "vol": port_vol, "sharpe": sharpe,
        "gross_exposure": gross_exp, "factor_exposures": exposures,
    }
    return portfolio, stats


if __name__ == "__main__":
    portfolio, stats = create_portfolio(
        STOCKS_FILE, FUNDAMENTALS_FILE,
        shrinkage=0.75,
        target_vol=0.10,
        max_pos=0.05,
        dollar_neutral=True,
    )
