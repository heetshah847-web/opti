"""Returns, estimates, annualisation and statistics.  Formulas [F1]-[F5], [M1]-[M8], [M10]."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


@dataclass
class Estimates:
    """Annualised inputs of every optimisation model."""
    tickers: List[str]
    mu_daily: np.ndarray
    sigma_daily: np.ndarray
    mu: np.ndarray            # mu_ann    [F4]
    sigma: np.ndarray         # Sigma_ann [F4], symmetrised, ridge only if documented below
    T: int
    min_eig: float
    ridge_applied: bool

    @property
    def n(self) -> int:
        return len(self.tickers)

    @property
    def vols(self) -> np.ndarray:
        return np.sqrt(np.diag(self.sigma))


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """[F1] r(t,i) = (P(t,i) - P(t-1,i)) / P(t-1,i)."""
    return prices.pct_change().dropna(how="any")


def estimate(returns: pd.DataFrame) -> Estimates:
    """[F2] mu = mean, [F3] sample covariance (ddof=1), [F4] x252."""
    T = len(returns)
    mu_d = returns.mean().to_numpy()                                   # [F2]
    sig_d = returns.cov(ddof=1).to_numpy()                             # [F3]
    mu = config.TRADING_DAYS * mu_d                                    # [F4]
    sigma = config.TRADING_DAYS * sig_d                                # [F4]
    sigma = (sigma + sigma.T) / 2                                      # symmetrise
    min_eig = float(np.linalg.eigvalsh(sigma).min())
    ridge = False
    if min_eig < 0:
        sigma = sigma + config.RIDGE * np.eye(len(mu))
        ridge = True
        log.warning("Sigma had smallest eigenvalue %.3e < 0; added ridge %.0e * I.",
                    min_eig, config.RIDGE)
    return Estimates(list(returns.columns), mu_d, (sig_d + sig_d.T) / 2, mu, sigma, T,
                     min_eig, ridge)


def correlation(sigma: np.ndarray) -> np.ndarray:
    """[F5] rho_ij = Sigma_ij / (sigma_i sigma_j)."""
    s = np.sqrt(np.diag(sigma))
    return sigma / np.outer(s, s)


def wealth_curve(daily: pd.Series) -> pd.Series:
    """Cumulative wealth W(t) = prod(1 + R_p) starting from $1."""
    return (1.0 + daily).cumprod()


def max_drawdown(daily: pd.Series) -> float:
    """[M4] Largest peak-to-trough fall of W(t) (a negative number)."""
    w = pd.concat([pd.Series([1.0]), wealth_curve(daily).reset_index(drop=True)])
    return float((w / w.cummax() - 1.0).min())


def stock_stats(returns: pd.DataFrame, est: Estimates, rf: float = 0.0) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(est.tickers):
        r = returns[t]
        vol = float(np.sqrt(est.sigma[i, i]))
        rows.append({
            "ticker": t, "sector": config.SECTORS.get(t, ""),
            "mean_return_ann": est.mu[i], "volatility_ann": vol,
            "sharpe": (est.mu[i] - rf) / vol, "max_drawdown": max_drawdown(r),
            "best_day": r.max(), "worst_day": r.min(),
        })
    return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------- portfolio level
def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """R_p(t) = sum_i w_i r(t,i)."""
    return pd.Series(returns.to_numpy() @ w, index=returns.index)


def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """[M7] RC_i = w_i (Sigma w)_i / (w' Sigma w); sums to 1."""
    var = float(w @ sigma @ w)
    return w * (sigma @ w) / var


def portfolio_metrics(w: np.ndarray, est: Estimates, returns: pd.DataFrame, rf: float = 0.0,
                      w0: Optional[np.ndarray] = None, c: float = 0.0) -> Dict[str, float]:
    var = float(w @ est.sigma @ w)
    vol = float(np.sqrt(var))                                          # [M2]
    ret = float(est.mu @ w)                                            # [M1]
    rp = portfolio_returns(returns, w)
    q = float(np.quantile(rp, 0.05))
    tail = rp[rp <= q]
    out = {
        "return_ann": ret,                                             # [M1]
        "variance": var, "volatility_ann": vol,                        # [M2]
        "sharpe": (ret - rf) / vol,                                    # [M3]
        "max_drawdown": max_drawdown(rp),                              # [M4]
        "var95_daily": -q,                                             # [M5]
        "cvar95_daily": -float(tail.mean()) if len(tail) else float("nan"),  # [M5]
        "effective_n": 1.0 / float(np.sum(w ** 2)),                    # [M6]
        "n_held": int(np.sum(w > config.HELD_THRESHOLD)),              # [M6]
        "diversification_ratio": float(w @ est.vols / vol),            # [M8]
    }
    if w0 is not None:
        turnover = float(np.abs(w - w0).sum())                         # [M10]
        out["turnover"] = turnover
        out["transaction_cost"] = c * turnover                         # [M10]
    return out
