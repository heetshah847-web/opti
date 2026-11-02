"""Baselines B1-B4 used for the 'impact of optimisation' comparison."""
from __future__ import annotations

from typing import Dict

import numpy as np

import config
from .easy_problem import min_variance
from .stats import Estimates


def equal_weight(est: Estimates) -> np.ndarray:
    """B1: w_i = 1/N."""
    return np.full(est.n, 1.0 / est.n)


def best_single_stock(est: Estimates) -> np.ndarray:
    """B2: 100% in the stock with the highest annual mean return (hindsight)."""
    w = np.zeros(est.n)
    w[int(np.argmax(est.mu))] = 1.0
    return w


def familiar_names(est: Estimates) -> np.ndarray:
    """B3: equal weight across AAPL, MSFT, NVDA, GOOGL, META, AMZN."""
    w = np.zeros(est.n)
    idx = [est.tickers.index(t) for t in config.FAMILIAR_NAMES if t in est.tickers]
    w[idx] = 1.0 / len(idx)
    return w


def global_min_variance(est: Estimates) -> np.ndarray:
    """B4: P1 without [C2]."""
    sol = min_variance(est)
    if not sol.ok:
        raise RuntimeError(f"Minimum-variance problem failed: {sol.status}")
    return sol.w


def all_baselines(est: Estimates) -> Dict[str, np.ndarray]:
    return {
        "B1 Equal weight": equal_weight(est),
        "B2 Best single stock": best_single_stock(est),
        "B3 Familiar names": familiar_names(est),
        "B4 Min-variance": global_min_variance(est),
    }
