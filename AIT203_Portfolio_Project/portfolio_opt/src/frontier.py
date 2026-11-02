"""Efficient frontiers of P1 and P2 and the price of realism [M9]."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from . import easy_problem as ep
from . import hard_problem as hp
from .stats import Estimates


def price_of_realism(p1: ep.Solution, p2: ep.Solution) -> Dict[str, float]:
    """[M9] Delta_var = var*(P2) - var*(P1); Delta_vol likewise (annualised); also in %."""
    d_var = p2.variance - p1.variance
    d_vol = p2.volatility - p1.volatility
    return {"delta_var": d_var, "delta_var_pct": 100 * d_var / p1.variance,
            "delta_vol": d_vol, "delta_vol_pct": 100 * d_vol / p1.volatility}


def efficient_frontier(est: Estimates, p: config.HardParams, w0: np.ndarray, n: int = 40,
                       solver: str = "SCIP") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep r_target from the min-variance return to the largest feasible return.

    P1's range ends at max(mu) (all in one stock). P2 points above its own maximum
    achievable return are kept in the table but marked infeasible.
    """
    mv = ep.min_variance(est)
    r_lo = float(est.mu @ mv.w)
    r_hi = float(est.mu.max())
    grid = np.linspace(r_lo, r_hi, n)
    rows1, rows2 = [], []
    for r in grid:
        s1 = ep.solve_p1(est, r)
        rows1.append({"r_target": r, "feasible": s1.ok, "variance": s1.variance,
                      "volatility": s1.volatility, "return": float(est.mu @ s1.w) if s1.ok else np.nan,
                      "n_held": int((s1.w > config.HELD_THRESHOLD).sum()) if s1.ok else 0})
        s2 = hp.solve_p2_miqp(est, p, r, w0, solver)
        rows2.append({"r_target": r, "feasible": s2.ok, "variance": s2.variance,
                      "volatility": s2.volatility, "return": float(est.mu @ s2.w) if s2.ok else np.nan,
                      "n_held": int(s2.z.sum()) if s2.ok else 0, "status": s2.status})
    f1, f2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    f2["delta_vol"] = f2["volatility"] - f1["volatility"]
    return f1, f2
