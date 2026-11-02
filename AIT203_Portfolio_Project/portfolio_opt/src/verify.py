"""Constraint checks, KKT check and cross-checks, exported as a PASS/FAIL table."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

import config
from .easy_problem import Solution
from .stats import Estimates

TOL = config.TOL


def _row(model: str, check: str, residual: float, tol: float = TOL, detail: str = "") -> dict:
    ok = bool(np.isfinite(residual) and residual <= tol)
    return {"model": model, "check": check, "residual": float(residual), "tolerance": tol,
            "status": "PASS" if ok else "FAIL", "detail": detail}


def verify_p1(sol: Solution, est: Estimates, r_target: float) -> List[dict]:
    w = sol.w
    rows = [
        _row("P1", "[C1] sum(w) = 1", abs(w.sum() - 1)),
        _row("P1", "[C2] mu'w >= r_target", max(0.0, r_target - est.mu @ w)),
        _row("P1", "[C3] w >= 0", max(0.0, -w.min())),
    ]
    # KKT: stationarity of  L = w'Sw + lam (r - mu'w) + nu (1'w - 1) - eta'w , using cvxpy duals:
    #   2 Sigma w - lam*mu + nu*1 - eta = 0   (lam, eta >= 0)
    lam, nu, eta = sol.duals["C2"], sol.duals["C1"], sol.duals["C3"]
    stat = 2 * est.sigma @ w - lam * est.mu + nu * np.ones(est.n) - eta
    rows.append(_row("P1", "KKT stationarity |2Sw - lam*mu + nu*1 - eta|_inf", np.abs(stat).max(),
                     detail=f"lam={lam:.6g}, nu={nu:.6g}"))
    rows.append(_row("P1", "KKT dual feasibility (lam, eta >= 0)",
                     max(0.0, -lam, float(-eta.min()))))
    comp = max(abs(lam * (r_target - est.mu @ w)), float(np.abs(eta * w).max()))
    rows.append(_row("P1", "KKT complementary slackness", comp))
    return rows


def verify_p2(sol: Solution, est: Estimates, p: config.HardParams, r_target: float,
              w0: np.ndarray) -> List[dict]:
    w, z = sol.w, sol.z
    cost = p.c * np.abs(w - w0).sum()
    rows = [
        _row("P2", "[C1] sum(w) = 1", abs(w.sum() - 1)),
        _row("P2", "[C4] mu'w - cost >= r_target", max(0.0, r_target - (est.mu @ w - cost)),
             detail=f"net return {est.mu @ w - cost:.6f}, cost paid {cost:.6f}"),
        _row("P2", "[C6] l*z <= w", max(0.0, float((p.l * z - w).max()))),
        _row("P2", "[C6] w <= u*z", max(0.0, float((w - p.u * z).max()))),
        _row("P2", "[C7] sum(z) <= K", max(0.0, z.sum() - p.K)),
        _row("P2", "z binary", float(np.abs(z - np.round(z)).max())),
    ]
    return rows


def cross_checks(est: Estimates, p1: Solution, p2: Optional[Solution], p2_enum: Optional[Solution],
                 w_equal: np.ndarray) -> List[dict]:
    rows = []
    var_ew = float(w_equal @ est.sigma @ w_equal)
    ok_tgt = est.mu @ w_equal >= p1.r_target - TOL           # B1 feasible for P1 at this target
    if ok_tgt:
        rows.append(_row("cross", "var*(P1) <= var(equal weight)", max(0.0, p1.variance - var_ew),
                         detail=f"P1 {p1.variance:.6f} vs EW {var_ew:.6f}"))
    if p2 is not None and p2.ok:
        rows.append(_row("cross", "var*(P2) >= var*(P1)", max(0.0, p1.variance - p2.variance),
                         detail=f"P2 {p2.variance:.6f} vs P1 {p1.variance:.6f}"))
    if p2 is not None and p2.ok and p2_enum is not None and p2_enum.status == "optimal":
        rel = abs(p2.variance - p2_enum.variance) / p2_enum.variance
        rows.append(_row("cross", "P2 MIQP optimum = enumeration optimum (rel.)", rel,
                         detail=f"MIQP {p2.variance:.8f}, enum {p2_enum.variance:.8f}, "
                                f"{p2_enum.duals.get('n_solved', '?')} subsets, "
                                f"{p2_enum.runtime:.1f}s"))
    return rows


def to_frame(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
