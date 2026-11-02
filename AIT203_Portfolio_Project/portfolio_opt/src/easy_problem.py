"""[P1] Classic mean-variance QP, and the global minimum-variance reference point."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cvxpy as cp
import numpy as np

from .stats import Estimates


@dataclass
class Solution:
    """Result container shared by P1 and P2."""
    name: str
    status: str
    solver: str
    w: Optional[np.ndarray] = None
    variance: float = float("nan")
    z: Optional[np.ndarray] = None
    duals: Dict[str, np.ndarray] = field(default_factory=dict)
    runtime: float = 0.0
    r_target: Optional[float] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.w is not None and self.status in ("optimal", "optimal_inaccurate")

    @property
    def volatility(self) -> float:
        return float(np.sqrt(self.variance)) if self.ok else float("nan")


def quad_obj(w: cp.Variable, sigma: np.ndarray) -> cp.Expression:
    """Portfolio variance w' Sigma w (Sigma is PSD by construction, see stats.estimate)."""
    return cp.quad_form(w, cp.psd_wrap(sigma))


def solve_p1(est: Estimates, r_target: Optional[float], solver: str = "CLARABEL",
             name: str = "P1") -> Solution:
    """Solve [P1]; with r_target=None the return constraint [C2] is dropped (min-variance)."""
    w = cp.Variable(est.n)
    c1 = cp.sum(w) == 1                                    # [C1] full budget
    c3 = w >= 0                                            # [C3] no short selling
    cons = [c1, c3]
    c2 = None
    if r_target is not None:
        c2 = est.mu @ w >= r_target                        # [C2] target return
        cons.append(c2)
    prob = cp.Problem(cp.Minimize(quad_obj(w, est.sigma)), cons)   # [P1]
    t0 = time.perf_counter()
    used = solver
    for s in (solver, "OSQP"):                             # Clarabel, fall back to OSQP
        try:
            prob.solve(solver=getattr(cp, s))
            used = s
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.SolverError:
            continue
    rt = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return Solution(name, prob.status, used, runtime=rt, r_target=r_target,
                        message=f"P1 failed with status {prob.status}")
    duals = {"C1": float(c1.dual_value), "C3": np.asarray(c3.dual_value)}
    if c2 is not None:
        duals["C2"] = float(c2.dual_value)
    wv = np.asarray(w.value, dtype=float)
    return Solution(name, prob.status, used, wv, float(wv @ est.sigma @ wv), None, duals, rt,
                    r_target)


def min_variance(est: Estimates) -> Solution:
    """Global minimum-variance portfolio (P1 without [C2]); baseline B4."""
    return solve_p1(est, None, name="B4 min-variance")


def max_return_at_variance(est: Estimates, var_cap: float) -> Solution:
    """max mu'w  s.t. w'Sigma w <= var_cap, [C1], [C3].  Used for 'return gain at same risk'."""
    w = cp.Variable(est.n)
    cons = [cp.sum(w) == 1, w >= 0, quad_obj(w, est.sigma) <= var_cap]
    prob = cp.Problem(cp.Maximize(est.mu @ w), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    rt = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return Solution("max-return", prob.status, "CLARABEL", runtime=rt)
    wv = np.asarray(w.value, dtype=float)
    return Solution("max-return", prob.status, "CLARABEL", wv, float(wv @ est.sigma @ wv),
                    runtime=rt)
