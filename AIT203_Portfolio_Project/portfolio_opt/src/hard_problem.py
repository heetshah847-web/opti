"""[P2] Real-world portfolio problem (MIQP) and the independent enumeration verifier.

Assumption: the one-time trading cost in [C4] is charged against the ANNUAL
return, i.e. the portfolio is assumed to be held for about one year.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import cvxpy as cp
import numpy as np

import config
from .easy_problem import Solution, quad_obj
from .stats import Estimates

MAX_SUBSETS = 300_000


def trade_cost(w: np.ndarray, w0: np.ndarray, c: float) -> float:
    """[M10] cost actually paid = c * sum_i |w_i - w0_i| (from final weights)."""
    return float(c * np.abs(w - w0).sum())


@dataclass
class Feasibility:
    feasible: bool
    message: str
    max_return_gross: float
    max_return_net: float


def check_feasibility(est: Estimates, p: config.HardParams, r_target: float,
                      w0: np.ndarray) -> Feasibility:
    """Pre-solve checks: K*u >= 1, K*l <= 1 and a greedy bound on the best return.

    Greedy: fill the highest-mu stocks up to u, at most K of them. Its gross return
    upper-bounds any feasible return, so r_target above it is certainly infeasible.
    The greedy portfolio net of costs is a feasible point (a lower bound on the best
    net return); a target between the two is left to the solver.
    """
    if p.K * p.u < 1 - 1e-12:
        return Feasibility(False, f"infeasible: K*u = {p.K * p.u:.3f} < 1", 0.0, 0.0)
    if p.K * p.l > 1 + 1e-12:
        return Feasibility(False, f"infeasible: K*l = {p.K * p.l:.3f} > 1", 0.0, 0.0)
    order = np.argsort(-est.mu)
    w = np.zeros(est.n)
    left = 1.0
    for i in order[: p.K]:
        w[i] = min(p.u, left)
        left -= w[i]
        if left <= 1e-15:
            break
    gross = float(est.mu @ w)
    net = gross - trade_cost(w, w0, p.c)
    if r_target > gross + 1e-12:
        return Feasibility(False, f"infeasible: max achievable return is {net:.4f}"
                                  f" (before costs {gross:.4f})", gross, net)
    return Feasibility(True, "ok", gross, net)


def _constraints(w, b, s, z_or_mask, est, p, w0, r_target):
    return [
        cp.sum(w) == 1,                                         # [C1]
        est.mu @ w - p.c * cp.sum(b + s) >= r_target,           # [C4]
        w - w0 == b - s,                                        # [C5]
        w >= p.l * z_or_mask,                                   # [C6]
        w <= p.u * z_or_mask,                                   # [C6]
    ]


def solve_p2_miqp(est: Estimates, p: config.HardParams, r_target: float, w0: np.ndarray,
                  solver: str = "SCIP", name: str = "P2") -> Solution:
    """[P2] mixed-integer QP with binary z; solved by SCIP (or Gurobi)."""
    feas = check_feasibility(est, p, r_target, w0)
    if not feas.feasible:
        return Solution(name, "infeasible", solver, r_target=r_target, message=feas.message)
    if solver.upper() not in cp.installed_solvers():
        raise RuntimeError(f"Solver {solver} is not installed (available: {cp.installed_solvers()})")
    n = est.n
    w = cp.Variable(n)
    b = cp.Variable(n, nonneg=True)
    s = cp.Variable(n, nonneg=True)
    z = cp.Variable(n, boolean=True)
    cons = _constraints(w, b, s, z, est, p, w0, r_target)
    cons.append(cp.sum(z) <= p.K)                                # [C7]
    prob = cp.Problem(cp.Minimize(quad_obj(w, est.sigma)), cons)  # [P2]
    kwargs = {}
    if solver.upper() == "SCIP":
        kwargs["scip_params"] = {"limits/gap": 1e-10, "limits/absgap": 1e-12,
                                 "numerics/feastol": 1e-9}
    t0 = time.perf_counter()
    try:
        prob.solve(solver=getattr(cp, solver.upper()), **kwargs)
    except cp.SolverError as exc:
        return Solution(name, "solver_error", solver, r_target=r_target,
                        runtime=time.perf_counter() - t0, message=str(exc))
    rt = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return Solution(name, prob.status, solver, runtime=rt, r_target=r_target,
                        message=f"P2 solver returned status {prob.status}")
    wv = np.asarray(w.value, dtype=float)
    wv = np.where(np.abs(wv) < 1e-9, 0.0, wv)
    zv = np.round(np.asarray(z.value, dtype=float))
    return Solution(name, prob.status, solver.upper(), wv, float(wv @ est.sigma @ wv), zv, {},
                    rt, r_target)


def solve_p2_enum(est: Estimates, p: config.HardParams, r_target: float, w0: np.ndarray,
                  name: str = "P2 (enumeration)") -> Solution:
    """Exact check: for every support S with ceil(1/u) <= |S| <= K fix z=1 on S, z=0 elsewhere,
    solve the convex QP ([C1],[C4],[C5],[C6]) and keep the best feasible one."""
    feas = check_feasibility(est, p, r_target, w0)
    if not feas.feasible:
        return Solution(name, "infeasible", "CLARABEL", r_target=r_target, message=feas.message)
    n = est.n
    k_min = max(1, math.ceil(1.0 / p.u - 1e-12))
    sizes = range(k_min, min(p.K, n) + 1)
    total = sum(math.comb(n, k) for k in sizes)
    if total > MAX_SUBSETS:
        return Solution(name, "too_many_subsets", "CLARABEL", r_target=r_target,
                        message=f"{total} subsets exceed the limit of {MAX_SUBSETS}")
    w = cp.Variable(n)
    b = cp.Variable(n, nonneg=True)
    s = cp.Variable(n, nonneg=True)
    mask = cp.Parameter(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(quad_obj(w, est.sigma)),
                      _constraints(w, b, s, mask, est, p, w0, r_target))
    best_val, best_w, best_set = np.inf, None, None
    solved = 0
    t0 = time.perf_counter()
    for k in sizes:
        for support in combinations(range(n), k):
            m = np.zeros(n)
            m[list(support)] = 1.0
            mask.value = m
            try:
                prob.solve(solver=cp.CLARABEL)
            except cp.SolverError:
                continue
            solved += 1
            if prob.status in ("optimal", "optimal_inaccurate") and prob.value < best_val:
                best_val, best_w, best_set = prob.value, np.asarray(w.value, float), support
    rt = time.perf_counter() - t0
    sol = Solution(name, "infeasible" if best_w is None else "optimal", "CLARABEL",
                   runtime=rt, r_target=r_target,
                   message=f"{solved} of {total} subsets solved")
    if best_w is not None:
        best_w = np.where(np.abs(best_w) < 1e-9, 0.0, best_w)
        z = np.zeros(n)
        z[list(best_set)] = 1.0
        sol.w, sol.z = best_w, z
        sol.variance = float(best_w @ est.sigma @ best_w)
    sol.duals = {"n_subsets": total, "n_solved": solved}
    return sol
