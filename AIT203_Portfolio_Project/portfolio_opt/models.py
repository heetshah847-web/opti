"""P1 (mean-variance QP) and P2 (cardinality + buy-in limits + trading costs).

P2 is a MIQP. No integer-capable QP solver ships with cvxpy by default, so it is
solved exactly by enumerating every support set of size <= K and solving a convex
QP for each one; the best QP wins.
"""
from itertools import combinations

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm


def _psd_factor(sigma):
    """Factor F with F.T @ F ~= sigma, so w'Σw = ||F w||^2 (avoids PSD-check noise)."""
    sym = (sigma + sigma.T) / 2
    return np.real(sqrtm(sym))


def solve_p1(mu, sigma, r_target):
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(_psd_factor(sigma) @ w)),
        [cp.sum(w) == 1, mu @ w >= r_target, w >= 0],
    )
    prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return {"w": np.clip(w.value, 0, None), "variance": prob.value}


class P2Solver:
    """Reusable P2 solver: the QP is built once with the support as a parameter."""

    def __init__(self, mu, sigma, K=5, l=0.02, u=0.40, c=0.001, w0=None):
        n = len(mu)
        self.n, self.mu, self.K = n, mu, K
        self.w0 = np.full(n, 1 / n) if w0 is None else w0
        self.c = c
        self.w = cp.Variable(n)
        self.mask = cp.Parameter(n, nonneg=True)
        self.r_target = cp.Parameter()
        cost = c * cp.norm1(self.w - self.w0)  # = c * sum(b + s)
        self.prob = cp.Problem(
            cp.Minimize(cp.sum_squares(_psd_factor(sigma) @ self.w)),
            [
                cp.sum(self.w) == 1,
                mu @ self.w - cost >= self.r_target,
                self.w >= l * self.mask,
                self.w <= u * self.mask,
            ],
        )

    def solve(self, r_target):
        self.r_target.value = r_target
        best = None
        for k in range(1, self.K + 1):
            for support in combinations(range(self.n), k):
                m = np.zeros(self.n)
                m[list(support)] = 1
                self.mask.value = m
                try:
                    self.prob.solve(solver=cp.CLARABEL)
                except cp.SolverError:
                    continue
                if self.prob.status != "optimal":
                    continue
                if best is None or self.prob.value < best["variance"]:
                    best = {"w": self.w.value.copy(), "variance": self.prob.value,
                            "support": support}
        if best is not None:
            best["w"] = np.where(best["w"] < 1e-6, 0.0, best["w"])
        return best


def check_solution(w, mu, sigma, r_target, K=None, l=None, u=None, c=0.0, w0=None):
    """Constraint checks the milestone asks us to report."""
    tol = 1e-6
    checks = {
        "weights sum to 1": abs(w.sum() - 1) < tol,
        "weights >= 0": bool((w >= -tol).all()),
    }
    cost = 0.0 if w0 is None else c * np.abs(w - w0).sum()
    checks["target return met"] = bool(mu @ w - cost >= r_target - tol)
    held = w > tol
    if K is not None:
        checks[f"at most {K} stocks"] = int(held.sum()) <= K
    if l is not None and u is not None:
        checks["held weights within [l, u]"] = bool(
            ((w[held] >= l - tol) & (w[held] <= u + tol)).all())
    return checks
