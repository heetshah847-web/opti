"""Single entry point that runs the whole analysis; used by both run_all.py and app.py."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import config
from . import backtest as bt
from . import baselines as bl
from . import data as data_mod
from . import easy_problem as ep
from . import frontier as fr
from . import hard_problem as hp
from . import stats
from . import verify as vf

log = logging.getLogger(__name__)


@dataclass
class Results:
    cfg: config.Config
    info: dict
    prices: pd.DataFrame
    returns: pd.DataFrame
    est: stats.Estimates
    stock_stats: pd.DataFrame
    w0: np.ndarray
    r_target: float
    r_target_range: tuple
    baselines: Dict[str, np.ndarray]
    p1: ep.Solution
    p2: ep.Solution
    p2_enum: Optional[ep.Solution]
    feasibility: hp.Feasibility
    portfolios: Dict[str, np.ndarray]
    comparison: pd.DataFrame
    weights: pd.DataFrame
    risk_contrib: pd.DataFrame
    impact: Dict[str, float]
    frontier_p1: Optional[pd.DataFrame]
    frontier_p2: Optional[pd.DataFrame]
    wealth: pd.DataFrame
    oos: Optional[bt.OutOfSample]
    verification: pd.DataFrame
    warnings: list = field(default_factory=list)


def default_r_target(est: stats.Estimates) -> float:
    """Default target = expected annual return of the equal-weight portfolio (B1)."""
    return float(est.mu @ bl.equal_weight(est))


def prepare(cfg: config.Config):
    """Load data and estimate inputs. Cheap enough to reuse for the GUI sidebar."""
    prices, info = data_mod.load_prices(cfg.start, cfg.end, refresh=cfg.refresh)
    returns = stats.daily_returns(prices)
    return prices, info, returns, stats.estimate(returns)


def run_pipeline(cfg: config.Config, with_frontier: bool = True,
                 progress: Optional[Callable[[str], None]] = None) -> Results:
    say = progress or (lambda m: log.info(m))
    warnings = []
    say("Loading data")
    prices, info, returns, est = prepare(cfg)
    if est.ridge_applied:
        warnings.append(f"Sigma had min eigenvalue {est.min_eig:.2e} < 0; ridge "
                        f"{config.RIDGE:.0e}*I added.")
    n = est.n
    w0 = np.full(n, 1.0 / n) if cfg.w0 is None else np.asarray(cfg.w0, dtype=float)
    if w0.shape != (n,) or abs(w0.sum() - 1) > 1e-6 or (w0 < 0).any():
        raise ValueError("Custom starting portfolio w0 must be non-negative and sum to 1.")

    base = bl.all_baselines(est)
    r_target = cfg.r_target if cfg.r_target is not None else default_r_target(est)
    if r_target > est.mu.max() + 1e-12:
        raise ValueError(f"r_target {r_target:.4f} exceeds the largest annual mean return "
                         f"{est.mu.max():.4f}; no long-only split can reach it.")

    say("Solving P1 (convex QP)")
    p1 = ep.solve_p1(est, r_target)
    if not p1.ok:
        raise RuntimeError(p1.message)

    say("Solving P2 (MIQP)")
    feas = hp.check_feasibility(est, cfg.hard, r_target, w0)
    p2 = hp.solve_p2_miqp(est, cfg.hard, r_target, w0, cfg.solver)
    if not p2.ok:
        warnings.append(f"P2: {p2.message or p2.status}")
    p2_enum = None
    if cfg.run_enumeration and p2.ok:
        say("Verifying P2 by enumeration")
        p2_enum = hp.solve_p2_enum(est, cfg.hard, r_target, w0)
        if p2_enum.status == "too_many_subsets":
            warnings.append(f"Enumeration skipped: {p2_enum.message}")

    portfolios = dict(base)
    portfolios["P1 Easy"] = p1.w
    if p2.ok:
        portfolios["P2 Hard"] = p2.w

    comp = pd.DataFrame({name: stats.portfolio_metrics(w, est, returns, cfg.rf, w0, cfg.hard.c)
                         for name, w in portfolios.items()}).T
    weights = pd.DataFrame(portfolios, index=est.tickers)
    rc = pd.DataFrame({name: stats.risk_contributions(w, est.sigma)
                       for name, w in portfolios.items()}, index=est.tickers)

    # impact metrics
    w_ew = base["B1 Equal weight"]
    vol_ew = float(np.sqrt(w_ew @ est.sigma @ w_ew))
    impact: Dict[str, float] = {
        "risk_reduction_pct": 100 * (vol_ew - p1.volatility) / vol_ew,
        "vol_ew": vol_ew, "vol_p1": p1.volatility,
    }
    gain = ep.max_return_at_variance(est, vol_ew ** 2)
    if gain.ok:
        impact["return_ew"] = float(est.mu @ w_ew)
        impact["return_max_at_ew_risk"] = float(est.mu @ gain.w)
        impact["return_gain"] = impact["return_max_at_ew_risk"] - impact["return_ew"]
    if p2.ok:
        impact.update(fr.price_of_realism(p1, p2))

    f1 = f2 = None
    if with_frontier:
        say("Computing efficient frontiers")
        f1, f2 = fr.efficient_frontier(est, cfg.hard, w0, cfg.n_frontier, cfg.solver)

    oos = None
    try:
        say("Out-of-sample evaluation")
        oos = bt.out_of_sample(returns, cfg, w0)
        warnings.extend(oos.notes)
    except ValueError as exc:
        warnings.append(f"Out-of-sample check skipped: {exc}")

    say("Verification")
    rows = vf.verify_p1(p1, est, r_target)
    if p2.ok:
        rows += vf.verify_p2(p2, est, cfg.hard, r_target, w0)
    rows += vf.cross_checks(est, p1, p2, p2_enum, w_ew)
    verification = vf.to_frame(rows)

    return Results(
        cfg=cfg, info=info, prices=prices, returns=returns, est=est,
        stock_stats=stats.stock_stats(returns, est, cfg.rf), w0=w0, r_target=r_target,
        r_target_range=(float(est.mu @ base["B4 Min-variance"]), float(est.mu.max())),
        baselines=base, p1=p1, p2=p2, p2_enum=p2_enum, feasibility=feas, portfolios=portfolios,
        comparison=comp, weights=weights, risk_contrib=rc, impact=impact,
        frontier_p1=f1, frontier_p2=f2, wealth=bt.wealth_table(returns, portfolios), oos=oos,
        verification=verification, warnings=warnings)
