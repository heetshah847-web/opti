"""In-sample curves and the honest out-of-sample (train/test) evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from . import baselines as bl
from . import easy_problem as ep
from . import hard_problem as hp
from . import stats


def wealth_table(returns: pd.DataFrame, portfolios: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Growth of $1 per portfolio (constant weights, i.e. rebalanced daily), starting at 1."""
    cols = {}
    for name, w in portfolios.items():
        cols[name] = stats.wealth_curve(stats.portfolio_returns(returns, w))
    out = pd.DataFrame(cols)
    start = pd.DataFrame([np.ones(out.shape[1])], columns=out.columns,
                         index=[returns.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, out])


def drawdown_table(wealth: pd.DataFrame) -> pd.DataFrame:
    return wealth / wealth.cummax() - 1.0


def realized_metrics(returns: pd.DataFrame, portfolios: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {}
    for name, w in portfolios.items():
        rp = stats.portfolio_returns(returns, w)
        rows[name] = {
            "realized_return_ann": float(rp.mean() * config.TRADING_DAYS),
            "realized_volatility_ann": float(rp.std(ddof=1) * np.sqrt(config.TRADING_DAYS)),
            "max_drawdown": stats.max_drawdown(rp),
            "total_return": float((1 + rp).prod() - 1),
            "growth_of_1": float((1 + rp).prod()),
        }
    return pd.DataFrame(rows).T


@dataclass
class OutOfSample:
    weights: Dict[str, np.ndarray]
    metrics: pd.DataFrame
    wealth: pd.DataFrame
    notes: list = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    r_target: Optional[float] = None


def out_of_sample(returns: pd.DataFrame, cfg: config.Config, w0: Optional[np.ndarray] = None
                  ) -> OutOfSample:
    """Estimate mu, Sigma on the train window only, solve P1/P2, evaluate on the test window."""
    train = returns.loc[cfg.train_start:cfg.train_end]
    test = returns.loc[cfg.test_start:cfg.test_end]
    if len(train) < 60 or len(test) < 20:
        raise ValueError(f"Train/test windows too short ({len(train)} / {len(test)} days).")
    est = stats.estimate(train)
    w0 = np.full(est.n, 1.0 / est.n) if w0 is None else w0
    port = bl.all_baselines(est)
    r_target = float(est.mu @ port["B1 Equal weight"])      # same default rule, train data only
    notes = []
    s1 = ep.solve_p1(est, r_target)
    if s1.ok:
        port["P1 Easy"] = s1.w
    else:
        notes.append(f"P1 failed on the training window: {s1.message}")
    s2 = hp.solve_p2_miqp(est, cfg.hard, r_target, w0, cfg.solver)
    if s2.ok:
        port["P2 Hard"] = s2.w
    else:
        notes.append(f"P2 not available on the training window: {s2.message or s2.status}")
    return OutOfSample(port, realized_metrics(test, port), wealth_table(test, port), notes,
                       len(train), len(test), r_target)
