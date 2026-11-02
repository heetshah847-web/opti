"""Model tests on a small synthetic dataset (fixed seed) plus one on the cached real data."""
import numpy as np
import pandas as pd
import pytest

import config
from src import baselines as bl
from src import data as data_mod
from src import easy_problem as ep
from src import hard_problem as hp
from src import stats, verify

TOL = config.TOL
N = 8


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(config.RANDOM_SEED)
    T = 500
    market = rng.normal(0.0004, 0.01, size=(T, 1))
    betas = rng.uniform(0.5, 1.5, size=(1, N))
    drift = rng.uniform(0.0, 0.0012, size=(1, N))
    r = market @ betas + rng.normal(0, 0.012, size=(T, N)) + drift
    return pd.DataFrame(r, columns=[f"S{i}" for i in range(N)],
                        index=pd.bdate_range("2022-01-03", periods=T))


@pytest.fixture(scope="module")
def est(synth):
    return stats.estimate(synth)


@pytest.fixture(scope="module")
def setup(est):
    w_ew = bl.equal_weight(est)
    return w_ew, float(est.mu @ w_ew)


P = config.HardParams(K=3, l=0.02, u=0.5, c=0.001)


def test_p1_constraints(est, setup):
    _, r = setup
    sol = ep.solve_p1(est, r)
    assert sol.ok
    rows = verify.verify_p1(sol, est, r)
    assert all(row["status"] == "PASS" for row in rows), rows


def test_p2_constraints(est, setup):
    w_ew, r = setup
    sol = hp.solve_p2_miqp(est, P, r, w_ew)
    assert sol.ok
    rows = verify.verify_p2(sol, est, P, r, w_ew)
    assert all(row["status"] == "PASS" for row in rows), rows


def test_p1_beats_equal_weight(est, setup):
    w_ew, r = setup
    sol = ep.solve_p1(est, r)
    assert sol.variance <= w_ew @ est.sigma @ w_ew + 1e-12


def test_p2_not_better_than_p1(est, setup):
    w_ew, r = setup
    assert hp.solve_p2_miqp(est, P, r, w_ew).variance >= ep.solve_p1(est, r).variance - 1e-9


def test_p2_relaxed_equals_p1(est, setup):
    w_ew, r = setup
    relaxed = config.HardParams(K=N, l=0.0, u=1.0, c=0.0)
    p1 = ep.solve_p1(est, r)
    p2 = hp.solve_p2_miqp(est, relaxed, r, w_ew)
    assert p2.ok
    assert abs(p2.variance - p1.variance) / p1.variance < TOL


def test_miqp_matches_enumeration(est, setup):
    w_ew, r = setup
    miqp = hp.solve_p2_miqp(est, P, r, w_ew)
    enum = hp.solve_p2_enum(est, P, r, w_ew)
    assert miqp.ok and enum.status == "optimal"
    assert abs(miqp.variance - enum.variance) / enum.variance < TOL


def test_risk_contributions_sum_to_one(est, setup):
    w = ep.solve_p1(est, setup[1]).w
    assert abs(stats.risk_contributions(w, est.sigma).sum() - 1) < 1e-9


def test_sigma_symmetric_psd(est):
    assert np.allclose(est.sigma, est.sigma.T)
    assert np.linalg.eigvalsh(est.sigma).min() >= -1e-12


def test_infeasible_target_is_reported_not_raised(est, setup):
    w_ew, _ = setup
    sol = hp.solve_p2_miqp(est, P, est.mu.max() * 2, w_ew)
    assert not sol.ok and "infeasible" in sol.message


@pytest.mark.skipif(not list(config.DATA_DIR.glob("raw_prices_*.csv")), reason="no cached data")
def test_real_data_pipeline():
    raw = sorted(config.DATA_DIR.glob("raw_prices_*.csv"))[0]
    start, end = raw.stem.replace("raw_prices_", "").split("_")
    prices, info = data_mod.load_prices(start, end)
    assert prices.shape[1] == 18
    est_r = stats.estimate(stats.daily_returns(prices))
    r = float(est_r.mu @ bl.equal_weight(est_r))
    p1 = ep.solve_p1(est_r, r)
    w_ew = bl.equal_weight(est_r)
    assert p1.variance <= w_ew @ est_r.sigma @ w_ew + 1e-12
    p2 = hp.solve_p2_miqp(est_r, config.HardParams(), r, w_ew)
    assert p2.ok and p2.variance >= p1.variance - 1e-9
