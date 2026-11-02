"""Run the full analysis without the GUI and write CSVs, PNGs and summary.md to results/.

    python run_all.py [--refresh] [--r-target 0.2] [--solver SCIP] [--no-enum] [--no-png]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import backtest as bt
from src import plots
from src.pipeline import Results, run_pipeline


def export_tables(R: Results, out: Path) -> None:
    est, p = R.est, R.cfg.hard
    R.stock_stats.to_csv(out / "stock_stats.csv")
    w1 = pd.DataFrame({"weight": R.p1.w, "risk_contribution": R.risk_contrib["P1 Easy"]},
                      index=est.tickers)
    w1.index.name = "ticker"
    w1.to_csv(out / "weights_P1.csv")
    if R.p2.ok:
        w2 = pd.DataFrame({"weight": R.p2.w, "z": R.p2.z, "w0": R.w0,
                           "buy": np.clip(R.p2.w - R.w0, 0, None),
                           "sell": np.clip(R.w0 - R.p2.w, 0, None),
                           "risk_contribution": R.risk_contrib["P2 Hard"]}, index=est.tickers)
        w2.index.name = "ticker"
        w2.to_csv(out / "weights_P2.csv")
    comp = R.comparison.copy()
    comp.index.name = "portfolio"
    comp.to_csv(out / "comparison.csv")
    if R.frontier_p1 is not None:
        R.frontier_p1.to_csv(out / "frontier_P1.csv", index=False)
        R.frontier_p2.to_csv(out / "frontier_P2.csv", index=False)
    R.verification.to_csv(out / "verification.csv", index=False)

    ins = bt.realized_metrics(R.returns, R.portfolios).assign(scope="in-sample")
    parts = [ins]
    if R.oos is not None:
        parts.append(R.oos.metrics.assign(scope="out-of-sample"))
    back = pd.concat(parts)
    back.index.name = "portfolio"
    back.to_csv(out / "backtest.csv")


def summary_md(R: Results) -> str:
    est, i = R.est, R.impact
    v = R.verification
    L = [
        "# Optimal Portfolio Allocation: results summary", "",
        f"- Dataset: **{R.info['n_days']} trading days x {R.info['n_stocks']} stocks** "
        f"({R.info['first_date']} to {R.info['last_date']}), source: {R.info['source']}, "
        f"rows dropped: {R.info['rows_dropped']}",
        f"- Returns T = {est.T}; annualised with 252 days; target return r_target = "
        f"{R.r_target:.4%} (equal-weight return unless overridden)", "",
        "## Optimal values", "",
        f"- **P1 (easy QP)**: variance {R.p1.variance:.6f}, volatility {R.p1.volatility:.4%}, "
        f"expected return {est.mu @ R.p1.w:.4%}",
    ]
    L += [f"  - {t}: {w:.2%}" for t, w in zip(est.tickers, R.p1.w) if w > config.HELD_THRESHOLD]
    if R.p2.ok:
        L += [f"- **P2 (hard MIQP, {R.p2.solver})**: variance {R.p2.variance:.6f}, volatility "
              f"{R.p2.volatility:.4%}, expected return {est.mu @ R.p2.w:.4%}"]
        L += [f"  - {t}: {w:.2%}" for t, w in zip(est.tickers, R.p2.w) if w > config.HELD_THRESHOLD]
    else:
        L += [f"- **P2**: not solved ({R.p2.message or R.p2.status})"]
    L += ["", "## Impact metrics", "",
          f"- Risk reduction of P1 vs equal weight at the same return: **{i['risk_reduction_pct']:.2f}%** "
          f"(vol {i['vol_ew']:.4%} -> {i['vol_p1']:.4%})"]
    if "return_gain" in i:
        L += [f"- Return gain of P1 vs equal weight at the same risk: **{i['return_gain']:.4%}** "
              f"({i['return_ew']:.4%} -> {i['return_max_at_ew_risk']:.4%})"]
    if "delta_vol" in i:
        L += [f"- Price of realism [M9]: dVar = {i['delta_var']:.6f} ({i['delta_var_pct']:.2f}% of P1), "
              f"dVol = {i['delta_vol']:.4%} ({i['delta_vol_pct']:.2f}% of P1)"]
    n_fail = int((v["status"] == "FAIL").sum())
    L += ["", "## Verification", "",
          f"- {len(v)} checks, **{len(v) - n_fail} PASS / {n_fail} FAIL** (see verification.csv)"]
    if R.p2_enum is not None and R.p2_enum.status == "optimal":
        L += [f"- P2 enumeration: {R.p2_enum.duals.get('n_solved')} subsets solved in "
              f"{R.p2_enum.runtime:.1f}s; matches MIQP optimum"
              f" (rel. diff {abs(R.p2.variance - R.p2_enum.variance) / R.p2_enum.variance:.2e})"]
    if R.warnings:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in R.warnings]
    L += ["", "*Statistics are estimated from history; in-sample results do not guarantee "
          "out-of-sample performance (see backtest.csv).*", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--r-target", type=float, default=None, help="annual target return")
    ap.add_argument("--solver", default="SCIP", help="MIQP solver (SCIP or GUROBI)")
    ap.add_argument("--no-enum", action="store_true", help="skip the enumeration check")
    ap.add_argument("--no-png", action="store_true", help="skip PNG export")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("choreographer", "kaleido", "logistro", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = config.Config(refresh=args.refresh, r_target=args.r_target, solver=args.solver,
                        run_enumeration=not args.no_enum)
    R = run_pipeline(cfg)
    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)
    export_tables(R, out)
    if not args.no_png:
        figs = plots.all_figures(R)
        done = sum(plots.save_png(f, out / f"{k}.png") for k, f in figs.items())
        print(f"PNG figures written: {done}/{len(figs)}")
    (out / "summary.md").write_text(summary_md(R), encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8").encode("ascii", "replace").decode())
    return 1 if (R.verification["status"] == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
