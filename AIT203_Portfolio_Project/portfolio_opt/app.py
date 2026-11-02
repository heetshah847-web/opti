"""Streamlit dashboard.  Run with:  streamlit run app.py

All model logic lives in src/; this file only collects inputs and displays results.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import baselines as bl
from src import pipeline, plots, stats

st.set_page_config(page_title="Optimal Portfolio Allocation", layout="wide")

# ------------------------------------------------------------------ cached steps
@st.cache_data(show_spinner="Loading data...")
def cached_prepare(start: str, end: str, refresh: bool):
    cfg = config.Config(start=start, end=end, refresh=refresh)
    _, info, _, est = pipeline.prepare(cfg)
    mv = bl.global_min_variance(est)
    return info, est.tickers, float(est.mu @ mv), float(est.mu.max()), pipeline.default_r_target(est)


@st.cache_data(show_spinner="Solving P1, P2, frontiers, backtest (about 30-60 s)...")
def cached_run(cfg: config.Config):
    return pipeline.run_pipeline(cfg)


def show(fig_key: str, fig, key: str | None = None):
    st.plotly_chart(fig, width="stretch", key=key or fig_key)
    cap = plots.caption(fig_key)
    if cap:
        st.caption(cap)


def pct(x: float, d: int = 2) -> str:
    return f"{x:.{d}%}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Inputs")
    start = st.text_input("Start date", config.START_DATE)
    end = st.text_input("End date (exclusive)", config.END_DATE)
    refresh = st.checkbox("Refresh data from Yahoo Finance", False)
    try:
        info0, tickers, r_lo, r_hi, r_default = cached_prepare(start, end, refresh)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()
    r_target = st.slider("Target annual return r_target", round(r_lo, 4), round(r_hi, 4),
                         round(r_default, 4), 0.001, format="%.3f",
                         help="Default = equal-weight portfolio return (B1).")
    st.subheader("Hard problem P2")
    K = st.number_input("K (max stocks held)", 1, len(tickers), config.HardParams.K)
    l = st.number_input("l (min weight if held)", 0.0, 1.0, config.HardParams.l, 0.005, format="%.3f")
    u = st.number_input("u (max weight per stock)", 0.01, 1.0, config.HardParams.u, 0.05, format="%.2f")
    c = st.number_input("c (transaction cost)", 0.0, 0.1, config.HardParams.c, 0.0005, format="%.4f")
    rf = st.number_input("Risk-free rate rf (for Sharpe)", 0.0, 0.2, 0.0, 0.005, format="%.3f")
    w0_mode = st.radio("Starting portfolio w0", ["Equal split", "Custom"])
    w0 = None
    if w0_mode == "Custom":
        txt = st.text_area(f"{len(tickers)} comma-separated weights (sum to 1)",
                           ",".join(f"{1 / len(tickers):.4f}" for _ in tickers))
        try:
            w0 = tuple(float(x) for x in txt.split(","))
        except ValueError:
            st.error("Could not parse w0.")
    solver_opts = [s for s in ("SCIP", "GUROBI") if s in cp.installed_solvers()]
    solver = st.selectbox("MIQP solver", solver_opts or ["SCIP"])
    st.subheader("Out-of-sample split")
    tr0, tr1 = st.text_input("Train start", config.TRAIN_START), st.text_input("Train end", config.TRAIN_END)
    te0, te1 = st.text_input("Test start", config.TEST_START), st.text_input("Test end", config.TEST_END)
    run = st.button("Run", type="primary")

cfg = config.Config(start=start, end=end, refresh=refresh, r_target=float(r_target),
                    hard=config.HardParams(int(K), float(l), float(u), float(c)), w0=w0,
                    rf=float(rf), solver=solver, train_start=tr0, train_end=tr1,
                    test_start=te0, test_end=te1)
if run:
    st.session_state["cfg"] = cfg
if "cfg" not in st.session_state:
    st.title("Optimal Portfolio Allocation Across a Diversified Stock Portfolio")
    st.info(f"Loaded {info0['n_days']} trading days x {info0['n_stocks']} stocks. "
            "Set the inputs in the sidebar and press **Run**.")
    st.stop()

try:
    R = cached_run(st.session_state["cfg"])
except (ValueError, RuntimeError) as exc:
    st.error(f"{exc}")
    st.stop()
est, tick = R.est, R.est.tickers
for w in R.warnings:
    st.warning(w)

tabs = st.tabs(["Overview", "Data", "Stock statistics", "Easy problem (P1)", "Hard problem (P2)",
                "Comparison & impact", "Efficient frontiers", "Backtest", "Verification"])

# ------------------------------------------------------------------ 1 overview
with tabs[0]:
    st.title("Optimal Portfolio Allocation Across a Diversified Stock Portfolio")
    st.markdown("**Question:** how should one budget be split across 18 well-known stocks so that "
                "risk (variance) is as small as possible while still reaching a target return? "
                "The *easy* problem P1 is a convex quadratic program; the *hard* problem P2 adds "
                "a stock-count limit, minimum/maximum position sizes and trading costs.")
    a, b, c3, d = st.columns(4)
    a.metric("Data points (trading days)", R.info["n_days"])
    b.metric("Stocks (features)", R.info["n_stocks"])
    c3.metric("Date range", f"{R.info['first_date']} to {R.info['last_date']}")
    d.metric("Source", R.info["source"])
    st.caption(f"Rows dropped for missing prices: {R.info['rows_dropped']}. Only 'Adj Close' is used.")
    st.subheader("Headline results")
    head = R.comparison.loc[[n for n in ("B1 Equal weight", "P1 Easy", "P2 Hard") if n in R.comparison.index],
                            ["return_ann", "volatility_ann", "sharpe", "max_drawdown", "n_held"]]
    st.dataframe(head.style.format({"return_ann": "{:.2%}", "volatility_ann": "{:.2%}",
                                    "sharpe": "{:.2f}", "max_drawdown": "{:.2%}"}))
    st.caption(f"Target return r_target = {pct(R.r_target)}; risk-free rate rf = {R.cfg.rf:.3f}.")

# ------------------------------------------------------------------ 2 data
with tabs[1]:
    st.subheader(f"Dataset shape: {R.info['n_days']} trading days x {R.info['n_stocks']} stocks")
    st.dataframe(R.prices.head(10).style.format("{:.2f}"))
    st.caption("First rows of the adjusted close price table (price corrected for splits and dividends).")
    show("normalized_prices", plots.normalized_prices(R.prices))
    show("return_distributions", plots.return_distributions(R.returns))
    st.subheader("Sectors")
    st.dataframe(pd.DataFrame({"sector": [config.SECTORS[t] for t in tick]}, index=tick))

# ------------------------------------------------------------------ 3 stock stats
with tabs[2]:
    st.dataframe(R.stock_stats.style.format({
        "mean_return_ann": "{:.2%}", "volatility_ann": "{:.2%}", "sharpe": "{:.2f}",
        "max_drawdown": "{:.2%}", "best_day": "{:.2%}", "worst_day": "{:.2%}"}))
    st.latex(r"\mu_i=\tfrac1T\sum_t r(t,i)\ \text{[F2]},\quad \mu^{ann}=252\mu\ \text{[F4]}")
    show("risk_return", plots.risk_return(R.stock_stats))
    st.latex(r"\rho_{ij}=\Sigma_{ij}/(\sigma_i\sigma_j)\ \text{[F5]}")
    show("corr_heatmap", plots.corr_heatmap(stats.correlation(est.sigma), tick))
    st.latex(r"\Sigma_{ij}=\tfrac1{T-1}\sum_t (r_{ti}-\mu_i)(r_{tj}-\mu_j)\ \text{[F3]}")
    show("cov_heatmap", plots.cov_heatmap(est.sigma, tick))


def metrics_panel(name: str):
    m = R.comparison.loc[name]
    cols = st.columns(4)
    cols[0].metric("Return [M1]", pct(m.return_ann))
    cols[1].metric("Volatility [M2]", pct(m.volatility_ann))
    cols[2].metric("Sharpe [M3]", f"{m.sharpe:.2f}")
    cols[3].metric("Max drawdown [M4]", pct(m.max_drawdown))
    cols = st.columns(4)
    cols[0].metric("95% VaR / CVaR daily [M5]", f"{pct(m.var95_daily)} / {pct(m.cvar95_daily)}")
    cols[1].metric("Effective holdings [M6]", f"{m.effective_n:.2f} ({int(m.n_held)} held)")
    cols[2].metric("Diversification ratio [M8]", f"{m.diversification_ratio:.2f}")
    cols[3].metric("Turnover / cost [M10]", f"{pct(m.turnover)} / {pct(m.transaction_cost, 3)}")


def verification_panel(model: str):
    v = R.verification
    v = v[v["model"].isin([model, "cross"])] if model != "all" else v
    st.dataframe(v.style.map(lambda s: "color: green" if s == "PASS" else ("color: red" if s == "FAIL" else ""),
                             subset=["status"]).format({"residual": "{:.2e}", "tolerance": "{:.0e}"}))


# ------------------------------------------------------------------ 4 P1
with tabs[3]:
    st.latex(r"\min_w\ w^\top\Sigma w \quad \text{s.t.}\quad \mathbf 1^\top w=1\ [C1],\ \ \mu^\top w\ge r_{target}\ [C2],\ \ w\ge0\ [C3]\qquad [P1]")
    st.latex(r"f_0(w)=w^\top\Sigma w\ \ \text{s.t.}\ \ r_{target}-\mu^\top w\le0,\ -w_i\le0,\ \mathbf 1^\top w-1=0")
    st.write(f"Target return **{pct(R.r_target)}**; solver {R.p1.solver}, status **{R.p1.status}**, "
             f"optimal variance **{R.p1.variance:.6f}**, volatility **{pct(R.p1.volatility, 3)}**.")
    show("weights_P1", plots.weights_bar(R.p1.w, tick, "P1 optimal weights"), "p1w")
    st.dataframe(pd.DataFrame({"weight": R.p1.w}, index=tick).style.format("{:.2%}"))
    st.subheader("Metrics")
    metrics_panel("P1 Easy")
    st.latex(r"RC_i=\dfrac{w_i(\Sigma w)_i}{w^\top\Sigma w}\ \text{[M7]}")
    show("risk_contrib_P1", plots.risk_contrib(R.risk_contrib["P1 Easy"].to_numpy(), tick,
                                                "P1 risk contribution by stock [M7]"), "p1rc")
    st.subheader("Verification")
    verification_panel("P1")

# ------------------------------------------------------------------ 5 P2
with tabs[4]:
    st.latex(r"\min_w\ w^\top\Sigma w\ \ \text{s.t.}\ \ \mathbf 1^\top w=1\,[C1],\ \ \mu^\top w-c\sum_i(b_i+s_i)\ge r_{target}\,[C4],\ \ w_i-w_{0,i}=b_i-s_i\,[C5],\ \ l z_i\le w_i\le u z_i\,[C6],\ \ \sum_i z_i\le K\,[C7]\qquad [P2]")
    st.caption("Assumption: the one-time cost in [C4] is charged against the annual return "
               "(holding period of about one year). Cost actually paid is c*sum|w - w0| from the final weights.")
    p = R.cfg.hard
    st.write(f"Parameters: K={p.K}, l={p.l}, u={p.u}, c={p.c}; feasibility: {R.feasibility.message} "
             f"(max return before costs {pct(R.feasibility.max_return_gross)}, "
             f"greedy net {pct(R.feasibility.max_return_net)}).")
    if not R.p2.ok:
        st.error(f"P2 not solved: {R.p2.message or R.p2.status}")
    else:
        st.write(f"Solver {R.p2.solver}, status **{R.p2.status}**, variance **{R.p2.variance:.6f}**, "
                 f"volatility **{pct(R.p2.volatility, 3)}**, runtime {R.p2.runtime:.2f}s.")
        show("weights_P2", plots.weights_bar(R.p2.w, tick, "P2 optimal weights"), "p2w")
        trades = pd.DataFrame({"weight": R.p2.w, "z (held)": R.p2.z.astype(int), "w0": R.w0,
                               "buy": np.clip(R.p2.w - R.w0, 0, None),
                               "sell": np.clip(R.w0 - R.p2.w, 0, None)}, index=tick)
        st.dataframe(trades.style.format({"weight": "{:.2%}", "w0": "{:.2%}", "buy": "{:.2%}", "sell": "{:.2%}"}))
        mrow = R.comparison.loc["P2 Hard"]
        st.metric("Transaction cost paid [M10] = c * turnover", pct(mrow.transaction_cost, 3),
                  help=f"turnover = {pct(mrow.turnover)}")
        metrics_panel("P2 Hard")
        show("risk_contrib_P2", plots.risk_contrib(R.risk_contrib["P2 Hard"].to_numpy(), tick,
                                                    "P2 risk contribution by stock [M7]"), "p2rc")
        st.subheader("Enumeration verification")
        if R.p2_enum is not None and R.p2_enum.status == "optimal":
            rel = abs(R.p2.variance - R.p2_enum.variance) / R.p2_enum.variance
            st.write(f"{R.p2_enum.duals['n_solved']} of {R.p2_enum.duals['n_subsets']} subsets solved in "
                     f"{R.p2_enum.runtime:.1f}s; enumeration variance {R.p2_enum.variance:.8f}; "
                     f"relative difference **{rel:.2e}** -> "
                     f"{'MATCH' if rel < config.TOL else 'MISMATCH'}")
        else:
            st.info("Enumeration not run or not possible for these parameters.")
        st.subheader("Verification")
        verification_panel("P2")

# ------------------------------------------------------------------ 6 comparison
with tabs[5]:
    nice = R.comparison.rename(columns={
        "return_ann": "Return [M1]", "volatility_ann": "Volatility [M2]", "sharpe": "Sharpe [M3]",
        "max_drawdown": "Max DD [M4]", "var95_daily": "VaR95 [M5]", "cvar95_daily": "CVaR95 [M5]",
        "effective_n": "Eff. N [M6]", "n_held": "# held [M6]", "diversification_ratio": "Div. ratio [M8]",
        "turnover": "Turnover [M10]", "transaction_cost": "Cost [M10]", "variance": "Variance"})
    st.dataframe(nice.style.format("{:.4f}"))
    st.caption("B1 equal weight, B2 hindsight best single stock, B3 familiar tech names, "
               "B4 global minimum variance, P1 easy optimum, P2 hard optimum. All use annualised inputs.")
    show("weights_grouped", plots.weights_grouped(R.weights))
    i = R.impact
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk reduction, P1 vs equal weight", f"{i['risk_reduction_pct']:.2f}%")
    c1.latex(r"\dfrac{\sigma_{EW}-\sigma_{P1}}{\sigma_{EW}}")
    if "return_gain" in i:
        c2.metric("Return gain at equal risk", pct(i["return_gain"]))
        c2.latex(r"\max \mu^\top w\ \text{s.t. } w^\top\Sigma w\le\sigma_{EW}^2,[C1],[C3]\ \Rightarrow\ \text{gain}=\mu^\top w^*-\mu^\top w_{EW}")
    if "delta_vol" in i:
        c3.metric("Price of realism [M9]", f"{pct(i['delta_vol'], 3)} vol ({i['delta_vol_pct']:.2f}%)")
        c3.latex(r"\Delta_{vol}=\sigma^*_{P2}-\sigma^*_{P1},\ \ \Delta_{var}=\text{var}^*_{P2}-\text{var}^*_{P1}")
        st.caption(f"Variance gap {i['delta_var']:.6f} ({i['delta_var_pct']:.2f}% of the P1 variance).")

# ------------------------------------------------------------------ 7 frontiers
with tabs[6]:
    if R.frontier_p1 is None:
        st.info("Frontier not computed.")
    else:
        show("frontier", plots.frontier(R))
        show("delta_vol", plots.delta_vol(R.frontier_p2))
        st.caption("Targets above P2's maximum achievable return are infeasible and omitted from P2's curve.")

# ------------------------------------------------------------------ 8 backtest
with tabs[7]:
    st.markdown("**In-sample** (estimated and evaluated on the same data).")
    show("growth", plots.growth(R.wealth))
    show("drawdown", plots.drawdown(R.wealth))
    st.markdown("**Out-of-sample** (mu and Sigma estimated on the training window only).")
    if R.oos is None:
        st.info("Out-of-sample check skipped (see warnings).")
    else:
        st.write(f"Train: {R.cfg.train_start} to {R.cfg.train_end} ({R.oos.n_train} days). "
                 f"Test: {R.cfg.test_start} to {R.cfg.test_end} ({R.oos.n_test} days). "
                 f"Target on train data: {pct(R.oos.r_target)}.")
        st.dataframe(R.oos.metrics.style.format({
            "realized_return_ann": "{:.2%}", "realized_volatility_ann": "{:.2%}",
            "max_drawdown": "{:.2%}", "total_return": "{:.2%}", "growth_of_1": "{:.3f}"}))
        show("oos_growth", plots.growth(R.oos.wealth, "Out-of-sample growth of $1 (test window)"))
        st.warning("Out-of-sample performance is not guaranteed to match in-sample results.")

# ------------------------------------------------------------------ 9 verification
with tabs[8]:
    verification_panel("all")
    n_fail = int((R.verification["status"] == "FAIL").sum())
    (st.success if n_fail == 0 else st.error)(
        f"{len(R.verification) - n_fail} of {len(R.verification)} checks PASS")
    st.download_button("Download verification.csv", R.verification.to_csv(index=False), "verification.csv")
