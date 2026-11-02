"""Plotly figure builders shared by the GUI and the static PNG export.

Every builder returns a figure with a title and labelled axes; CAPTIONS holds the
one-sentence caption for each figure.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from . import stats
from .pipeline import Results

log = logging.getLogger(__name__)

PCT = ".0%"
PORT_COLORS = {"B1 Equal weight": "#7f7f7f", "B2 Best single stock": "#bcbd22",
               "B3 Familiar names": "#e377c2", "B4 Min-variance": "#8c564b",
               "P1 Easy": "#1f77b4", "P2 Hard": "#d62728"}

CAPTIONS = {
    "normalized_prices": "Each stock's adjusted close rebased to 1 on the first day, showing relative growth.",
    "return_distributions": "Distribution of daily simple returns [F1] per stock; wider boxes mean riskier stocks.",
    "risk_return": "Each stock's annualised volatility against its annualised mean return, coloured by sector.",
    "corr_heatmap": "Pairwise correlation [F5] of daily returns; low values mean better diversification.",
    "cov_heatmap": "Annualised covariance matrix [F3,F4] that drives the portfolio variance w'Σw.",
    "weights_bar": "Optimal weight per stock; stocks with 0% are not held.",
    "risk_contrib": "Share of total portfolio variance contributed by each stock [M7]; the bars sum to 100%.",
    "weights_grouped": "Weights of every baseline and optimised portfolio side by side.",
    "frontier": "Lowest achievable volatility for each target return; the shaded gap is the price of realism [M9].",
    "delta_vol": "Extra annual volatility P2 needs over P1 at each target return (Δvol, [M9]).",
    "growth": "Growth of $1 using fixed portfolio weights (daily rebalanced) over the in-sample period.",
    "drawdown": "Fall from the running peak of wealth [M4]; deeper dips mean larger losses.",
    "oos_growth": "Out-of-sample growth of $1 for weights estimated on the training window only.",
}


def _fmt(fig: go.Figure, title: str, x: str, y: str, h: int = 480) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=h,
                      template="plotly_white", legend_title_text="")
    return fig


def normalized_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0]
    fig = go.Figure()
    for t in norm.columns:
        fig.add_scatter(x=norm.index, y=norm[t], name=t, mode="lines",
                        line=dict(color=config.SECTOR_COLORS.get(config.SECTORS.get(t, ""))))
    return _fmt(fig, "Normalised adjusted close prices (start = 1)", "Date", "Price relative to start (x)")


def return_distributions(returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t in returns.columns:
        fig.add_box(y=returns[t], name=t, boxpoints=False,
                    marker_color=config.SECTOR_COLORS.get(config.SECTORS.get(t, "")))
    fig.update_yaxes(tickformat=".0%")
    return _fmt(fig, "Distribution of daily returns", "Stock", "Daily return (%)")


def risk_return(stats_df: pd.DataFrame) -> go.Figure:
    d = stats_df.reset_index()
    fig = px.scatter(d, x="volatility_ann", y="mean_return_ann", color="sector", text="ticker",
                     color_discrete_map=config.SECTOR_COLORS)
    fig.update_traces(textposition="top center", marker_size=10)
    fig.update_xaxes(tickformat=PCT)
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, "Risk vs return of the 18 stocks", "Annualised volatility (%)",
                "Annualised mean return (%)")


def _heatmap(m: np.ndarray, labels, title: str, zlabel: str, fmt: str, **kw) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=m, x=labels, y=labels, colorbar_title=zlabel,
                               colorscale="RdBu_r", texttemplate=fmt, **kw))
    fig.update_yaxes(autorange="reversed")
    return _fmt(fig, title, "Stock", "Stock", 620)


def corr_heatmap(corr: np.ndarray, labels) -> go.Figure:
    return _heatmap(corr, labels, "Correlation matrix of daily returns", "ρ", "%{z:.2f}",
                    zmin=-1, zmax=1)


def cov_heatmap(sigma: np.ndarray, labels) -> go.Figure:
    return _heatmap(sigma, labels, "Annualised covariance matrix", "Σ (annual)", "%{z:.3f}",
                    zmid=0)


def weights_bar(w: np.ndarray, tickers, title: str) -> go.Figure:
    colors = [config.SECTOR_COLORS.get(config.SECTORS.get(t, ""), "#999") for t in tickers]
    fig = go.Figure(go.Bar(x=list(tickers), y=w, marker_color=colors,
                           text=[f"{v:.1%}" for v in w], textposition="outside"))
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, title, "Stock", "Portfolio weight (% of budget)")


def risk_contrib(rc: np.ndarray, tickers, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(tickers), y=rc, marker_color="#1f77b4"))
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, title, "Stock", "Share of portfolio variance (%)")


def weights_grouped(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in weights.columns:
        fig.add_bar(x=weights.index, y=weights[name], name=name, marker_color=PORT_COLORS.get(name))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, "Portfolio weights by stock", "Stock", "Portfolio weight (% of budget)", 520)


def frontier(R: Results) -> go.Figure:
    f1, f2, est = R.frontier_p1, R.frontier_p2, R.est
    fig = go.Figure()
    ok2 = f2["feasible"]
    f1_match = f1[ok2.values]
    fig.add_scatter(x=f1_match["volatility"], y=f1_match["return"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=f2.loc[ok2, "volatility"], y=f2.loc[ok2, "return"], mode="lines+markers",
                    name="P2 hard frontier (gap = price of realism)", line=dict(color="#d62728"),
                    fill="tonextx", fillcolor="rgba(214,39,40,0.15)")
    fig.add_scatter(x=f1["volatility"], y=f1["return"], mode="lines+markers",
                    name="P1 easy frontier", line=dict(color="#1f77b4"))
    for sector, color in config.SECTOR_COLORS.items():
        idx = [i for i, t in enumerate(est.tickers) if config.SECTORS.get(t) == sector]
        fig.add_scatter(x=est.vols[idx], y=est.mu[idx], mode="markers+text", name=sector,
                        text=[est.tickers[i] for i in idx], textposition="top center",
                        marker=dict(color=color, size=8, opacity=0.8))
    for name, w in R.baselines.items():
        fig.add_scatter(x=[np.sqrt(w @ est.sigma @ w)], y=[est.mu @ w], mode="markers", name=name,
                        marker=dict(symbol="diamond", size=13, color=PORT_COLORS[name],
                                    line=dict(width=1, color="black")))
    for name, sol in (("P1 Easy", R.p1), ("P2 Hard", R.p2)):
        if sol.ok:
            fig.add_scatter(x=[sol.volatility], y=[est.mu @ sol.w], mode="markers",
                            name=f"{name} selected", marker=dict(symbol="star", size=18,
                                                                 color=PORT_COLORS[name],
                                                                 line=dict(width=1, color="black")))
    fig.update_xaxes(tickformat=PCT)
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, "Efficient frontiers: easy (P1) vs hard (P2)", "Annualised volatility (%)",
                "Annualised expected return (%)", 650)


def delta_vol(f2: pd.DataFrame) -> go.Figure:
    d = f2[f2["feasible"]]
    fig = go.Figure(go.Scatter(x=d["r_target"], y=d["delta_vol"], mode="lines+markers",
                               line=dict(color="#d62728")))
    fig.update_xaxes(tickformat=PCT)
    fig.update_yaxes(tickformat=".2%")
    return _fmt(fig, "Price of realism: Δvol = vol*(P2) − vol*(P1)", "Target annual return (%)",
                "Extra annualised volatility (percentage points)")


def growth(wealth: pd.DataFrame, title: str = "Growth of $1 (in-sample)") -> go.Figure:
    fig = go.Figure()
    for name in wealth.columns:
        fig.add_scatter(x=wealth.index, y=wealth[name], name=name, mode="lines",
                        line=dict(color=PORT_COLORS.get(name)))
    return _fmt(fig, title, "Date", "Portfolio value ($, start = 1)")


def drawdown(wealth: pd.DataFrame) -> go.Figure:
    dd = wealth / wealth.cummax() - 1
    fig = go.Figure()
    for name in dd.columns:
        fig.add_scatter(x=dd.index, y=dd[name], name=name, mode="lines",
                        line=dict(color=PORT_COLORS.get(name)))
    fig.update_yaxes(tickformat=PCT)
    return _fmt(fig, "Drawdown from running peak (in-sample)", "Date", "Drawdown (% below peak)")


def save_png(fig: go.Figure, path: Path) -> bool:
    """Write a static PNG; returns False (and logs) if the kaleido/Chrome export fails."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), width=1200, height=700, scale=2)
        return True
    except Exception as exc:  # kaleido needs a Chrome install
        log.warning("PNG export failed for %s: %s", path.name, exc)
        return False


def all_figures(R: Results) -> Dict[str, go.Figure]:
    """Every chart used in the GUI, keyed by file name stem."""
    est = R.est
    figs: Dict[str, go.Figure] = {
        "normalized_prices": normalized_prices(R.prices),
        "return_distributions": return_distributions(R.returns),
        "risk_return": risk_return(R.stock_stats),
        "corr_heatmap": corr_heatmap(stats.correlation(est.sigma), est.tickers),
        "cov_heatmap": cov_heatmap(est.sigma, est.tickers),
        "weights_P1": weights_bar(R.p1.w, est.tickers, "P1 optimal weights"),
        "risk_contrib_P1": risk_contrib(R.risk_contrib["P1 Easy"].to_numpy(), est.tickers,
                                        "P1 risk contribution by stock [M7]"),
        "weights_grouped": weights_grouped(R.weights),
        "growth": growth(R.wealth),
        "drawdown": drawdown(R.wealth),
    }
    if R.p2.ok:
        figs["weights_P2"] = weights_bar(R.p2.w, est.tickers, "P2 optimal weights")
        figs["risk_contrib_P2"] = risk_contrib(R.risk_contrib["P2 Hard"].to_numpy(), est.tickers,
                                               "P2 risk contribution by stock [M7]")
    if R.frontier_p1 is not None:
        figs["frontier"] = frontier(R)
        figs["delta_vol"] = delta_vol(R.frontier_p2)
    if R.oos is not None:
        figs["oos_growth"] = growth(R.oos.wealth, "Out-of-sample growth of $1 (test window)")
    return figs


def caption(key: str) -> Optional[str]:
    base = key.split("_P")[0] if key.startswith(("weights_P", "risk_contrib_P")) else key
    base = {"weights": "weights_bar", "risk_contrib": "risk_contrib"}.get(base, base)
    return CAPTIONS.get(base)
