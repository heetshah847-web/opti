# Optimal Portfolio Allocation (AIT 203)

**Question.** Given 18 well-known stocks, how should one budget be split so that the portfolio's risk
(variance of returns) is as small as possible while still reaching a target annual return?
Two problems are solved on the same data:

* **P1 (easy)**: classic mean-variance, a convex quadratic program.
* **P2 (hard)**: the same objective plus real-world limits (at most K stocks, minimum/maximum position
  size, transaction costs): a mixed-integer QP. The gap between the two is the *price of realism*.

## Install and run

```
pip install -r requirements.txt
python run_all.py          # CLI pipeline -> results/
streamlit run app.py       # dashboard
python -m pytest           # tests
```

`run_all.py` options: `--refresh` (re-download), `--r-target 0.2`, `--solver SCIP|GUROBI`,
`--no-enum`, `--no-png`. PNG export uses kaleido, which needs a local Chrome/Chromium install;
if it is missing the PNG step logs a warning and everything else still runs.

## Data

Yahoo Finance via `yfinance`, 18 tickers, 2021-01-01 to 2026-01-01 (end exclusive), `auto_adjust=False`;
only **Adj Close** feeds the models. Raw and cleaned prices are cached in `data/` (use "refresh" to
re-download). Dates with any missing price are dropped and the count is logged.

## Formulations

Annualised inputs feed all models.

| ID | Formula |
|----|---------|
| F1 | r(t,i) = (P(t,i) - P(t-1,i)) / P(t-1,i) |
| F2 | mu_i = (1/T) sum_t r(t,i) |
| F3 | Sigma_ij = 1/(T-1) sum_t (r(t,i)-mu_i)(r(t,j)-mu_j) |
| F4 | mu_ann = 252 mu; Sigma_ann = 252 Sigma; vol_ann = sqrt(252) vol_daily |
| F5 | rho_ij = Sigma_ij / (sigma_i sigma_j) |

**P1**: minimise w'Σw s.t. [C1] 1'w = 1, [C2] μ'w ≥ r_target, [C3] w ≥ 0.

**P2**: minimise w'Σw s.t. [C1]; [C4] μ'w − c·Σ(b_i+s_i) ≥ r_target; [C5] w_i − w0_i = b_i − s_i;
[C6] l·z_i ≤ w_i ≤ u·z_i; [C7] Σz_i ≤ K; z_i ∈ {0,1}, b, s ≥ 0.

Metrics: M1 return, M2 volatility, M3 Sharpe, M4 max drawdown, M5 historical 95% VaR/CVaR,
M6 effective number of holdings, M7 risk contributions, M8 diversification ratio, M9 price of realism,
M10 turnover and transaction cost paid.

## Parameters

| Name | Default | Meaning |
|------|---------|---------|
| r_target | equal-weight return (B1) | minimum expected annual return |
| K | 5 | maximum number of stocks held |
| l / u | 0.02 / 0.40 | minimum / maximum weight of a held stock |
| c | 0.001 | proportional transaction cost |
| w0 | 1/18 each | starting portfolio |
| rf | 0.0 | risk-free rate used in the Sharpe ratio |

Baselines: B1 equal weight, B2 hindsight best single stock, B3 equal weight on AAPL/MSFT/NVDA/GOOGL/META/AMZN,
B4 global minimum variance.

## Verification

P2 is solved by SCIP (or Gurobi) as a MIQP **and** independently by exact enumeration: every support set
with ceil(1/u) ≤ |S| ≤ K is fixed and the resulting convex QP solved with Clarabel. Both optima must agree
within 1e-6 relative. `results/verification.csv` also holds constraint residuals (PASS/FAIL at 1e-6),
the P1 KKT check, P1 ≤ equal-weight, and P2 ≥ P1.

## Assumptions and limitations

* mu and Sigma are estimated from history; the future will differ. In-sample results are optimistic;
  the out-of-sample section estimates on the training window only and is no guarantee either.
* The one-time transaction cost in [C4] is charged against the annual return (about one-year holding).
* No short selling. Portfolio curves assume constant weights (daily rebalancing).
* Sigma is symmetrised; a ridge of 1e-10·I is added (and logged) only if its smallest eigenvalue is negative.
* P2 can be infeasible for high r_target (the 40% cap limits the reachable return); this is reported,
  not raised. The greedy bound used for the pre-check: gross return is an upper bound, net is a feasible point.
* The enumeration check is capped at 300,000 subsets.

## Which file implements which formula

| File | IDs |
|------|-----|
| `src/data.py` | download, cache, cleaning |
| `src/stats.py` | F1-F5, M1-M8, M10 |
| `src/easy_problem.py` | P1, C1-C3, global min variance |
| `src/hard_problem.py` | P2, C1, C4-C7, enumeration verifier, feasibility pre-checks |
| `src/baselines.py` | B1-B4 |
| `src/frontier.py` | frontier sweeps, M9 |
| `src/backtest.py` | in-sample and out-of-sample evaluation |
| `src/verify.py` | constraint checks, KKT, cross-checks |
| `src/pipeline.py` | orchestration shared by the CLI and the GUI |
| `src/plots.py` | all charts |
| `app.py` / `run_all.py` | GUI / CLI only, no model logic |
