# Optimal Portfolio Allocation: results summary

- Dataset: **1255 trading days x 18 stocks** (2021-01-04 to 2025-12-31), source: cache, rows dropped: 0
- Returns T = 1254; annualised with 252 days; target return r_target = 18.6436% (equal-weight return unless overridden)

## Optimal values

- **P1 (easy QP)**: variance 0.015736, volatility 12.5445%, expected return 18.6436%
  - MSFT: 2.11%
  - NVDA: 7.05%
  - GOOGL: 5.39%
  - PG: 7.61%
  - KO: 25.12%
  - JPM: 5.67%
  - XOM: 15.77%
  - JNJ: 29.37%
  - UNH: 1.91%
- **P2 (hard MIQP, SCIP)**: variance 0.016186, volatility 12.7223%, expected return 18.7880%
  - NVDA: 7.34%
  - GOOGL: 7.20%
  - KO: 34.08%
  - XOM: 16.58%
  - JNJ: 34.80%

## Impact metrics

- Risk reduction of P1 vs equal weight at the same return: **26.35%** (vol 17.0316% -> 12.5445%)
- Return gain of P1 vs equal weight at the same risk: **11.0129%** (18.6436% -> 29.6565%)
- Price of realism [M9]: dVar = 0.000449 (2.86% of P1), dVol = 0.1778% (1.42% of P1)

## Verification

- 15 checks, **15 PASS / 0 FAIL** (see verification.csv)
- P2 enumeration: 12444 subsets solved in 13.1s; matches MIQP optimum (rel. diff 1.88e-07)

*Statistics are estimated from history; in-sample results do not guarantee out-of-sample performance (see backtest.csv).*
