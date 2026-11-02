# Mathematical Formulation (Milestone 2, not needed for the proposal)

The project PDF asks for two problems on the same data: an "easy" one (a linear program or convex problem that solves the original approximately) and a "hard" one that adds complexity and captures the real problem more faithfully. The professor's Option 6 slide suggests exactly this split: convex mean-variance QP, then transaction costs and cardinality constraints.

---

## 1. Notation

```
i = 1, ..., N      stocks, N = 18
t = 1, ..., T      trading days of returns, T ≈ 1,259
                   (one fewer than price days: the first day has no "yesterday")
```

## 2. Parameters (computed from the data, then held fixed)

Daily return, from adjusted close P(t,i):
```
r(t,i) = ( P(t,i) - P(t-1,i) ) / P(t-1,i)
```

Expected return of each stock (historical average):
```
μi = (1/T) · Σt r(t,i)            μ = (μ1, ..., μ18)
```

Covariance matrix (18 × 18):
```
Σij = 1/(T-1) · Σt ( r(t,i) - μi ) ( r(t,j) - μj )
```

Annualizing (recommended for readable numbers):
```
μ_annual = 252 · μ          Σ_annual = 252 · Σ
```

Chosen by us: r_target, the minimum expected annual return. It must be at most the largest μi, or no split can reach it.

## 3. Easy problem (P1): classic mean-variance

Decision variables: w = (w1, ..., w18), the fraction of the budget in each stock.

```
minimize     w^T Σ w                      portfolio risk (variance)
subject to   w1 + w2 + ... + w18 = 1      invest the full budget
             μ^T w >= r_target            reach the target return
             wi >= 0,  i = 1,...,18       no short selling
```

Standard form (constraints as "<= 0" or "= 0"):
```
minimize     f0(w) = w^T Σ w
subject to   r_target - μ^T w <= 0
             -wi <= 0,            i = 1,...,18
             1^T w - 1 = 0
```

Type: quadratic program, 18 continuous variables. Σ is positive semi-definite, so the objective is bowl-shaped and any best point found is the global best.

## 4. Hard problem (P2): real-world limits

Extra parameters (suggested values):
```
K  = 5        maximum number of stocks held
l  = 0.02     minimum weight if a stock is held (2%)
u  = 0.40     maximum weight in any one stock (40%)
c  = 0.001    transaction cost: 0.1% of the amount traded
w0 = 1/18     starting portfolio (equal split), for each stock
```

Extra variables:
```
zi ∈ {0, 1}     1 if stock i is held, 0 if not
bi >= 0         amount of stock i bought, relative to w0
si >= 0         amount of stock i sold, relative to w0
```

Model:
```
minimize     w^T Σ w

subject to   w1 + ... + w18 = 1                       full budget
             μ^T w - c · Σi (bi + si) >= r_target    target return after trading costs
             wi - w0i = bi - si                       trades that move w0 to w
             l · zi <= wi <= u · zi                   hold 0, or between 2% and 40%
             z1 + ... + z18 <= K                      at most K stocks
             zi ∈ {0,1},  bi >= 0,  si >= 0
```

What each piece does:
- Cardinality (Σ zi <= K) caps how many stocks can be owned.
- Buy-in limits: if zi = 0 the weight is forced to 0; if zi = 1 it must be between 2% and 40%. This replaces wi >= 0.
- Transaction costs: splitting each change into bought and sold parts makes the cost c·|wi - w0i| linear. Assumption: the portfolio is held for about a year, so a one-time cost is charged against the annual return.

Type: mixed-integer quadratic program (MIQP), 72 variables. The yes/no variables make the set of allowed splits non-convex. There are C(18,5) = 8,568 possible groups of 5 stocks before choosing weights. Needs an integer-capable solver (e.g. Gurobi).

Feasibility checks: K·u = 2 >= 1 ✓ and K·l = 0.1 <= 1 ✓

Note: proportional transaction costs alone keep the problem convex. It's the yes/no variables (cardinality, minimum size) that make it hard.

## 5. How easy and hard connect

- P1 is a special case of P2 (set c = 0, K = 18, l = 0, u = 1). So for the same r_target: risk*(hard) >= risk*(easy).
- Price of realism = risk*(hard) - risk*(easy), easiest to read as a difference in volatility (square root of variance).
- Efficient frontier: solve both for a range of r_target values and plot both curves. The gap between them is the main visual for the Analysis milestone.

## 6. What to report later (Solution milestone)

For each problem: the optimal objective value, the optimal weights, and checks that every constraint holds (weights sum to 1, target return met, at most K stocks, held weights within limits).

Sanity check for P1: the equal split obeys all of P1's rules, so P1's optimal risk must be at most the equal split's risk.
