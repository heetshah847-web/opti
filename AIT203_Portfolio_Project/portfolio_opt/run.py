"""Entry point:  python -m portfolio_opt.run [--r-target 0.25] [--frontier] [--refresh]"""
import argparse
import warnings

warnings.filterwarnings("ignore")  # infeasible supports trigger solver warnings

import numpy as np

from . import data, models

K, L, U, C = 5, 0.02, 0.40, 0.001


def report(name, sol, tickers, mu, sigma, r_target, **kw):
    print(f"\n=== {name} ===")
    if sol is None:
        print("infeasible")
        return
    w = sol["w"]
    print(f"volatility {np.sqrt(sol['variance']):.4f}   expected return {mu @ w:.4f}")
    for t, wi in sorted(zip(tickers, w), key=lambda x: -x[1]):
        print(f"  {t:6s} {wi:7.2%}")
    for label, ok in models.check_solution(w, mu, sigma, r_target, **kw).items():
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-target", type=float, default=0.25,
                    help="minimum expected annual return")
    ap.add_argument("--frontier", action="store_true",
                    help="solve both problems over a range of targets and plot")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    args = ap.parse_args()

    tickers, mu, sigma = data.estimate(data.load_prices(args.refresh))
    n = len(tickers)
    w0 = np.full(n, 1 / n)
    print(f"{n} stocks; max annual mu = {mu.max():.3f}; r_target = {args.r_target}")

    eq_var = w0 @ sigma @ w0
    p1 = models.solve_p1(mu, sigma, args.r_target)
    report("P1: mean-variance QP", p1, tickers, mu, sigma, args.r_target)
    if p1 and mu @ w0 >= args.r_target:
        print(f"  sanity: P1 var {p1['variance']:.5f} <= equal split var {eq_var:.5f}:"
              f" {p1['variance'] <= eq_var + 1e-9}")

    p2s = models.P2Solver(mu, sigma, K, L, U, C, w0)
    p2 = p2s.solve(args.r_target)
    report("P2: cardinality + limits + trading costs", p2, tickers, mu, sigma,
           args.r_target, K=K, l=L, u=U, c=C, w0=w0)
    if p1 and p2:
        print(f"\nPrice of realism: {np.sqrt(p2['variance']) - np.sqrt(p1['variance']):.4f}"
              " volatility")

    if args.frontier:
        plot_frontier(mu, sigma, p2s)


def plot_frontier(mu, sigma, p2s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.linspace(0.10, 0.50, 9)
    v1, v2 = [], []
    for r in targets:
        a, b = models.solve_p1(mu, sigma, r), p2s.solve(r)
        v1.append(np.sqrt(a["variance"]) if a else np.nan)
        v2.append(np.sqrt(b["variance"]) if b else np.nan)
        print(f"r={r:.3f}  P1 vol={v1[-1]:.4f}  P2 vol={v2[-1]:.4f}")
    plt.plot(v1, targets, "o-", label="P1 (easy)")
    plt.plot(v2, targets, "s-", label="P2 (hard)")
    plt.xlabel("volatility (annual)")
    plt.ylabel("target return (annual)")
    plt.legend()
    plt.title("Efficient frontiers")
    plt.savefig("frontier.png", dpi=150, bbox_inches="tight")
    print("saved frontier.png")


if __name__ == "__main__":
    main()
