"""Load (or download) prices and compute the annualised mu and Sigma."""
from pathlib import Path

import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "PG",
    "KO", "PEP", "XOM", "CVX", "JNJ",
    "UNH", "HD", "DIS",
]
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
TRADING_DAYS = 252

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = DATA_DIR / "adjusted_close_prices.csv"


def download_prices() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)
    prices = raw["Adj Close"].dropna(axis=0, how="any")
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICES_CSV)
    return prices


def load_prices(refresh: bool = False) -> pd.DataFrame:
    if refresh or not PRICES_CSV.exists():
        return download_prices()
    return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)


def estimate(prices: pd.DataFrame):
    """Return (tickers, mu_annual, sigma_annual) as numpy arrays."""
    returns = prices.pct_change().dropna()
    mu = returns.mean().to_numpy() * TRADING_DAYS
    sigma = returns.cov().to_numpy() * TRADING_DAYS
    return list(returns.columns), mu, sigma
