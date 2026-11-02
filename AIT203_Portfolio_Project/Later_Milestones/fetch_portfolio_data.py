"""
fetch_portfolio_data.py

Downloads historical adjusted close prices for a chosen set of stocks
from Yahoo Finance, saves them to CSV, and computes the two quantities
your portfolio optimization actually needs:
  - expected_returns (mu): average daily return per stock
  - cov_matrix (Sigma):    covariance between every pair of stocks' returns

Run this in your own local environment (VS Code, Jupyter, Colab, etc.) --
it needs internet access to reach Yahoo Finance.

Install once:
    pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd

# 1. Choose your stock universe and time window -----------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "PG",
    "KO", "PEP", "XOM", "CVX", "JNJ",
    "UNH", "HD", "DIS",
]  # replace with your own chosen 15-20 stocks

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

# 2. Download adjusted close prices ------------------------------------------
raw = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,  # keep both "Close" (unadjusted) and "Adj Close"
    progress=False,
)
# Yahoo provides Open, High, Low, Close, Adj Close and Volume.
# We use only "Adj Close": the closing price corrected for splits and dividends.
prices = raw["Adj Close"]  # one column per ticker, one row per trading day

# 3. Clean the data -----------------------------------------------------------
prices = prices.dropna(axis=0, how="any")  # drop days where any stock is missing
print(f"Dataset shape: {prices.shape[0]} trading days x {prices.shape[1]} stocks")

prices.to_csv("adjusted_close_prices.csv")

# 4. Compute daily returns -----------------------------------------------------
returns = prices.pct_change().dropna()
returns.to_csv("daily_returns.csv")

# 5. Compute the two quantities your optimization needs ------------------------
expected_returns = returns.mean()   # mu: estimated as the average daily return per stock
cov_matrix = returns.cov()          # Sigma: covariance between every pair

expected_returns.to_csv("expected_returns.csv")
cov_matrix.to_csv("covariance_matrix.csv")

print("\nExpected daily returns (mu):")
print(expected_returns)
print("\nCovariance matrix (Sigma) shape:", cov_matrix.shape)
