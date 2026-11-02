"""Download, cache and clean Yahoo Finance prices. Only "Adj Close" feeds the models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import config

log = logging.getLogger(__name__)
SEP = "|"


class DataError(RuntimeError):
    """Raised when prices can be neither downloaded nor loaded from cache."""


def _raw_path(start: str, end: str) -> Path:
    return config.DATA_DIR / f"raw_prices_{start}_{end}.csv"


def _clean_path(start: str, end: str) -> Path:
    return config.DATA_DIR / f"prices_clean_{start}_{end}.csv"


def _download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(tickers, start=start, end=end, auto_adjust=False,
                       progress=False, threads=False)


def download_raw(tickers: List[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download all fields; re-request tickers that came back empty."""
    last_err: Exception | None = None
    raw = None
    for _ in range(retries):
        try:
            raw = _download(tickers, start, end)
            break
        except Exception as exc:  # network / yfinance failures
            last_err = exc
    if raw is None or raw.empty:
        raise DataError(f"Could not download prices from Yahoo Finance ({last_err}). "
                        "No cache is available; check the internet connection.")
    missing = [t for t in tickers if raw["Adj Close"][t].isna().all()]
    for t in missing:
        for _ in range(retries):
            try:
                one = _download([t], start, end)
                if not one.empty and not one["Adj Close"][t].isna().all():
                    for field_ in one.columns.get_level_values(0).unique():
                        raw[(field_, t)] = one[(field_, t)]
                    break
            except Exception as exc:
                last_err = exc
    still = [t for t in tickers if raw["Adj Close"][t].isna().all()]
    if still:
        raise DataError(f"No price data returned for: {still} ({last_err}).")
    return raw


def _save_raw(raw: pd.DataFrame, path: Path) -> None:
    flat = raw.copy()
    flat.columns = [f"{f}{SEP}{t}" for f, t in flat.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(path)


def _load_raw(path: Path) -> pd.DataFrame:
    flat = pd.read_csv(path, index_col=0, parse_dates=True)
    flat.columns = pd.MultiIndex.from_tuples([tuple(c.split(SEP, 1)) for c in flat.columns])
    return flat


def load_prices(start: str = config.START_DATE, end: str = config.END_DATE,
                tickers: List[str] | None = None,
                refresh: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Return (clean Adj Close prices [days x stocks], info dict).

    Cleaning: tickers are aligned on common trading dates and any date with a
    missing price is dropped; the number of dropped rows is logged and returned.
    """
    tickers = tickers or config.TICKERS
    raw_path = _raw_path(start, end)
    if raw_path.exists() and not refresh:
        raw = _load_raw(raw_path)
        source = "cache"
    else:
        try:
            raw = download_raw(tickers, start, end)
        except DataError:
            if raw_path.exists():
                log.warning("Download failed; falling back to cached data.")
                raw = _load_raw(raw_path)
                source = "cache (download failed)"
            else:
                raise
        else:
            _save_raw(raw, raw_path)
            source = "Yahoo Finance (yfinance)"

    adj = raw["Adj Close"][tickers]               # the ONLY field used by the models
    n_before = len(adj)
    prices = adj.dropna(axis=0, how="any")
    dropped = n_before - len(prices)
    log.info("Rows before cleaning: %d, dropped (missing price): %d", n_before, dropped)
    prices.to_csv(_clean_path(start, end))
    info = {"source": source, "rows_raw": n_before, "rows_dropped": dropped,
            "n_days": len(prices), "n_stocks": prices.shape[1],
            "first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date())}
    return prices, info
