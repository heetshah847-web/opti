"""All project defaults in one place (AIT 203: Optimal Portfolio Allocation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Fixed universe of 18 stocks, in this order.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "DIS", "AMZN", "TSLA", "HD",
    "PG", "KO", "PEP", "JPM", "V", "XOM", "CVX", "JNJ", "UNH",
]

# GICS sectors (metadata used for grouping and colouring).
SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "DIS": "Communication Services",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "JPM": "Financials", "V": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "JNJ": "Health Care", "UNH": "Health Care",
}

SECTOR_COLORS = {
    "Information Technology": "#1f77b4",
    "Communication Services": "#9467bd",
    "Consumer Discretionary": "#ff7f0e",
    "Consumer Staples": "#2ca02c",
    "Financials": "#8c564b",
    "Energy": "#d62728",
    "Health Care": "#17becf",
}

# Baseline B3 "familiar names" portfolio.
FAMILIAR_NAMES = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"          # exclusive
TRADING_DAYS = 252               # [F4]
TOL = 1e-6                       # verification tolerance
HELD_THRESHOLD = 1e-4            # a weight above this counts as "held" in [M6]
RIDGE = 1e-10                    # documented PSD safeguard for Sigma

TRAIN_START, TRAIN_END = "2021-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"
N_FRONTIER = 40
RANDOM_SEED = 0


@dataclass(frozen=True)
class HardParams:
    """Parameters of the hard problem [P2]."""
    K: int = 5          # max number of stocks held        [C7]
    l: float = 0.02     # min weight if held               [C6]
    u: float = 0.40     # max weight per stock             [C6]
    c: float = 0.001    # proportional transaction cost    [C4]


@dataclass(frozen=True)
class Config:
    start: str = START_DATE
    end: str = END_DATE
    refresh: bool = False
    r_target: Optional[float] = None          # None -> equal-weight return (B1)
    hard: HardParams = field(default_factory=HardParams)
    w0: Optional[Tuple[float, ...]] = None    # None -> equal split 1/N
    rf: float = 0.0                           # risk-free rate used in [M3]
    solver: str = "SCIP"                      # MIQP solver for P2
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    test_start: str = TEST_START
    test_end: str = TEST_END
    n_frontier: int = N_FRONTIER
    run_enumeration: bool = True
