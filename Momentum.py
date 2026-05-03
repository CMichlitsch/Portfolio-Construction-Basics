from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_stocks.csv"


def _zscore(s: pd.Series, limit: float = 3.0) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-limit, limit)


def get_momentum_signal(CSV, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    Momentum signal: 12-1 month return (12M window, skip last month to avoid reversal).
    Uses Adj Close prices. Cross-sectionally z-scored.
    Returns pd.Series indexed by Symbol.
    """
    prices = (
        pd.read_csv(CSV, parse_dates=["Date"])
        .dropna(subset=["Symbol", "Adj Close"])
        .sort_values(["Symbol", "Date"])
    )

    def _mom(group: pd.DataFrame) -> float:
        group = group.sort_values("Date")
        n = len(group)
        if n < lookback + 1:
            return np.nan
        p_start = group["Adj Close"].iloc[-(lookback + 1)]
        p_end   = group["Adj Close"].iloc[-(skip + 1)]
        if p_start <= 0:
            return np.nan
        return p_end / p_start - 1.0

    raw = prices[["Symbol", "Date", "Adj Close"]].groupby("Symbol").apply(_mom).dropna()
    return _zscore(raw).rename("momentum")


def get_winners_losers(CSV):
    signal = get_momentum_signal(CSV)
    df = signal.reset_index()
    df.columns = ["Symbol", "signal"]
    return df.nlargest(20, "signal"), df.nsmallest(20, "signal")
