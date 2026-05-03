from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_fundamentals.csv"


def _zscore(s: pd.Series, winsor: float = 3.0) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-winsor, winsor)


def get_defensive_signal(CSV) -> pd.Series:
    """
    Defensive (low-beta) signal: negated and z-scored beta.
    Lower beta → higher signal → more attractive long.
    Returns pd.Series indexed by Symbol.
    """
    df = pd.read_csv(CSV).dropna(subset=["Symbol", "Beta"])
    return _zscore(-df.set_index("Symbol")["Beta"]).rename("defensive")


def get_high_low_beta(CSV):
    signal = get_defensive_signal(CSV)
    df = signal.reset_index()
    df.columns = ["Symbol", "signal"]
    return df.nlargest(20, "signal"), df.nsmallest(20, "signal")
