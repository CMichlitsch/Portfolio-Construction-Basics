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


def get_value_signal(CSV) -> pd.Series:
    """
    Value signal: average z-score of trailing and forward earnings yield (E/P).
    Higher signal = cheaper stock (more attractive long).
    Returns pd.Series indexed by Symbol.
    """
    df = pd.read_csv(CSV).dropna(subset=["Symbol"])

    parts = []

    ep = df[df["Trailing P/E"].notna() & (df["Trailing P/E"] > 0)].copy()
    ep["EP"] = 1.0 / ep["Trailing P/E"]
    parts.append(ep.set_index("Symbol")["EP"].pipe(_zscore))

    fep = df[df["Forward P/E"].notna() & (df["Forward P/E"] > 0)].copy()
    fep["FEP"] = 1.0 / fep["Forward P/E"]
    parts.append(fep.set_index("Symbol")["FEP"].pipe(_zscore))

    combined = pd.concat(parts, axis=1).mean(axis=1)
    return _zscore(combined).rename("value")


# legacy export kept for any existing callers
def get_high_low_values(CSV):
    signal = get_value_signal(CSV)
    df = signal.reset_index()
    df.columns = ["Symbol", "signal"]
    return df.nlargest(20, "signal"), df.nsmallest(20, "signal")
