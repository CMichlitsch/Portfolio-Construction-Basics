from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_stocks.csv"

SP_stocks = pd.read_csv(CSV_FILE)
SP_stocks["Date"] = pd.to_datetime(SP_stocks["Date"])
clean_SP = SP_stocks.dropna(subset=["Date", "Symbol", "High", "Low"]).copy()
clean_SP["Mid Price"] = (clean_SP["High"] + clean_SP["Low"]) / 2

clean_SP = clean_SP.sort_values(["Symbol", "Date"])
clean_SP["Mid Price 1M Ago"] = clean_SP.groupby("Symbol")["Mid Price"].shift(21)

clean_SP["Mid Price Difference 1M"] = (clean_SP["Mid Price"] - clean_SP["Mid Price 1M Ago"])
clean_SP["1M Percent Change"] = clean_SP["Mid Price Difference 1M"]/clean_SP["Mid Price 1M Ago"]

top_20_winners = clean_SP.nlargest(20, "1M Percent Change").sort_values(by = "1M Percent Change")
top_20_losers = clean_SP.nsmallest(20, "1M Percent Change").sort_values(by = "1M Percent Change")