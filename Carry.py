from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_fundamentals.csv"

SP_carry = pd.read_csv(CSV_FILE)

clean_carry = SP_carry.dropna(subset = ["Dividend Yield"])
clean_carry = clean_carry[clean_carry["Dividend Yield"]>0]

def get_high_low_yield(clean_matrix):
    top_20_winners = clean_matrix.nlargest(20, "Dividend Yield").sort_values(by = "Dividend Yield")
    top_20_losers = clean_matrix.nsmallest(20, "Dividend Yield").sort_values(by = "Dividend Yield")
    print(top_20_winners, top_20_losers)
    return top_20_losers, top_20_losers

get_high_low_yield(clean_carry)