from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_fundamentals.csv"

anti_beta = pd.read_csv(CSV_FILE)
clean_beta = anti_beta.dropna(subset= ["Beta"])

def get_high_low_beta(clean_SP_matrix):
    top_20_winners = clean_SP_matrix.nlargest(20, "Beta").sort_values(by = "Beta")
    top_20_losers = clean_SP_matrix.nsmallest(20, "Beta").sort_values(by = "Beta")
    print(top_20_winners, top_20_losers)
    return top_20_losers, top_20_losers

get_high_low_beta(clean_beta)