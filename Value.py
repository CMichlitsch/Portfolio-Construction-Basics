from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_fundamentals.csv"

SP_pe_ratios = pd.read_csv(CSV_FILE)

clean_pe = SP_pe_ratios.dropna(subset = ["Trailing P/E"])
clean_pe = clean_pe[clean_pe["Trailing P/E"]>0]

def get_high_low_values(df):
    top_20_pe = clean_pe.nlargest(20, "Trailing P/E").sort_values(by = "Trailing P/E")
    bottom_20_pe = clean_pe.nsmallest(20, "Trailing P/E").sort_values(by = "Trailing P/E")
    print(top_20_pe, bottom_20_pe)
    return top_20_pe, bottom_20_pe

get_high_low_values(clean_pe)