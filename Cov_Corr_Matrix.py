from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("Data")
CSV_FILE = DATA_DIR / "sp500_stocks.csv"

SP_stocks = pd.read_csv(CSV_FILE)

clean_SP = SP_stocks.dropna(subset=["Date", "Symbol", "High", "Low"]).copy()
clean_SP["Avg High Low"] = (clean_SP["High"] + clean_SP["Low"]) / 2

avg_high_low_by_symbol = clean_SP.pivot(
    index="Date",
    columns="Symbol",
    values="Avg High Low",
)

cov_matrix = avg_high_low_by_symbol.cov()
corr_matrix = avg_high_low_by_symbol.corr()

upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_pairs = (
    corr_matrix.rename_axis(index="Ticker 1", columns="Ticker 2")
    .where(upper_triangle)
    .stack()
    .rename("Correlation")
    .reset_index()
)

highest_corr = corr_pairs.nlargest(20, "Correlation")
lowest_corr = corr_pairs.nsmallest(20, "Correlation")
selected_corr = pd.concat([highest_corr, lowest_corr], ignore_index=True)
print(selected_corr)

tickers = sorted(set(selected_corr["Ticker 1"]) | set(selected_corr["Ticker 2"]))
ticker_to_number = {ticker: number for number, ticker in enumerate(tickers)}

x = selected_corr["Ticker 1"].map(ticker_to_number)
y = selected_corr["Ticker 2"].map(ticker_to_number)
z = selected_corr["Correlation"]

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x,
    y,
    z,
    c=z,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    s=90,
)

for x_value, y_value, z_value in zip(x, y, z):
    ax.plot([x_value, x_value], [y_value, y_value], [0, z_value], color="gray", alpha=0.3)

fig.colorbar(scatter, ax=ax, shrink=0.65, label="Correlation")

ax.set_xticks(range(len(tickers)))
ax.set_yticks(range(len(tickers)))

ax.set_xticklabels(tickers, rotation=90)
ax.set_yticklabels(tickers)

ax.set_xlabel("Ticker 1")
ax.set_ylabel("Ticker 2")
ax.set_zlabel("Correlation")
ax.set_title("20 Highest and 20 Lowest Stock Correlations")
ax.set_zlim(-1, 1)

plt.tight_layout()
plt.show()
