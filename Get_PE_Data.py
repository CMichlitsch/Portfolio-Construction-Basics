from pathlib import Path
from time import sleep

import pandas as pd
import yfinance as yf


DATA_DIR = Path("Data")
COMPANIES_FILE = DATA_DIR / "sp500_companies.csv"
OUTPUT_FILE = DATA_DIR / "sp500_pe_ratios.csv"


def yahoo_symbol(symbol):
    return symbol.replace(".", "-")


def fetch_pe_data(symbols):
    rows = []

    for number, symbol in enumerate(symbols, start=1):
        yahoo_ticker = yahoo_symbol(symbol)

        try:
            info = yf.Ticker(yahoo_ticker).get_info()
        except Exception as error:
            print(f"Could not fetch {symbol}: {error}")
            info = {}

        rows.append(
            {
                "Symbol": symbol,
                "Yahoo Symbol": yahoo_ticker,
                "Company": info.get("shortName") or info.get("longName"),
                "Price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "Trailing P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "Market Cap": info.get("marketCap"),
                "Currency": info.get("currency"),
                "Quote Type": info.get("quoteType"),
            }
        )

        print(f"Fetched {number} of {len(symbols)}: {symbol}")
        sleep(0.25)

    return pd.DataFrame(rows)


sp500_companies = pd.read_csv(COMPANIES_FILE)
sp500_symbols = sp500_companies["Symbol"].dropna().drop_duplicates().tolist()

pe_df = fetch_pe_data(sp500_symbols)
pe_df = pe_df.merge(
    sp500_companies[["Symbol", "Sector", "Industry", "Weight"]],
    on="Symbol",
    how="left",
)

pe_df.to_csv(OUTPUT_FILE, index=False)

print(pe_df.head())
print(f"\nSaved {len(pe_df)} rows to {OUTPUT_FILE}")
