import json
from pathlib import Path

import pandas as pd
import yfinance as yf

# --- 1. CONFIGURATION & DATA ---
START_DATE = "2006-01-01"  
END_DATE = "2025-12-31"
INTERVAL = "1mo"

BASE = Path("data")
BASE.mkdir(exist_ok=True)


def load_markets(path: str = "markets.json") -> dict:
    with open(path, "r") as f:
        markets = json.load(f)
    print("Loaded Markets:", markets)
    return markets


def download_prices(tickers, start, end, interval):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        # multi-index: ('Close', 'VTI') etc.
        if ("Close" in data.columns.get_level_values(0)):
            data = data["Close"]
        elif ("Adj Close" in data.columns.get_level_values(0)):
            data = data["Adj Close"]
    else:
        if "Close" in data.columns:
            data = data["Close"]
        elif "Adj Close" in data.columns:
            data = data["Adj Close"]

    return data.dropna(how="all")


def update_dataset(name, tickers):
    price_file = BASE / f"{name}_prices_monthly.csv"
    returns_file = BASE / f"{name}_returns_monthly.csv"

    if not price_file.exists():
        print(f"[NEW] Creating dataset for {name}")
        data = download_prices(tickers, START_DATE, END_DATE, INTERVAL)
    else:
        print(f"[UPDATE] Updating dataset for {name}")
        existing = pd.read_csv(price_file, index_col=0, parse_dates=True)
        last_date = existing.index.max()

        # Start from next month to avoid overlap
        start = (last_date + pd.offsets.MonthBegin()).strftime("%Y-%m-%d")
        new_data = download_prices(tickers, start, END_DATE, INTERVAL)

        data = pd.concat([existing, new_data])
        data = data[~data.index.duplicated(keep="last")]

    # Ensure all tickers exist
    missing = [t for t in tickers if t not in data.columns]
    if missing:
        raise ValueError(f"{name} -> Missing tickers: {missing}")

    # Clean rows where any asset missing
    data = data.dropna()

    # Save prices
    data.to_csv(price_file)

    # Recompute returns every time
    rets = data.pct_change().dropna()
    rets.to_csv(returns_file)

    print(f"{name}: updated through {data.index.max().date()}")


# ---------------- RUN ----------------
markets = load_markets()

for name, tickers in markets.items():
    update_dataset(name, tickers)

print("\nAll datasets updated successfully.")