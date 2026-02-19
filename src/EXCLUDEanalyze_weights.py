import pandas as pd
from pathlib import Path

BASE = Path("results")   # change if your weights saved elsewhere

def analyze_weights(file):
    print("\n==============================")
    print("FILE:", file.name)

    w = pd.read_csv(file, index_col=0, parse_dates=True)

    # Average weight
    print("\nAverage weight:")
    print((w.mean() * 100).round(2))

    # Min / Max weight
    print("\nMin weight:")
    print((w.min() * 100).round(2))

    print("\nMax weight:")
    print((w.max() * 100).round(2))

    # Check concentration
    max_asset = w.idxmax(axis=1)
    dominance = max_asset.value_counts(normalize=True) * 100

    print("\n% of months each asset dominated:")
    print(dominance.round(2))


for file in BASE.glob("*risk_parity_weights*.csv"):
    analyze_weights(file)
