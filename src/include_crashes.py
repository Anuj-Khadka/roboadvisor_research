import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data")

for returns_file in DATA.glob("*_returns_monthly.csv"):
    name = returns_file.stem.replace("_returns_monthly","")
    rets = pd.read_csv(returns_file, index_col=0, parse_dates=True)

    wealth = (1 + rets).cumprod()

    wealth.plot(title=f"Cumulative Wealth (each ETF) — {name}")
    plt.tight_layout()
    plt.savefig(Path("outputs") / f"{name}_etf_wealth.png")
    plt.close()