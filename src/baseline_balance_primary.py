import pandas as pd
import matplotlib.pyplot as plt

rets = pd.read_csv("data/primary_total_market_returns_monthly.csv", index_col=0, parse_dates=True)

w = pd.Series({"VGT": 0.42, "VXUS": 0.18, "BND": 0.32, "TIP": 0.08})  # sums to 1.0
port_rets = (rets * w).sum(axis=1)
wealth = (1 + port_rets).cumprod()

wealth.plot(title="Baseline Balanced Portfolio Wealth — primary_total_market")
plt.tight_layout()
plt.savefig("outputs/baseline_balanced_primary.png")
plt.close()


