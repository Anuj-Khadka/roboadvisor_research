from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")

file = DATA_DIR / "primary_total_market_constant_weight_balanced.csv"
# file = DATA_DIR / "primary_total_market_prices_monthly.csv"
df = pd.read_csv(file, index_col=0, parse_dates=True)

# --- 1) Portfolio growth ---
plt.figure(figsize=(10,5))
plt.plot(df.index, df["portfolio_value"])
plt.title("Portfolio Value Over Time (Constant Weight Balanced)")
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 2) Monthly returns ---
plt.figure(figsize=(10,4))
plt.bar(df.index, df["portfolio_return"])
plt.title("Monthly Returns")
plt.ylabel("Return")
plt.xlabel("Date")
plt.tight_layout()
plt.show()


# --- 3) Drawdown curve ---
peak = df["portfolio_value"].cummax()
drawdown = df["portfolio_value"] / peak - 1

plt.figure(figsize=(10,5))
plt.plot(df.index, drawdown)
plt.title("Drawdown Over Time")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 4) Rebalancing activity ---
plt.figure(figsize=(10,3))
plt.scatter(df.index, df["rebalanced"])
plt.title("Rebalancing Events (1 = Rebalanced)")
plt.yticks([0,1])
plt.tight_layout()
plt.show()
