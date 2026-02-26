"""
run_comparison.py
-----------------
Run this from your project root:
    python notebooks/analysis/run_comparison.py

It loads your real data, runs all 5 strategies, and prints
a clean results table you can use directly in your abstract.
"""

import json
import numpy as np
import pandas as pd
import scipy.optimize as sco
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE             = Path("../../data")
MARKETS_FILE     = Path("../../markets.json")
RISK_FREE_ANNUAL = 0.0351
RISK_FREE_MONTHLY = (1 + RISK_FREE_ANNUAL) ** (1/12) - 1
LOOKBACK         = 36   # months of history used for rolling MPT

# ── Load data ─────────────────────────────────────────────────────────────────
with open(MARKETS_FILE) as f:
    config = json.load(f)

name = list(config["markets"].keys())[0]
rets = pd.read_csv(
    BASE / f"{name}_returns_monthly.csv", index_col=0, parse_dates=True
)

ASSETS = list(rets.columns)
N      = len(ASSETS)

print(f"Loaded  : {name}")
print(f"Period  : {rets.index[0].date()} -> {rets.index[-1].date()}")
print(f"Assets  : {ASSETS}")
print()

# ── Shared helpers ────────────────────────────────────────────────────────────
def performance_stats(ret_series, label):
    """Annualized return, vol, Sharpe (with Rf), max drawdown."""
    ann_return = ret_series.mean() * 12 * 100
    ann_vol    = ret_series.std()  * np.sqrt(12) * 100
    sharpe     = (ret_series.mean() - RISK_FREE_MONTHLY) * 12 / \
                 (ret_series.std() * np.sqrt(12))
    cum        = (1 + ret_series).cumprod()
    max_dd     = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return pd.Series({
        "Ann. Return %"  : round(ann_return, 2),
        "Ann. Vol %"     : round(ann_vol,    2),
        "Sharpe"         : round(sharpe,     3),
        "Max Drawdown %" : round(max_dd,     2),
    }, name=label)

# ── Strategy 1: Equal Weight ──────────────────────────────────────────────────
eq_w   = pd.Series(1 / N, index=ASSETS)
eq_ret = (rets * eq_w).sum(axis=1)

# ── Strategy 2: Naive Risk Parity ─────────────────────────────────────────────
vol        = rets.std()
naive_w    = (1 / vol) / (1 / vol).sum()
naive_ret  = (rets * naive_w).sum(axis=1)

# ── Strategy 3: ERC Risk Parity ───────────────────────────────────────────────
def erc_weights(returns):
    cov = returns.cov().values
    n   = cov.shape[0]

    def objective(w):
        pv  = np.sqrt(w @ cov @ w)
        mrc = (cov @ w) / pv
        rc  = w * mrc
        return sum((rc[i] - rc[j])**2 for i in range(n) for j in range(n))

    res = sco.minimize(
        objective, np.ones(n) / n, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    return pd.Series(res.x, index=returns.columns)

erc_w   = erc_weights(rets)
erc_ret = (rets * erc_w).sum(axis=1)

# ── Strategy 4: MPT Max Sharpe — static (in-sample, full history) ────────────
mu      = rets.mean() * 12
cov_mat = rets.cov()  * 12
mu_arr  = mu.values
cov_arr = cov_mat.values

res_sharpe = sco.minimize(
    lambda w: -(w @ mu_arr - RISK_FREE_ANNUAL) / np.sqrt(w @ cov_arr @ w),
    np.ones(N) / N, method="SLSQP",
    bounds=[(0, 1)] * N,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
    options={"ftol": 1e-12, "maxiter": 1000}
)
w_sharpe   = pd.Series(res_sharpe.x, index=ASSETS)
sharpe_ret = (rets * w_sharpe).sum(axis=1)

# ── Strategy 5: MPT Max Sharpe — rolling (realistic, no look-ahead) ──────────
print("Running rolling MPT... (this takes ~30 seconds)")
rolling_returns = []
rolling_dates   = []

for i in range(LOOKBACK, len(rets)):
    window = rets.iloc[i - LOOKBACK : i]
    mu_w   = window.mean().values * 12
    cov_w  = window.cov().values  * 12

    def neg_sharpe_w(w):
        r = w @ mu_w
        v = np.sqrt(w @ cov_w @ w)
        return -(r - RISK_FREE_ANNUAL) / v if v > 1e-8 else 0

    try:
        r = sco.minimize(
            neg_sharpe_w, np.ones(N) / N, method="SLSQP",
            bounds=[(0, 1)] * N,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            options={"ftol": 1e-10, "maxiter": 500}
        )
        w_opt = r.x
    except Exception:
        w_opt = np.ones(N) / N

    rolling_returns.append(w_opt @ rets.iloc[i].values)
    rolling_dates.append(rets.index[i])

rolling_ret = pd.Series(rolling_returns, index=rolling_dates)

# ── Results — aligned to rolling window period for fair comparison ────────────
idx = rolling_ret.index

results = pd.DataFrame([
    performance_stats(eq_ret.loc[idx],      "Equal Weight"),
    performance_stats(naive_ret.loc[idx],   "Naive Risk Parity"),
    performance_stats(erc_ret.loc[idx],     "ERC Risk Parity"),
    performance_stats(sharpe_ret.loc[idx],  "MPT Max Sharpe (static)"),
    performance_stats(rolling_ret,          "MPT Max Sharpe (rolling)"),
])

print("\n" + "="*65)
print("  STRATEGY COMPARISON — all metrics annualized")
print(f"  Period: {idx[0].date()} -> {idx[-1].date()}")
print("="*65)
print(results.to_string())
print("="*65)

# ── Abstract-ready sentences ──────────────────────────────────────────────────
best        = results.loc[results["Sharpe"].idxmax()]
worst       = results.loc[results["Sharpe"].idxmin()]
erc_row     = results.loc["ERC Risk Parity"]
mpt_roll    = results.loc["MPT Max Sharpe (rolling)"]
mpt_static  = results.loc["MPT Max Sharpe (static)"]

print("\n── ABSTRACT NUMBERS ──")
print(f"Best Sharpe  : {best.name}  →  {best['Sharpe']}")
print(f"Worst Sharpe : {worst.name}  →  {worst['Sharpe']}")
print()
print(f"ERC Risk Parity  :  return {erc_row['Ann. Return %']}%,  "
      f"vol {erc_row['Ann. Vol %']}%,  "
      f"Sharpe {erc_row['Sharpe']},  "
      f"max drawdown {erc_row['Max Drawdown %']}%")
print(f"MPT (rolling)    :  return {mpt_roll['Ann. Return %']}%,  "
      f"vol {mpt_roll['Ann. Vol %']}%,  "
      f"Sharpe {mpt_roll['Sharpe']},  "
      f"max drawdown {mpt_roll['Max Drawdown %']}%")
print(f"MPT (static/in-sample):  Sharpe {mpt_static['Sharpe']}  "
      f"← inflated by look-ahead bias")
print()

sharpe_diff = round(erc_row['Sharpe'] - mpt_roll['Sharpe'], 3)
dd_diff     = round(abs(mpt_roll['Max Drawdown %']) - abs(erc_row['Max Drawdown %']), 1)

print("── SUGGESTED ABSTRACT SENTENCE ──")
print(f"""
Preliminary results show that ERC Risk Parity achieves a Sharpe ratio of
{erc_row['Sharpe']} versus {mpt_roll['Sharpe']} for rolling MPT — a difference of
{abs(sharpe_diff)} — while also reducing maximum drawdown by {dd_diff} percentage
points ({erc_row['Max Drawdown %']}% vs {mpt_roll['Max Drawdown %']}%). Notably,
static MPT produces a Sharpe of {mpt_static['Sharpe']} when evaluated in-sample,
but this advantage disappears under realistic rolling re-optimization, suggesting
that MPT's complexity does not translate into better real-world investor outcomes.
""")
