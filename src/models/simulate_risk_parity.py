from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKETS_PATH = Path("markets.json")
PROFILES_PATH = Path("profiles.json")


# ---------------------------
# Helpers
# ---------------------------
def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_drawdown(value: pd.Series) -> pd.Series:
    peak = value.cummax()
    return (value / peak) - 1.0


def inv_vol_weights(returns_window: pd.DataFrame, risk_budget: dict | None = None) -> pd.Series:
    """
    Risk parity-ish: weight ∝ (risk_budget / volatility).
    If risk_budget is None, equal risk budget.
    """
    vol = returns_window.std()
    vol = vol.replace(0, np.nan).dropna()

    if vol.empty:
        return pd.Series(dtype=float)

    if risk_budget is None:
        rb = pd.Series(1.0, index=vol.index)
    else:
        rb = pd.Series({k: float(v) for k, v in risk_budget.items()})
        rb = rb.reindex(vol.index).fillna(0.0)

    raw = rb / vol
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    s = raw.sum()
    if s <= 0:
        return pd.Series(0.0, index=vol.index)

    return raw / s


def apply_caps(w: pd.Series, max_weight: float | None) -> pd.Series:
    if max_weight is None:
        return w
    max_weight = float(max_weight)

    # cap + re-normalize iteratively (simple and stable for small N)
    w = w.copy()
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if under.any() and excess > 0:
            w[under] = w[under] + excess * (w[under] / w[under].sum())
        else:
            break

    # final normalize
    total = w.sum()
    if total > 0:
        w /= total
    return w


# ---------------------------
# Simulation
# ---------------------------
def simulate_risk_parity(
    prices: pd.DataFrame,
    profile_name: str,
    profile_cfg: dict,
    market_name: str,
    lookback_months: int = 12,
):
    """
    Monthly strategy:
    - compute rolling vol over last lookback_months
    - compute inverse-vol weights (optionally with risk budgets)
    - optional band rule to avoid rebalancing unless drift > band
    """
    prices = prices.sort_index().copy()

    # returns used for vol
    rets = prices.pct_change().dropna()

    # We need enough history for the lookback window
    start_idx = lookback_months
    if len(rets) <= start_idx:
        raise ValueError("Not enough data for lookback window.")

    start_value = float(profile_cfg.get("start_value", 10_000))
    band = float(profile_cfg.get("band", 0.0))  # 0 = rebalance every month
    max_weight = profile_cfg.get("max_weight", None)

    # Optional risk budget: {"VGT": 1, "VEU": 1, "BND": 1, "TIP": 1}
    risk_budget = profile_cfg.get("risk_budget", None)

    dates = rets.index[start_idx:]

    # portfolio state
    wealth = start_value
    current_weights = pd.Series(0.0, index=prices.columns)

    out_rows = []
    rebalance_flags = []

    for t in range(start_idx, len(rets)):
        date = rets.index[t]

        # rolling window for vol
        window = rets.iloc[t - lookback_months : t]
        target_w = inv_vol_weights(window, risk_budget=risk_budget)
        target_w = target_w.reindex(prices.columns).fillna(0.0)
        target_w = apply_caps(target_w, max_weight=max_weight)

        # current portfolio return this month given current weights
        r_t = float((current_weights * rets.iloc[t]).sum())
        wealth *= (1.0 + r_t)

        # compute drift (how far current is from target)
        drift = (current_weights - target_w).abs().max()

        did_rebalance = False
        turnover = 0.0

        # rebalance decision
        if (band == 0.0) or (drift > band):
            # turnover proxy: sum(|Δw|)/2
            turnover = float((current_weights - target_w).abs().sum() / 2.0)
            current_weights = target_w
            did_rebalance = True

        rebalance_flags.append(1 if did_rebalance else 0)

        out_rows.append(
            {
                "Date": date,
                "Wealth": wealth,
                "MonthlyReturn": r_t,
                "Turnover": turnover,
                **{f"W_{c}": float(current_weights[c]) for c in prices.columns},
            }
        )

    df = pd.DataFrame(out_rows).set_index("Date")
    df["Drawdown"] = compute_drawdown(df["Wealth"])
    df["Rebalanced"] = rebalance_flags

    # Save
    out_file = OUT_DIR / f"{market_name}_risk_parity_{profile_name}.csv"
    df.to_csv(out_file)

    # ---------------------------
    # Plots (same vibe as your baseline)
    # ---------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Wealth"])
    plt.title(f"Portfolio Value Over Time (Risk Parity — {profile_name})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["MonthlyReturn"])
    plt.title(f"Monthly Returns (Risk Parity — {profile_name})")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Drawdown"])
    plt.title(f"Drawdown Over Time (Risk Parity — {profile_name})")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 2.5))
    plt.scatter(df.index, df["Rebalanced"], s=12)
    plt.title(f"Rebalancing Events (1 = Rebalanced) — {profile_name}")
    plt.xlabel("Date")
    plt.ylabel("Rebalanced")
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.show()

    # Quick stats in terminal
    ann_ret = (df["Wealth"].iloc[-1] / df["Wealth"].iloc[0]) ** (12 / len(df)) - 1
    ann_vol = df["MonthlyReturn"].std() * np.sqrt(12)
    sharpe = (df["MonthlyReturn"].mean() * 12) / (df["MonthlyReturn"].std() * np.sqrt(12) + 1e-12)
    max_dd = df["Drawdown"].min()

    print("\nSTATS (Risk Parity Robo)")
    print(f"  market: {market_name}")
    print(f"  profile: {profile_name}")
    print(f"  start: {df.index.min().date()}")
    print(f"  end:   {df.index.max().date()}")
    print(f"  ann_return: {ann_ret:.3%}")
    print(f"  ann_vol:    {ann_vol:.3%}")
    print(f"  sharpe:     {sharpe:.2f}")
    print(f"  max_dd:     {max_dd:.3%}")
    print(f"  avg_turnover/month: {df['Turnover'].mean():.4f}")
    print(f"  rebalance_rate: {df['Rebalanced'].mean():.2%}")
    print(f"  saved: {out_file}")

    return df


# ---------------------------
# RUN
# ---------------------------
def main():
    markets = load_json(MARKETS_PATH)
    profiles = load_json(PROFILES_PATH)

    # For each market, load the prices CSV you already create in main.py
    for market_name, tickers in markets.items():
        price_file = DATA_DIR / f"{market_name}_prices_monthly.csv"
        if not price_file.exists():
            raise FileNotFoundError(f"Missing: {price_file}. Run your data downloader first.")

        prices = pd.read_csv(price_file, index_col=0, parse_dates=True)

        # Ensure correct column ordering / presence
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"{market_name} -> Missing tickers in CSV: {missing}")

        prices = prices[tickers].dropna()

        for profile_name, profile_cfg in profiles.items():
            simulate_risk_parity(
                prices=prices,
                profile_name=profile_name,
                profile_cfg=profile_cfg,
                market_name=market_name,
                lookback_months=int(profile_cfg.get("lookback_months", 12)),
            )


if __name__ == "__main__":
    main()
