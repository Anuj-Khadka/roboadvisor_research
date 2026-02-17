import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

MARKET_NAME = "primary_total_market"  # matches your markets.json key
RET_FILE = DATA_DIR / f"{MARKET_NAME}_returns_monthly.csv"

PROFILES_FILE = Path("profiles.json")

# --- rebalancing rule ---
BAND = 0.05  # ±5% drift band
START_VALUE = 10_000


def load_returns() -> pd.DataFrame:
    rets = pd.read_csv(RET_FILE, index_col=0, parse_dates=True)
    # Ensure numeric + sorted
    rets = rets.apply(pd.to_numeric, errors="coerce").dropna()
    rets = rets.sort_index()
    return rets


def load_profiles() -> dict:
    with open(PROFILES_FILE, "r") as f:
        profiles = json.load(f)
    return profiles


def normalize_weights(w: dict) -> dict:
    s = sum(w.values())
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: v / s for k, v in w.items()}


def portfolio_sim_constant_weight(returns: pd.DataFrame, target_w: dict):
    """
    Band rebalancing:
    - We let weights drift naturally each month.
    - If ANY asset weight drifts beyond target +/- BAND -> rebalance to targets.
    """
    tickers = list(target_w.keys())

    # align data to tickers
    r = returns[tickers].copy()

    # state
    value = START_VALUE
    weights = np.array([target_w[t] for t in tickers], dtype=float)
    weights = weights / weights.sum()

    values = []
    port_rets = []
    drawdowns = []
    rebalanced = []
    turnovers = []

    peak = value

    for dt, row in r.iterrows():
        # portfolio return for the month
        asset_r = row.values
        p_ret = float(np.dot(weights, asset_r))
        value *= (1.0 + p_ret)

        # weights drift after returns (before any rebalance)
        new_vals = weights * (1.0 + asset_r)
        if new_vals.sum() == 0:
            # extremely unlikely with ETFs, but guard anyway
            new_vals = np.ones_like(new_vals) / len(new_vals)
        weights = new_vals / new_vals.sum()

        # check drift
        target_vec = np.array([target_w[t] for t in tickers], dtype=float)
        drift = weights - target_vec
        need_rebal = np.any(np.abs(drift) > BAND)

        # turnover proxy: sum of absolute trades when we rebalance
        turnover = 0.0
        if need_rebal:
            turnover = float(np.sum(np.abs(weights - target_vec)))
            weights = target_vec.copy()

        # drawdown
        peak = max(peak, value)
        dd = (value / peak) - 1.0

        values.append(value)
        port_rets.append(p_ret)
        drawdowns.append(dd)
        rebalanced.append(1 if need_rebal else 0)
        turnovers.append(turnover)

    df = pd.DataFrame(
        {
            "value": values,
            "return": port_rets,
            "drawdown": drawdowns,
            "rebalanced": rebalanced,
            "turnover": turnovers,
        },
        index=r.index,
    )
    return df


def perf_stats(port_rets: pd.Series, values: pd.Series):
    # Monthly -> annualized (simple assumptions)
    r = port_rets.dropna()
    if len(r) < 12:
        return {}

    ann_return = (values.iloc[-1] / values.iloc[0]) ** (12 / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = (r.mean() * 12) / (r.std(ddof=1) * np.sqrt(12)) if r.std(ddof=1) > 0 else np.nan
    max_dd = float(((values / values.cummax()) - 1).min())

    return {
        "AnnReturn": ann_return,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
    }


def save_plots(df: pd.DataFrame, profile_name: str):
    """Save individual profile plots."""
    plots = [
        ("value", "Portfolio Value Over Time", "Portfolio Value ($)"),
        ("return", "Monthly Returns", "Return"),
        ("drawdown", "Drawdown Over Time", "Drawdown"),
    ]

    for col, title, ylabel in plots:
        plt.figure(figsize=(10, 5))
        if col == "return":
            plt.vlines(df.index, 0, df[col])
        else:
            plt.plot(df.index, df[col])
        plt.title(f"{title} — {profile_name}")
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{profile_name}_{col}.png", dpi=100)
        plt.close()

    # Rebalance events
    plt.figure(figsize=(10, 3))
    plt.scatter(df.index, df["rebalanced"], s=20, alpha=0.6)
    plt.title(f"Rebalancing Events (1 = Rebalanced) — {profile_name}")
    plt.xlabel("Date")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{profile_name}_rebalances.png", dpi=100)
    plt.close()


def save_comparison_plot(results: dict):
    """Save combined comparison plots for all profiles."""
    # Value comparison
    plt.figure(figsize=(12, 6))
    for pname, df in results.items():
        plt.plot(df.index, df["value"], label=pname, linewidth=2)
    plt.title("Portfolio Value Comparison — All Profiles")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_value.png", dpi=100)
    plt.close()

    # Drawdown comparison
    plt.figure(figsize=(12, 6))
    for pname, df in results.items():
        plt.plot(df.index, df["drawdown"], label=pname, linewidth=2)
    plt.title("Drawdown Comparison — All Profiles")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_drawdown.png", dpi=100)
    plt.close()


def main():
    returns = load_returns()
    profiles = load_profiles()

    all_stats = []
    results = {}

    for pname, w in profiles.items():
        w = normalize_weights(w)

        # quick safety check: profile tickers exist in returns file
        missing = [t for t in w.keys() if t not in returns.columns]
        if missing:
            raise ValueError(f"Profile '{pname}' uses tickers not in returns data: {missing}")

        df = portfolio_sim_constant_weight(returns, w)
        results[pname] = df

        # save full time series (so you can reuse later)
        df.to_csv(OUT_DIR / f"{pname}_timeseries.csv")

        # plots
        save_plots(df, pname)

        # stats
        stats = perf_stats(df["return"], df["value"])
        stats.update(
            {
                "Profile": pname,
                "Start": df.index.min().date(),
                "End": df.index.max().date(),
                "Rebalances": int(df["rebalanced"].sum()),
                "AvgTurnover": float(df["turnover"].mean()),
            }
        )
        all_stats.append(stats)

        print(f"[DONE] {pname} | rebalances={stats['Rebalances']}")

    # Save comparison plots
    save_comparison_plot(results)

    stats_df = pd.DataFrame(all_stats).set_index("Profile")
    # nicer formatting for print
    print("\n=== SUMMARY STATS ===")
    print(stats_df[["Start", "End", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "Rebalances", "AvgTurnover"]])

    stats_df.to_csv(OUT_DIR / "profiles_summary_stats.csv")
    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()