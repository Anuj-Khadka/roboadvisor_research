from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")


def max_drawdown(values: pd.Series) -> float:
    peak = values.cummax()
    dd = values / peak - 1.0
    return float(dd.min())


def annualized_sharpe(monthly_returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    excess = monthly_returns - rf_monthly
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.sqrt(12) * excess.mean() / sd)


def run_constant_weight(prices: pd.DataFrame, target: dict, band: float = 0.05, initial: float = 10_000.0):
    tickers = list(target.keys())
    w_target = np.array([target[t] for t in tickers], dtype=float)
    w_target = w_target / w_target.sum()

    px = prices[tickers].dropna().copy()
    if px.empty:
        raise ValueError("No overlapping data after dropna(). Check tickers/date range.")

    # initial shares
    shares = (initial * w_target) / px.iloc[0].values

    vals_list, turnover_list, reb_list = [], [], []

    for dt, row in px.iterrows():
        vals = shares * row.values
        total = vals.sum()
        w_now = vals / total

        drift = np.abs(w_now - w_target)
        turnover = 0.0
        did_rebalance = False

        if (drift > band).any():
            target_vals = total * w_target
            new_shares = target_vals / row.values

            traded_dollars = np.sum(np.abs((new_shares - shares) * row.values))
            turnover = traded_dollars / total

            shares = new_shares
            did_rebalance = True

        vals_list.append(total)
        turnover_list.append(turnover)
        reb_list.append(did_rebalance)

    out = pd.DataFrame(
        {"portfolio_value": vals_list, "turnover": turnover_list, "rebalanced": reb_list},
        index=px.index,
    )

    out["portfolio_return"] = out["portfolio_value"].pct_change()
    out = out.dropna()

    stats = {
        "start": str(out.index.min().date()),
        "end": str(out.index.max().date()),
        "cagr": float((out["portfolio_value"].iloc[-1] / out["portfolio_value"].iloc[0]) ** (12 / len(out)) - 1),
        "vol_annual": float(out["portfolio_return"].std(ddof=1) * np.sqrt(12)),
        "sharpe": annualized_sharpe(out["portfolio_return"]),
        "max_drawdown": max_drawdown(out["portfolio_value"]),
        "avg_turnover": float(out["turnover"].mean()),
        "rebalance_rate": float(out["rebalanced"].mean()),
    }

    return out, stats


if __name__ == "__main__":
    prices_file = DATA_DIR / "primary_total_market_prices_monthly.csv"
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)

    # Balanced example (you can change later)
    target = {"VGT": 0.30, "VEU": 0.30, "BND": 0.35, "TIP": 0.05}

    out, stats = run_constant_weight(prices, target, band=0.05, initial=10_000)

    print("\nSTATS (Constant Weight Robo)")
    for k, v in stats.items():
        print(f"{k:>15}: {v}")

    out.to_csv(DATA_DIR / "primary_total_market_constant_weight_balanced.csv")
    print("\nSaved: data/primary_total_market_constant_weight_balanced.csv")
