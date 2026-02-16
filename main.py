import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIGURATION & DATA ---
start_date = '2008-01-01'  # Includes the Great Financial Crisis to test drawdown
end_date = '2025-12-31'
interval = '1mo'

# risk_free_rate = 0.03      # 3% annual risk-free rate assumption

BASE = Path("data")
BASE.mkdir(exist_ok=True)

with open("markets.json", "r") as f:
    markets = json.load(f)
    print("Loaded Markets:", markets)

for name, tickers in markets.items():
    print(f"Downloading data for {name} ({', '.join(tickers)})...")
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=False)["Adj Close"]
    
    data = data.dropna(how="all")

    missing = [t for t in tickers if t not in data.columns]
    if missing:
        raise ValueError(f"{name} -> Missing data for tickers: {missing}")
    
    data = data.dropna()
    
    data.to_csv(BASE / f"{name}_prices_monthly.csv")

    rets = data.pct_change().dropna()
    rets.to_csv(BASE / f"{name}_returns_monthly.csv")

print("data download complete.")









# # Risk Profiles (Equity Weight, Bond Weight)
# profiles = {
#     'Conservative': [0.20, 0.80],
#     'Moderate':     [0.60, 0.40],
#     'Aggressive':   [0.80, 0.20]
# }

# # DIY Execution Probabilities (Probability of SUCCESSFUL rebalance)
# # p = 1.0 is effectively the Robo-Advisor
# execution_probs = [1.0, 0.75, 0.50]

# # Monte Carlo Settings for DIY
# n_simulations = 1000  # Number of distinct "lifetimes" to simulate for each probability

# def get_data(tickers, start, end):
#     data = yf.download(tickers, start=start, end=end)
#     # For multiple tickers, select the 'Close' price level (adjusted close)
#     if isinstance(data.columns, pd.MultiIndex):
#         # The MultiIndex has (Price, Ticker) format - we want the 'Close' level
#         data = data['Close']
#     else:
#         # Single ticker case
#         data = data[['Adj Close']]
#     returns = data.pct_change().dropna()
#     return returns

# # Fetch Data
# print("Downloading Data...")
# returns_df = get_data(tickers, start_date, end_date)

# # Resample to Quarterly (Rebalancing Schedule)
# # We calculate quarterly returns to match the decision frequency
# q_returns = returns_df.resample('Q').apply(lambda x: (x + 1).prod() - 1)

# # --- 2. THE SIMULATION ENGINE ---

# def run_portfolio_sim(returns, target_weights, p_execute):
#     """
#     Simulates one investor's path.
#     returns: DataFrame of quarterly asset returns
#     target_weights: list [equity_w, bond_w]
#     p_execute: probability of rebalancing at any given quarter
#     """
#     n_quarters = len(returns)
#     portfolio_values = [100.0] # Start with $100
    
#     # Initialize current weights to target
#     current_weights = np.array(target_weights)
    
#     for i in range(n_quarters):
#         # 1. Calculate growth for this quarter based on current weights
#         r_equity = returns.iloc[i]['SPY']
#         r_bond = returns.iloc[i]['AGG']
        
#         # Portfolio return this quarter
#         port_return = (current_weights[0] * r_equity) + (current_weights[1] * r_bond)
#         new_value = portfolio_values[-1] * (1 + port_return)
#         portfolio_values.append(new_value)
        
#         # 2. Update weights (Drift)
#         # Weights change naturally as assets grow/shrink at different rates
#         equity_val = current_weights[0] * (1 + r_equity)
#         bond_val = current_weights[1] * (1 + r_bond)
#         total_val = equity_val + bond_val
        
#         drifted_weights = np.array([equity_val / total_val, bond_val / total_val])
        
#         # 3. Decision Node: To Rebalance or Not?
#         # Generate random check against p_execute
#         if np.random.random() < p_execute:
#             # SUCCESS: Reset to target
#             current_weights = np.array(target_weights)
#         else:
#             # FAIL: Keep drifted weights for next quarter
#             current_weights = drifted_weights
            
#     return pd.Series(portfolio_values)

# def calculate_metrics(daily_series):
#     """Calculates Sharpe and Max Drawdown from value series"""
#     # Convert value series to returns
#     rets = daily_series.pct_change().dropna()
    
#     # Annualized Return (Geometric)
#     total_ret = (daily_series.iloc[-1] / daily_series.iloc[0]) - 1
#     n_years = len(daily_series) / 4 # Quarterly data
#     ann_ret = (1 + total_ret)**(1/n_years) - 1
    
#     # Annualized Volatility
#     ann_vol = rets.std() * np.sqrt(4) 
    
#     # Sharpe Ratio
#     sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    
#     # Max Drawdown
#     rolling_max = daily_series.cummax()
#     drawdown = (daily_series - rolling_max) / rolling_max
#     max_dd = drawdown.min()
    
#     return sharpe, max_dd

# # --- 3. RUNNING THE EXPERIMENT ---

# results = []

# print(f"Running Simulations ({n_simulations} runs per profile)...")

# for profile_name, weights in profiles.items():
#     for p in execution_probs:
#         sharpes = []
#         mdds = []
        
#         # If p=1.0 (Robo), we strictly don't need Monte Carlo (it's deterministic),
#         # but we run once for consistency in data structure.
#         loops = 1 if p == 1.0 else n_simulations
        
#         for _ in range(loops):
#             sim_path = run_portfolio_sim(q_returns, weights, p)
#             s, m = calculate_metrics(sim_path)
#             sharpes.append(s)
#             mdds.append(m)
        
#         # Aggregate results
#         results.append({
#             'Profile': profile_name,
#             'Execution Prob (p)': p,
#             'Avg Sharpe': np.mean(sharpes),
#             'Avg Max Drawdown': np.mean(mdds),
#             'Worst Case MDD': np.min(mdds) # The worst outcome in 1000 lifetimes
#         })

# # --- 4. DISPLAY RESULTS ---
# results_df = pd.DataFrame(results)

# # Formatting for readability
# print("\n=== Research Results: Robo (p=1.0) vs DIY (p<1.0) ===")
# print(results_df.sort_values(by=['Profile', 'Execution Prob (p)'], ascending=[True, False]).to_string(index=False))

# # Optional: Plotting a specific case (e.g., Moderate)
# subset = results_df[results_df['Profile'] == 'Moderate']
# print("\n--- Deep Dive: Moderate Profile ---")
# print(subset[['Execution Prob (p)', 'Avg Sharpe', 'Avg Max Drawdown']])