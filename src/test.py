import pandas as pd

file = "data/outputs/primary_total_market_risk_parity_conservative.csv"

df = pd.read_csv(file, parse_dates=["Date"])

weights = df[["W_VGT","W_VEU","W_BND","W_TIP"]]

print("\nAverage allocation (%)")
print((weights.mean()*100).round(2))

print("\nMin allocation (%)")
print((weights.min()*100).round(2))

print("\nMax allocation (%)")
print((weights.max()*100).round(2))

# Which asset dominated each month
dominant = weights.idxmax(axis=1).value_counts(normalize=True)*100
print("\nDominant asset frequency (%)")
print(dominant.round(2))
