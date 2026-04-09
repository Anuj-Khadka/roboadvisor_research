# Robo-Advisors and the Investor Experience
### Anuj Khadka & Dr. Vladislav D. Veksler — Caldwell University

---

## Research Question

Does algorithmic sophistication in portfolio construction produce meaningfully different investor outcomes when the underlying asset universe is fixed?

---

## Background

Asset allocation policy is the dominant driver of portfolio performance. Robo-advisors have automated and scaled this process, but platforms market their algorithms — not their asset universes — as the key differentiator. This study holds the asset universe fixed across seven strategies to isolate whether the algorithm produces meaningfully different investor outcomes.

---

## Asset Universe — 6 ETFs

| Ticker | Name | Asset Class |
|--------|------|-------------|
| BND | Vanguard Total Bond Market | US Aggregate Bonds |
| GLD | SPDR Gold Shares | Gold |
| TIP | iShares TIPS Bond | Inflation-Protected Bonds |
| TLT | iShares 20+ Year Treasury | Long-Term Treasuries |
| VEU | Vanguard FTSE All-World ex-US | International Equities |
| VGT | Vanguard Information Technology | US Technology |

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Initial Investment | $1,000 (lump sum) |
| Comparison Period | May 2010 – December 2025 |
| Total Months | 188 months |
| Rolling Lookback Window | 36 months |
| Rebalancing Threshold | 5% drift |
| Volatility Target (Markowitz & FSO) | 10% annualized |
| Risk-Free Rate | FRED TB3MS (time-varying, monthly) |

---

## The 7 Strategies

### 1. Equal Weight
Allocates 1/N (16.7%) to each of the 6 assets. No optimization. Rebalanced monthly. Serves as the human-judgment baseline.

### 2. Sample Portfolio
Static allocation: BND 20%, GLD 10%, TIP 10%, TLT 10%, VEU 25%, VGT 25%. Rebalances only when any weight drifts more than 5% from target. No optimization, no model.

### 3. ERC RP — Equal Risk Contribution
Risk parity model. Weights are computed so each asset contributes equally to total portfolio variance. Uses Ledoit-Wolf shrinkage on the covariance matrix. Rolling 36-month window. Long-only, weights sum to 1 (no leverage).

### 4. Markowitz (MPT)
Classic mean-variance optimization. Maximizes expected return subject to a hard 10% annualized volatility constraint. Rolling 36-month window. Covariance and mean estimated from historical returns.

### 5. Black-Litterman
Bayesian model that blends market-implied equilibrium returns with forward-looking investor views. Market-cap weights sourced from State Street Global Market Portfolio 2025 (Figure 3). Views sourced from State Street GMP 2025 (Figure 9). Views are fixed across the full period — the covariance updates rolling every 36 months. Maximizes Sharpe ratio on posterior expected returns.

**Market Cap Weights (BND/GLD/TIP/TLT/VEU/VGT):** 0.11 / 0.06 / 0.02 / 0.26 / 0.20 / 0.35

**Investor Views (annualized):** 4.9% / 2.5% / 3.9% / 4.3% / 7.5% / 7.5%

### 6. Factor Investing
4-factor composite model. Each asset is scored on four factors, z-score normalized, equally weighted (25% each), then converted to portfolio weights via min-max shifting.

| Factor | Window | Description |
|--------|--------|-------------|
| Momentum | t-13 to t-2 | 12-month return, skipping most recent month |
| Value | t-36 to t-13 | Older price history, mean-reversion signal |
| Quality | Full 36 months | Sharpe-like ratio over full lookback |
| Low Volatility | Last 12 months | Inverse of annualized vol |

### 7. FSO — Full Scale Optimization (γ = 3.0)
Maximizes expected CRRA (Constant Relative Risk Aversion) utility over the full empirical return distribution — not just mean and variance. Unlike MPT, FSO captures skewness, kurtosis, and fat tails.

**Utility Function:**
```
U(W) = W^(1 - γ) / (1 - γ)    for γ ≠ 1
U(W) = log(W)                   for γ = 1
```

γ = 3.0 represents a moderately loss-averse investor (Mehra & Prescott, 1985). FSO is **constrained to 10% annualized volatility** — the same constraint as Markowitz — to ensure a fair, apples-to-apples comparison. Without this constraint, FSO concentrates into a single asset (corner solution), which is not appropriate for a retail robo-advisor context.

---

## Performance Results (May 2010 – December 2025)

| Strategy | CAGR % | Vol % | Sharpe | Sortino | Max DD % | Final Value |
|----------|--------|-------|--------|---------|----------|-------------|
| Equal Weight | 7.37 | 8.09 | 0.750 | 1.232 | −21.17 | $3,048 |
| Sample Portfolio | 8.89 | 9.33 | 0.813 | 1.351 | −22.80 | $3,795 |
| ERC RP | 6.53 | 6.95 | 0.746 | 1.202 | −19.99 | $2,692 |
| Factor Investing | 7.38 | 7.34 | 0.820 | 1.392 | −17.81 | $3,053 |
| Black-Litterman | 8.40 | 10.42 | 0.694 | 1.180 | −26.88 | $3,539 |
| FSO (γ=3.0) | 10.25 | 10.48 | 0.854 | 1.457 | −22.95 | $4,616 |
| Markowitz | 10.46 | 10.44 | 0.875 | 1.502 | −22.94 | $4,749 |

> **Note:** CAGR is true geometric: `(∏(1+r_t))^(12/T) − 1`.
> Sortino uses full-sample downside deviation (Sortino & van der Meer, 1991): `√(mean(min(r−rf, 0)²)) × √12`.
> FSO vol-constrained to 10% annualized — same constraint as Markowitz.

---

## Overall Rankings (1 = best)

| Strategy | Return | Vol | Sharpe | Sortino | Max DD | Avg Rank |
|----------|--------|-----|--------|---------|--------|----------|
| **Factor Investing** | 5 | 3 | 3 | 3 | **1** | **3.0** |
| Markowitz | 1 | 8 | 1 | 1 | 7 | 3.6 |
| Sample Portfolio | 3 | 5 | 4 | 4 | 5 | 4.2 |
| FSO (γ=3.0) | 2 | 9 | 2 | 2 | 8 | 4.6 |
| ERC RP | 7 | 2 | 6 | 6 | 3 | 4.8 |
| Equal Weight | 6 | 4 | 5 | 5 | 4 | 4.8 |
| Black-Litterman | 4 | 7 | 7 | 7 | 9 | 6.8 |

> **Overall winner: Factor Investing** (avg rank 3.0) — the only strategy to rank top-3 in both return quality and downside protection simultaneously.

---

## Return Distribution

| Strategy | Skewness | Excess Kurtosis |
|----------|----------|-----------------|
| Equal Weight | −0.146 | 0.694 |
| Sample Portfolio | −0.156 | 0.414 |
| ERC RP | −0.278 | 1.029 |
| Factor Investing | −0.062 | 0.210 |
| **Black-Litterman** | **+0.273** | **3.786** |
| FSO (γ=3.0) | −0.112 | −0.243 |
| Markowitz | −0.110 | −0.210 |

> Black-Litterman is the **only strategy with positive skewness** — meaning upside surprises are more likely than downside ones. However, its excess kurtosis of 3.786 indicates fat tails in both directions.

---

## Crisis Recovery Table

| Crisis | Period | EW | Sample | ERC | Factor | BL | FSO | Markowitz |
|--------|--------|----|--------|-----|--------|----|-----|-----------|
| GFC | 2008–2009 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| COVID | 2020-02 to 2020-04 | 1m | 2m | 1m | 1m | 1m | 1m | 1m |
| Rate Hikes | 2022-01 to 2022-12 | 21m | 18m | 23m | 17m | **9m** | 22m | 22m |

> GFC shows N/A because the 36-month rolling warmup means all strategies begin in May 2010, after the March 2009 trough.
> Black-Litterman recovered from the 2022 Rate Hike crisis in 9 months — less than half the time of every other strategy.

---

## Key Findings

**1. Simplicity Wins.**
A static portfolio (Sample Portfolio) outperformed ERC RP, Equal Weight, and Black-Litterman with zero optimization or model risk.

**2. Factor Investing Had the Smallest Loss.**
At −17.8% max drawdown and avg rank 3.0 across all metrics, Factor Investing was the only strategy to rank top-3 in both return quality and downside protection simultaneously.

**3. Complexity Without Constraint Is Misleading.**
Markowitz and FSO — built on fundamentally different mathematical frameworks — produced nearly identical outcomes at equal risk (10.46% vs 10.25% CAGR, −22.94% vs −22.95% max drawdown). The algorithm did not drive the result. The risk budget did.

**4. The Most Sophisticated Model Fell Deepest but Recovered Fastest.**
Black-Litterman fell the hardest (−26.9%) but recovered in 9 months — forward-looking views outperform backward-looking models only under stress.

---

## Conclusion

The asset universe — not the algorithm — drives investor outcomes. A simple, static portfolio outperformed three complex strategies with zero optimization. When FSO and Markowitz were held to the same 10% risk budget, the fancier utility optimizer performed no better than 1952 mean-variance theory. Platforms must justify complexity with evidence, not assumptions.

---

## Formula Reference

| Metric | Formula |
|--------|---------|
| Geometric CAGR | `(∏(1+r_t))^(12/T) − 1` |
| Ann. Volatility | `std(r) × √12` |
| Sharpe Ratio | `(mean(r) − rf_monthly) × 12 / (std(r) × √12)` |
| Sortino Ratio | `(mean(r) − rf_monthly) × 12 / (√(mean(min(r−rf,0)²)) × √12)` |
| Max Drawdown | `min((cum − cummax) / cummax)` |
| CRRA Utility | `W^(1−γ) / (1−γ)` for γ≠1; `log(W)` for γ=1 |
| ERC Objective | `min Σᵢⱼ (RCᵢ − RCⱼ)²` where `RCᵢ = wᵢ(Σw)ᵢ / σₚ` |
| BL Posterior | `[（τΣ)⁻¹ + Ω⁻¹]⁻¹ × [(τΣ)⁻¹π + Ω⁻¹Q]` |

---

## Known Limitations

1. The study replicates publicly known methodologies. Real platforms may use proprietary optimizations and constraints not captured here.
2. The 36-month rolling lookback creates a structural lag — the model reacts to crises only after they enter the window and discards them 36 months later.
3. GFC recovery cannot be measured for any rolling strategy because the warmup window ends after the March 2009 trough.
4. Black-Litterman views are fixed from State Street GMP 2025 — applying 2025 views to data starting in 2010 introduces a look-ahead bias in the views themselves.
5. The 60/40 benchmark uses VEU (international equity), not a US equity proxy. VEU significantly underperformed US equity over 2010–2025, which disadvantages the benchmark relative to a US-centric 60/40.

---

## References

- Cremers, M., Kritzman, M., & Page, S. (2005). Optimal Hedge Fund Allocations. *Journal of Portfolio Management.*
- He, G., & Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. *Goldman Sachs Investment Management.*
- Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management.*
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio Management.*
- Mehra, R., & Prescott, E. (1985). The Equity Premium: A Puzzle. *Journal of Monetary Economics.*
- Sortino, F., & van der Meer, R. (1991). Downside Risk. *Journal of Portfolio Management.*
- State Street Investment Management (2025). *Global Market Portfolio 2025: A Portfolio of Everything.*
