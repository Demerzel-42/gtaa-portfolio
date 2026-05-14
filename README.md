# GTAA — Momentum & SMA Strategy

A backtest of a **Global Tactical Asset Allocation (GTAA)** strategy that combines a **momentum ranking** with a **trend-following SMA filter**, benchmarked against the S&P 500 and an equal-weight multi-asset portfolio.

Built as part of a university project in quantitative finance.

## Strategy Overview

The strategy allocates capital across five asset classes, weighting each by its recent relative performance and switching to cash whenever an asset trades below its long-term moving average.

**Asset universe**

| Ticker | Asset Class |
|--------|-------------|
| SPY | US Equities |
| EFA | International Developed Equities |
| IEF | US Treasuries (7–10y) |
| VNQ | US Real Estate |
| DBC | Commodities |
| SHV | Cash proxy (short-term Treasuries) |

**Logic**

1. **Momentum ranking** — every asset is ranked by its 120-day cumulative return (~6 months). Weights are proportional to rank, so the strongest performer gets the largest allocation.
2. **SMA filter** — for each asset, if the price is above its 150-day simple moving average, the position is held; otherwise the allocation rotates into cash (SHV).
3. **Rebalancing** — weights and signals are refreshed every 7 trading days.
4. **No look-ahead** — signals and weights are shifted by one day before being applied to returns.

## Parameters

```python
STARTING_BALANCE = 100      # initial capital
MOMENTUM_WINDOW  = 120      # ~6 months
SMA_WINDOW       = 150      # trend filter
REBALANCE_DAYS   = 7        # rebalance frequency
```

Backtest period: **2007-01-01** to today.

## Benchmarks

The strategy is compared against:

- **SPY Buy & Hold** — passive S&P 500 exposure
- **Equal-Weight Buy & Hold** — 20% in each of the five assets, no rebalancing
- **Equal-Weight + Rebalancing** — 20% in each asset, rebalanced every 7 days

Performance metrics computed for all four: **CAGR, Sharpe ratio, Sortino ratio, Maximum Drawdown**.

## Notes & Limitations

- Backtest assumes zero transaction costs, taxes, and slippage.
- Returns use adjusted close prices (dividends reinvested).
- The strategy is for educational purposes; it is not investment advice.

## References

- Faber, M. T. (2007). *A Quantitative Approach to Tactical Asset Allocation.* The Journal of Wealth Management.
