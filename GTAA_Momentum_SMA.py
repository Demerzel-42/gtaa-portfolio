#!/usr/bin/env python
# coding: utf-8

# # Ribilanciamento ogni 14 giorni (daily data)

# In[4]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

# --- Initial settings ---
STARTING_BALANCE = 100 # initial balance
MOMENTUM_WINDOW = 120  # approximately 6 months of trading days
REBALANCE_DAYS = 7     # rebalance every n days
SMA_WINDOW = 150       # n° of trading days for SMA

# Asset tickers for the strategy
ASSET_TICKERS = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
CASH_TICKER = "SHV"
ALL_TICKERS = ASSET_TICKERS + [CASH_TICKER]

# Download dates
data_start = "2007-01-01" # start date 
data_end = datetime.date.today().strftime('%Y-%m-%d') # Scarica fino ad oggi
# data_end = "2025-11-25" # end date editable as needed

# --- 1. Download Data with yfinance ---

print(f"Download data fot {ALL_TICKERS} from {data_start} to {data_end}...")
data = yf.download(
    ALL_TICKERS,
    start=data_start,
    end=data_end,
    auto_adjust=False, 
    progress=False
)

prices = data["Adj Close"]
if prices.empty:
    raise SystemExit("Error:  No data downloaded.")
else:
    print("Data downloaded with success.")

# --- 2. Data Preparation ---

# Calculate daily returns (R)
daily_returns = prices.pct_change()

# Calculate (1 + R) for 'cumprod' calculations
daily_returns_plus_one = 1 + daily_returns

# --- 3. Weight Calculation (Momentum Ranking) ---

# Calculate Momentum: rolling product of (1 + R) over TRADING_DAYS for the ASSETS
momentum = daily_returns_plus_one[ASSET_TICKERS].rolling(
    window=MOMENTUM_WINDOW
).apply(np.prod, raw=True)

# Calculate Ranks: ascending=True (Rank min = worst, Rank max = best)
ranks = momentum.rank(axis=1, method='min', ascending=True)

# Calculate Weights: Weight = Rank / Sum_of_Ranks
weights = ranks.apply(lambda r: r / r.sum(), axis=1)

# --- 4. Signal Calculation (SMA Timing) ---

# Calculate 200_d SMA for the 5 ASSETS (using 210 trading days)
sma_200d = prices[ASSET_TICKERS].rolling(
    window=SMA_WINDOW
).mean()

# Create "Buy" signals (True if Price > SMA)
signals = prices[ASSET_TICKERS] > sma_200d

# --- 5. Rebalancing Logic (Every 14 Days) ---

# Create a rebalancing schedule: mark every 14th day
rebalance_dates = prices.index[::REBALANCE_DAYS]

# Forward-fill weights and signals to hold them constant between rebalances
# First, filter to only rebalance dates
weights_rebalance = weights.loc[weights.index.isin(rebalance_dates)]
signals_rebalance = signals.loc[signals.index.isin(rebalance_dates)]

# Then forward-fill to apply to all days until next rebalance
weights_held = weights_rebalance.reindex(prices.index, method='ffill')
signals_held = signals_rebalance.reindex(prices.index, method='ffill')

# --- 6. Strategy Backtest (Combination of Weights and Signals) ---

# Shift signals and weights by 1 day to avoid 'lookahead bias'
shifted_signals = signals_held.shift(1)
shifted_weights = weights_held.shift(1)

# Extract returns for assets and cash
asset_returns = daily_returns[ASSET_TICKERS]
cash_returns = daily_returns[CASH_TICKER]

# Allocation logic:
# Where the (shifted) signal is True, use the asset return.
# Where it's False, use the cash return (SHV).

# Transform 'cash_returns' (Series) into an aligned DataFrame
cash_returns_df = pd.DataFrame(
    {ticker: cash_returns for ticker in ASSET_TICKERS}, 
    index=cash_returns.index
)

# Choose the return to use for each asset
chosen_returns = np.where(
    shifted_signals,    # If signal == True
    asset_returns,      # Use this return
    cash_returns_df     # Otherwise use cash return
)
chosen_returns = pd.DataFrame(chosen_returns, index=asset_returns.index, columns=asset_returns.columns)

# Calculate weighted portfolio return
# R_portfolio = Sum( Weight_i * Chosen_Return_i )
portfolio_returns = (chosen_returns * shifted_weights).sum(axis=1)

# Replace first NaN value (due to shift) with 0
portfolio_returns = portfolio_returns.fillna(0)

# Calculate (1 + R_portfolio) for cumulative calculation
portfolio_returns_plus_one = 1 + portfolio_returns

# --- 7. Strategy Performance Calculation ---

# Create DataFrame for balance, starting from first valid day
gtaa_balance = pd.DataFrame(index=portfolio_returns_plus_one.dropna().index)

# Calculate cumulative balance
gtaa_balance['GTAA_Strategy_Balance'] = STARTING_BALANCE * portfolio_returns_plus_one.cumprod()

# Calculate CAGR
n_periods = len(gtaa_balance)
n_years = n_periods / 252  # 252 trading days per year
ending_balance = gtaa_balance['GTAA_Strategy_Balance'].iloc[-1]
total_strategy_cagr = round((((ending_balance / STARTING_BALANCE) ** (1 / n_years)) - 1) * 100, 2)
total_strategy_sharpe = round(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252), 2)
total_strategy_sortino = round(portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252), 2)

# Calculate Drawdowns
gtaa_balance['Peak'] = gtaa_balance['GTAA_Strategy_Balance'].cummax()
gtaa_balance['DD'] = gtaa_balance['GTAA_Strategy_Balance'] - gtaa_balance['Peak']
gtaa_dd_pct = (gtaa_balance['DD'] / gtaa_balance['Peak']).min()
gtaa_strategy_dd = round(gtaa_dd_pct * 100, 2)


# --- 8. Benchmark Calculation (SPY and Balanced Portfolio) ---

# Benchmark data aligned to strategy period
spy_raw_returns = daily_returns['SPY'].loc[gtaa_balance.index]
asset_raw_returns = daily_returns[ASSET_TICKERS].loc[gtaa_balance.index]
asset_returns_plus_one = daily_returns_plus_one[ASSET_TICKERS].loc[gtaa_balance.index]

# --- 8a. SPY Buy and Hold ---
spy_balance = STARTING_BALANCE * (1 + spy_raw_returns).cumprod()
spy_ending_balance = spy_balance.iloc[-1]
total_spx_cagr = round((((spy_ending_balance / STARTING_BALANCE) ** (1 / n_years)) - 1) * 100, 2)

# Sharpe & Sortino for SPY (Using RAW returns)
total_spx_sharpe = round(spy_raw_returns.mean() / spy_raw_returns.std() * np.sqrt(252), 2)
total_spx_sortino = round(spy_raw_returns.mean() / spy_raw_returns[spy_raw_returns < 0].std() * np.sqrt(252), 2)

spy_peak = spy_balance.cummax()
spy_dd_pct = ((spy_balance - spy_peak) / spy_peak).min()
spx_dd = round(spy_dd_pct * 100, 2)

# --- 8b. Weighted Portfolio - Buy and Hold (No Rebalancing) ---
balanced_weights = 1/len(ASSET_TICKERS)  # Equal weights

# Use geometric mean for return calculation
weighted_portf_returns_plus_one = asset_returns_plus_one.pow(balanced_weights).prod(axis=1)
weighted_portf_balance = STARTING_BALANCE * weighted_portf_returns_plus_one.cumprod()

# Calculate raw returns for metrics (restore from plus_one)
weighted_portf_raw_returns = weighted_portf_returns_plus_one - 1

weighted_portf_ending_balance = weighted_portf_balance.iloc[-1]
weighted_portf_cagr = round((((weighted_portf_ending_balance / STARTING_BALANCE) ** (1 / n_years)) - 1) * 100, 2)

# Metrics
weighted_portf_sharpe = round(weighted_portf_raw_returns.mean() / weighted_portf_raw_returns.std() * np.sqrt(252), 2)
weighted_portf_sortino = round(weighted_portf_raw_returns.mean() / weighted_portf_raw_returns[weighted_portf_raw_returns < 0].std() * np.sqrt(252), 2)

weighted_portf_peak = weighted_portf_balance.cummax()
weighted_portf_dd_pct = ((weighted_portf_balance - weighted_portf_peak) / weighted_portf_peak).min()
weighted_portf_dd = round(weighted_portf_dd_pct * 100, 2)

# --- 8c. Weighted Portfolio - WITH Rebalancing (Every 7 days) ---
# Start with equal allocation
rebalanced_weighted_portf = pd.Series(index=asset_returns_plus_one.index, dtype=float)
rebalanced_weighted_portf.iloc[0] = STARTING_BALANCE

# Track holdings for each asset
holdings = pd.DataFrame(index=asset_returns_plus_one.index, columns=ASSET_TICKERS, dtype=float)
target_weight = 1 / len(ASSET_TICKERS)  # Equal weight for each asset

for i in range(len(asset_returns_plus_one)):
    date = asset_returns_plus_one.index[i]
    
    if i == 0:
        # Initial allocation: equal weights
        for ticker in ASSET_TICKERS:
            holdings.loc[date, ticker] = STARTING_BALANCE * target_weight
    else:
        prev_date = asset_returns_plus_one.index[i-1]
        
        # Check if it's a rebalance day
        is_rebalance_day = date in rebalance_dates
        
        if is_rebalance_day:
            # Rebalance: set each asset back to initial target weight based on TOTAL Portfolio Value
            total_balance = rebalanced_weighted_portf.iloc[i-1]
            for ticker in ASSET_TICKERS:
                holdings.loc[date, ticker] = total_balance * target_weight
        else:
            # Not a rebalance day: holdings grow/shrink with daily returns
            for ticker in ASSET_TICKERS:
                holdings.loc[date, ticker] = holdings.loc[prev_date, ticker] * asset_returns_plus_one.loc[date, ticker]
    
    # Calculate total balance for the day
    rebalanced_weighted_portf.iloc[i] = holdings.loc[date, :].sum()

# Calculate metrics for rebalanced portfolio
rebalanced_weighted_portf_ending_balance = rebalanced_weighted_portf.iloc[-1]
rebalanced_weighted_portf_cagr = round((((rebalanced_weighted_portf_ending_balance / STARTING_BALANCE) ** (1 / n_years)) - 1) * 100, 2)

rebal_daily_returns = rebalanced_weighted_portf.pct_change().fillna(0)

rebalanced_weighted_portf_sharpe = round(rebal_daily_returns.mean() / rebal_daily_returns.std() * np.sqrt(252), 2)
rebalanced_weighted_portf_sortino = round(rebal_daily_returns.mean() / rebal_daily_returns[rebal_daily_returns < 0].std() * np.sqrt(252), 2)

rebalanced_weighted_portf_peak = rebalanced_weighted_portf.cummax()
rebalanced_weighted_portf_dd_pct = ((rebalanced_weighted_portf - rebalanced_weighted_portf_peak) / rebalanced_weighted_portf_peak).min()
rebalanced_weighted_portf_dd = round(rebalanced_weighted_portf_dd_pct * 100, 2)

# --- 9. Comparison Table and Chart ---

# Create comparison table
comparison_table = pd.DataFrame({
    'Strategy': ['GTAA Strategy (7-day Rebalance)', 
                 'SPY Buy and Hold', 
                 'Balanced Portfolio (No Rebalance)',
                 'Balanced Portfolio (7-day Rebalance)'],
    'CAGR (%)': [total_strategy_cagr, total_spx_cagr, weighted_portf_cagr, rebalanced_weighted_portf_cagr],
    'Max Drawdown (%)': [gtaa_strategy_dd, spx_dd, weighted_portf_dd, rebalanced_weighted_portf_dd],
    'Sharpe Ratio': [total_strategy_sharpe, total_spx_sharpe, weighted_portf_sharpe, rebalanced_weighted_portf_sharpe],
    'Sortino Ratio': [total_strategy_sortino, total_spx_sortino, weighted_portf_sortino, rebalanced_weighted_portf_sortino]
})
print('\nCOMPARISON TABLE:')
print(comparison_table)
print(f'\nNumber of rebalances: {len(rebalance_dates)}')

# Create chart
plt.figure(figsize=(10, 5))
plt.plot(gtaa_balance.index, gtaa_balance['GTAA_Strategy_Balance'], label='GTAA Strategy (7-day Rebalance)', color='blue')
plt.plot(spy_balance.index, spy_balance, label='SPY Buy and Hold', color='orange')
plt.plot(weighted_portf_balance.index, weighted_portf_balance, label='Equal Weight Portfolio (No Rebalance)', color='green')
plt.plot(rebalanced_weighted_portf.index, rebalanced_weighted_portf, label='Equal Weight Portfolio (7-day Rebalance)', color='red')

plt.title('Portfolio Growth (Logarithmic Scale)')
plt.yscale('log')  # Logarithmic scale for better comparison
plt.xlabel('Date')
plt.ylabel('Balance (Log)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()


# In[2]:


# --- 10. Detailed Output for a Specific Date ---

# GTAA weights and cash allocation at a specific date
specific_date = '2026-01-16'
if specific_date in weights_held.index:
    
    # Calculate cash allocation
    signals_on_date = shifted_signals.loc[specific_date]
    weights_on_date = shifted_weights.loc[specific_date]
    cash_allocation = (1 - weights_on_date[signals_on_date].sum()) * 100

    # Plot only cash and assets with signal True
    allocation = weights_on_date.copy()
    allocation_to_plot = allocation[signals_on_date].copy() * 100
    if cash_allocation > 0:
        allocation_to_plot['CASH (SHV)'] = cash_allocation
    allocation_to_plot.plot(kind='bar', title=f'Asset Allocation on {specific_date}', ylabel='Weight %', xlabel='Assets')
    plt.show()
    
    # Detailed breakdown
    print(f"\nDetailed Breakdown on {specific_date}:")
    for ticker in ASSET_TICKERS:
        signal = signals_on_date[ticker]
        weight = weights_on_date[ticker]
        status = "-> ASSET" if signal else "-> CASH"
        print(f"{ticker}: Weight={weight:.4f} ({weight*100:.2f}%) - Signal={signal} {status}")
    print(f"Cash Allocation = {cash_allocation:.2f}%")
else:
    print(f"\nDate {specific_date} not in the data.")