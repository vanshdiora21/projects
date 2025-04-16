
import os
import sys
sys.path.append(os.path.abspath('src'))

import pandas as pd
import numpy as np
from itertools import product

from src.data_loader import fetch_data
from src.strategy import get_hedge_ratio, compute_spread, generate_zscore_signals
from src.backtester import compute_returns, backtest_pairs, compute_pnl_metrics
from src.evaluator import evaluate_strategy

def optimize_zscore_thresholds(stock1='AAPL', stock2='MSFT', start_date="2020-01-01", end_date="2024-01-01"):
    df1 = fetch_data(stock1, start_date, end_date)
    df2 = fetch_data(stock2, start_date, end_date)

    combined = pd.concat([df1['Close'], df2['Close']], axis=1).dropna()
    combined.columns = [stock1, stock2]

    hedge_ratio = get_hedge_ratio(combined[stock1], combined[stock2])
    spread = compute_spread(combined[stock1], combined[stock2], hedge_ratio)
    returns = compute_returns(combined)

    entry_z_values = [0.5, 1.0, 1.5, 2.0]
    exit_z_values = [0.0, 0.25, 0.5, 1.0]

    results = []

    for entry_z, exit_z in product(entry_z_values, exit_z_values):
        try:
            signals, _ = generate_zscore_signals(spread, entry_z, exit_z)
            strat_returns = signals.shift(1) * (returns[stock1] - hedge_ratio * returns[stock2])
            pnl_df = compute_pnl_metrics(strat_returns)
            metrics = evaluate_strategy(pnl_df)
            results.append({
                'entry_z': entry_z,
                'exit_z': exit_z,
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Total Return': metrics['Total Return']
            })
        except Exception as e:
            print(f"Failed for entry_z={entry_z}, exit_z={exit_z}: {e}")

    result_df = pd.DataFrame(results)
    best = result_df.sort_values('Sharpe Ratio', ascending=False).head(5)
    print("\nTop 5 parameter combinations by Sharpe Ratio:")
    print(best)

    result_df.to_csv('outputs/zscore_optimization_results.csv', index=False)
    return result_df

if __name__ == "__main__":
    optimize_zscore_thresholds()
