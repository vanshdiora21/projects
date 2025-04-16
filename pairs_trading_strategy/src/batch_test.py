import os
import sys

# Add src directory to Python path
sys.path.append(os.path.abspath('src'))



from src.data_loader import fetch_data
from src.strategy import get_hedge_ratio, compute_spread, generate_zscore_signals
from src.backtester import compute_returns, backtest_pairs, compute_pnl_metrics
from src.evaluator import evaluate_strategy
import pandas as pd

pairs = [('AAPL', 'MSFT'), ('GOOG', 'META'), ('JPM', 'BAC'), ('XOM', 'CVX')]
start_date = "2020-01-01"
end_date = "2024-01-01"

results = []

for stock1, stock2 in pairs:
    print(f"Processing: {stock1} / {stock2}")
    df1 = fetch_data(stock1, start_date, end_date)
    df2 = fetch_data(stock2, start_date, end_date)
    combined = pd.concat([df1['Close'], df2['Close']], axis=1).dropna()
    combined.columns = [stock1, stock2]

    hedge_ratio = get_hedge_ratio(combined[stock1], combined[stock2])
    spread = compute_spread(combined[stock1], combined[stock2], hedge_ratio)
    signals, _ = generate_zscore_signals(spread)

    returns = compute_returns(combined)
    strat_returns = signals.shift(1) * (returns[stock1] - hedge_ratio * returns[stock2])
    pnl_df = compute_pnl_metrics(strat_returns)
    metrics = evaluate_strategy(pnl_df)
    metrics['Pair'] = f"{stock1}/{stock2}"
    results.append(metrics)

def run_batch_test():
    return pd.DataFrame(results).set_index("Pair")

if __name__ == "__main__":
    summary_df = run_batch_test()
    print(summary_df)
