from .data_loader import fetch_data
from .strategy import get_hedge_ratio, compute_spread, generate_zscore_signals
from .backtester import (
    compute_returns,
    backtest_pairs,
    compute_pnl_metrics,
    apply_trading_costs
)
from .evaluator import evaluate_strategy
from .trade_log import log_trades
from .plotter import (
    plot_cumulative_returns,
    plot_trade_signals
)
