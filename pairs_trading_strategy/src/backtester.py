import pandas as pd

def compute_returns(price_df):
    return price_df.pct_change().dropna()

def backtest_pairs(signals, returns, hedge_ratio):
    aligned_signals = signals.shift(1).loc[returns.index]
    strat_returns = aligned_signals * (returns['AAPL'] - hedge_ratio * returns['MSFT'])
    return strat_returns

def compute_pnl_metrics(strat_returns):
    pnl = pd.DataFrame(index=strat_returns.index)
    pnl['Daily Return'] = strat_returns
    pnl['Cumulative Return'] = (1 + strat_returns).cumprod()
    pnl['Cumulative Max'] = pnl['Cumulative Return'].cummax()
    pnl['Drawdown'] = pnl['Cumulative Return'] / pnl['Cumulative Max'] - 1
    return pnl

def apply_trading_costs(signals, strat_returns, cost_per_trade=0.001):
    trades = signals.diff().abs()
    costs = trades * cost_per_trade
    return strat_returns - costs.fillna(0)
