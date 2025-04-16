def evaluate_strategy(pnl_df, risk_free_rate=0.02):
    daily_returns = pnl_df['Daily Return']
    total_return = pnl_df['Cumulative Return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(pnl_df)) - 1
    annual_volatility = daily_returns.std() * (252 ** 0.5)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    max_drawdown = pnl_df['Drawdown'].min()

    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annual_return:.2%}",
        'Annualized Volatility': f"{annual_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }
