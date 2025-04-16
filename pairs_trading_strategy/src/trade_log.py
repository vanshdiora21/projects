import pandas as pd

def log_trades(signals, prices, ticker1='AAPL', ticker2='MSFT'):
    log = []
    prev_sig = 0
    for date, sig in signals.items():
        if sig != prev_sig:
            log.append({
                'Date': date,
                'Signal': sig,
                f'{ticker1}_Price': prices.loc[date, ticker1],
                f'{ticker2}_Price': prices.loc[date, ticker2]
            })
        prev_sig = sig
    return pd.DataFrame(log)
