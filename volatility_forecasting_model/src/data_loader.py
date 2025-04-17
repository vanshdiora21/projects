
import pandas as pd
import yfinance as yf
import numpy as np

def fetch_price_data(ticker, start_date, end_date):
    """
    Downloads historical price data for a given ticker.

    Parameters:
    ticker (str): Stock symbol
    start_date (str): Start date in 'YYYY-MM-DD'
    end_date (str): End date in 'YYYY-MM-DD'

    Returns:
    pd.DataFrame: Daily OHLCV data
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df

def compute_log_returns(price_series):
    """
    Computes daily log returns from price series.

    Parameters:
    price_series (pd.Series): Series of prices

    Returns:
    pd.Series: Log returns
    """
    return np.log(price_series / price_series.shift(1)).dropna()

def compute_realized_volatility(returns, window=20):
    """
    Computes rolling realized volatility (standard deviation).

    Parameters:
    returns (pd.Series): Daily log returns
    window (int): Rolling window size

    Returns:
    pd.Series: Realized volatility
    """
    return returns.rolling(window).std() * (252 ** 0.5)  # annualized vol
