import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_data(ticker, start_date, end_date):

    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only necessary columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()

    # Create a 2-level MultiIndex: (Column name, Ticker)
    df.columns = pd.MultiIndex.from_tuples([(col, ticker) for col in df.columns])
    return df
def compute_log_returns(price_series: pd.Series) -> pd.Series:
    """
    Computes log returns from a price series.
    """
    return np.log(price_series / price_series.shift(1)).dropna()
