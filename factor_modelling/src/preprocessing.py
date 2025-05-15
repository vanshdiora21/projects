# src/preprocessing.py

import pandas as pd
import numpy as np

def compute_log_returns(price_df):
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna()

def align_data(stock_returns, factors):
    # Ensure both have the same date index
    combined = stock_returns.join(factors, how='inner')
    return combined

if __name__ == "__main__":
    # Load raw data
    stock_prices = pd.read_csv('data/raw/stock_prices.csv', index_col=0, parse_dates=True)
    factors = pd.read_csv('data/raw/fama_french_factors.csv', index_col=0, parse_dates=True)
    
    # Compute stock returns
    stock_returns = compute_log_returns(stock_prices)
    stock_returns.to_csv('data/processed/stock_returns.csv')
    
    # Align returns and factors
    combined_data = align_data(stock_returns, factors)
    combined_data.to_csv('data/processed/combined_data.csv')
