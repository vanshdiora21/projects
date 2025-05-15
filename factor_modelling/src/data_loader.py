import yfinance as yf
import pandas as pd
from io import StringIO
import requests

def download_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval='1mo', auto_adjust=True)
    close_prices = data['Close']
    return close_prices

def download_fama_french_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_Monthly_CSV.zip"
    momentum_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_Monthly_CSV.zip"
    
    # Download 5 factors + risk-free
    five_factors = pd.read_csv(
        url, skiprows=3, index_col=0, parse_dates=True)
    
    # Clean and format
    five_factors.index = pd.to_datetime(five_factors.index, format='%Y%m')
    five_factors = five_factors.apply(pd.to_numeric, errors='coerce')
    five_factors = five_factors.loc['2014-01-01':]  # Filter to 2014 onwards

    # Download Momentum
    momentum = pd.read_csv(momentum_url, skiprows=13, index_col=0)
    momentum.index = pd.to_datetime(momentum.index, format='%Y%m')
    momentum = momentum.apply(pd.to_numeric, errors='coerce')
    momentum = momentum.loc['2014-01-01':]

    # Merge
    factors = five_factors.join(momentum)

    return factors

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    stock_data = download_stock_data(tickers, '2014-01-01', '2024-12-31')
    stock_data.to_csv('data/raw/stock_prices.csv')

    factors = download_fama_french_factors()
    factors.to_csv('data/raw/fama_french_factors.csv')
