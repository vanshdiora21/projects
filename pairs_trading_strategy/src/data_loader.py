import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"[ERROR] No data fetched for {ticker} between {start} and {end}")
    else:
        print(f"[OK] {ticker} → {df.shape[0]} rows")

    return df
