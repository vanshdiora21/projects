
import pandas as pd
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    print(f"[INFO] Fetching: {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"[ERROR] No data returned for {ticker} — check ticker or try different dates.")
    
    print(f"[OK] Retrieved {len(df)} rows for {ticker}")
    return df
