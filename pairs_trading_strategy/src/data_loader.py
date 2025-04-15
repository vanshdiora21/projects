def fetch_data(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df
