"""
Generate sample stock data for testing the ML trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


def generate_realistic_stock_data(ticker: str, start_date: str, end_date: str, 
                                 initial_price: float = 150.0) -> pd.DataFrame:
    """Generate realistic stock price data."""
    
    # Create date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Filter to business days only
    dates = dates[dates.dayofweek < 5]  # Monday=0, Sunday=6
    
    n_days = len(dates)
    
    # Set random seed for reproducible data
    np.random.seed(42)
    
    # Generate returns using random walk with drift
    daily_returns = np.random.normal(0.0005, 0.016, n_days)  # ~0.13% daily return, 1.6% volatility
    
    # Add some trend and seasonality
    trend = np.linspace(0, 0.3, n_days)  # 30% growth over the period
    seasonality = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # Annual cycle
    
    daily_returns += trend / n_days + seasonality / n_days
    
    # Calculate prices
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    opens = []
    highs = []
    lows = []
    closes = prices
    volumes = []
    
    for i, close in enumerate(closes):
        # Open is previous close + small gap
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.002) * closes[i-1]  # Small overnight gap
            open_price = closes[i-1] + gap
        
        # High and low based on intraday volatility
        intraday_vol = abs(np.random.normal(0, 0.01))
        high = max(open_price, close) * (1 + intraday_vol)
        low = min(open_price, close) * (1 - intraday_vol)
        
        # Volume (higher on big moves)
        base_volume = 50_000_000
        volume_multiplier = 1 + abs(daily_returns[i]) * 10
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'dividends': [0.0] * n_days,  # No dividends for simplicity
        'stock_splits': [0.0] * n_days,  # No splits
        'ticker': [ticker] * n_days
    })
    
    return df


def save_sample_data():
    """Generate and save sample data for all tickers in config."""
    
    # Load test config
    with open('test_config.json', 'r') as f:
        config = json.load(f)
    
    tickers = config['data']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"Generating sample data for {tickers} from {start_date} to {end_date}")
    
    # Different starting prices for variety
    starting_prices = {
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 250.0,
        'AMZN': 3200.0,
        'TSLA': 200.0
    }
    
    sample_data = {}
    
    for ticker in tickers:
        initial_price = starting_prices.get(ticker, 100.0)
        df = generate_realistic_stock_data(ticker, start_date, end_date, initial_price)
        sample_data[ticker] = df
        print(f"✓ Generated {len(df)} records for {ticker}")
    
    return sample_data


if __name__ == "__main__":
    # Generate sample data
    data = save_sample_data()
    
    # Save to pickle for easy loading
    import pickle
    with open('sample_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("\n✅ Sample data generated and saved to 'sample_data.pkl'")
    print("You can now run the trading system with sample data!")
