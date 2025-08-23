"""
Enhanced data_loader.py with rate limiting and retry logic
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os  # <-- ADD THIS LINE
import time
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Enhanced DataLoader with rate limiting handling."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = {}
        self.features = {}
    
    def load_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Load pre-generated sample data."""
        import pickle
        
        if os.path.exists('sample_data.pkl'):
            print("Loading sample data from cache...")
            with open('sample_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.data = data
            print(f"✓ Loaded sample data for {list(data.keys())}")
            return data
        else:
            print("No sample data found. Run generate_sample_data.py first.")
            return {}
    
    def fetch_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch stock price data with fallback to sample data."""
        
        # Try to load sample data first
        if os.path.exists('sample_data.pkl'):
            print("Using sample data to avoid rate limiting...")
            return self.load_sample_data()
        
        # Original fetch logic with rate limiting...
        print(f"Fetching live data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        data = {}
        for i, ticker in enumerate(tickers):
            print(f"Fetching {ticker} ({i+1}/{len(tickers)})...")
            
            # Add delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(1)  # 1 second delay between requests
            
            try:
                # Retry logic for rate limiting
                for attempt in range(3):
                    try:
                        stock = yf.Ticker(ticker)
                        df = stock.history(start=start_date, end=end_date)
                        
                        if df.empty:
                            print(f"Warning: No data found for {ticker}")
                            break
                            
                        # Clean column names
                        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                        df['ticker'] = ticker
                        df['date'] = df.index
                        df = df.reset_index(drop=True)
                        
                        data[ticker] = df
                        print(f"✓ Loaded {len(df)} records for {ticker}")
                        break  # Success, break retry loop
                        
                    except Exception as e:
                        if "rate limited" in str(e).lower() or "too many requests" in str(e).lower():
                            print(f"Rate limited for {ticker}, waiting {2 ** attempt} seconds...")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            print(f"Error fetching data for {ticker}: {str(e)}")
                            break
                            
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                continue
                
        self.data = data
        return data
    
    # ... rest of your methods stay the same ...
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Calculate returns for different periods."""
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators without TA-Lib."""
        # Simple Moving Averages
        for window in self.config['features']['lookback_periods']:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI (simple implementation)
        df['rsi_14'] = self._calculate_rsi_simple(df['close'], window=14)
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], window=20)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        
        return df
    
    def _calculate_rsi_simple(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI without TA-Lib."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_volatility_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volatility-based features."""
        df['volatility'] = df['return_1d'].rolling(window=window).std() * np.sqrt(252)
        df['realized_vol'] = df['log_return_1d'].rolling(window=window).std() * np.sqrt(252)
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        return df
    
    def create_target_variable(self, df: pd.DataFrame, method: str = 'direction') -> pd.DataFrame:
        """Create target variable for ML models."""
        df['next_return'] = df['return_1d'].shift(-1)
        df['target'] = (df['next_return'] > 0).astype(int)
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        df = self.calculate_returns(df, self.config['features']['return_periods'])
        df = self.calculate_technical_indicators(df)
        df = self.calculate_volatility_features(df, self.config['features']['volatility_window'])
        df = self.create_target_variable(df, method='direction')
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for machine learning models."""
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                       'dividends', 'stock_splits', 'target', 'next_return']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        X = X.fillna(method='ffill').fillna(method='bfill')
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, y


def load_and_prepare_data(config: Dict) -> Tuple[Dict, Dict]:
    """Load and prepare data using configuration."""
    loader = DataLoader(config)
    
    raw_data = loader.fetch_data(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    processed_data = {}
    for ticker, df in raw_data.items():
        processed_data[ticker] = loader.engineer_features(df.copy())
    
    loader.features = processed_data
    return raw_data, processed_data
