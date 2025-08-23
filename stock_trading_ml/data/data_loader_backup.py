"""
Data loading and feature engineering module for ML trading strategy.

This module handles:
- Fetching stock price data from Yahoo Finance
- Data cleaning and preprocessing
- Technical indicator calculation
- Feature engineering for ML models
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles data loading, cleaning, and feature engineering for stock trading data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader with configuration parameters.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data = {}
        self.features = {}
        
    def fetch_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock price data from Yahoo Finance.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping ticker symbols to price DataFrames
        """
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df.empty:
                    print(f"Warning: No data found for {ticker}")
                    continue
                    
                # Clean column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                df['ticker'] = ticker
                df['date'] = df.index
                df = df.reset_index(drop=True)
                
                data[ticker] = df
                print(f"✓ Loaded {len(df)} records for {ticker}")
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                continue
                
        self.data = data
        return data
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Calculate returns for different periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with return columns added
        """
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
            
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        # Simple Moving Averages
        for window in self.config['features']['lookback_periods']:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], window=14)
        
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
        
        # Average True Range (ATR)
        df['atr_14'] = self._calculate_atr(df, window=14)
        
        # Stochastic RSI
        df['stoch_rsi'] = self._calculate_stochastic_rsi(df['close'], window=14)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df, window=14)
        
        # Volume indicators (if volume data available)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate volatility-based features.
        
        Args:
            df: DataFrame with price data
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility features added
        """
        # Historical volatility
        df['volatility'] = df['return_1d'].rolling(window=window).std() * np.sqrt(252)
        
        # Realized volatility
        df['realized_vol'] = df['log_return_1d'].rolling(window=window).std() * np.sqrt(252)
        
        # High-Low ratio
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Close-to-close volatility
        df['cc_vol'] = np.log(df['close'] / df['close'].shift(1)).rolling(window=window).std()
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, method: str = 'next_day_return') -> pd.DataFrame:
        """
        Create target variable for ML models.
        
        Args:
            df: DataFrame with price data
            method: Method to create target ('next_day_return', 'direction', 'multi_class')
            
        Returns:
            DataFrame with target variable added
        """
        if method == 'next_day_return':
            # Predict next day's return
            df['target'] = df['return_1d'].shift(-1)
            
        elif method == 'direction':
            # Predict direction (up/down)
            df['next_return'] = df['return_1d'].shift(-1)
            df['target'] = (df['next_return'] > 0).astype(int)
            
        elif method == 'multi_class':
            # Multi-class classification (strong_buy, buy, hold, sell, strong_sell)
            df['next_return'] = df['return_1d'].shift(-1)
            df['target'] = pd.cut(df['next_return'], 
                                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                labels=[0, 1, 2, 3, 4])  # 0=strong_sell, 4=strong_buy
            
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: DataFrame with raw OHLCV data
            
        Returns:
            DataFrame with all engineered features
        """
        # Calculate returns
        df = self.calculate_returns(df, self.config['features']['return_periods'])
        
        # Technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Volatility features
        df = self.calculate_volatility_features(df, self.config['features']['volatility_window'])
        
        # Momentum features
        df = self._calculate_momentum_features(df)
        
        # Create target variable
        df = self.create_target_variable(df, method='direction')
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for machine learning models.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (features_df, target_df)
        """
        # Remove non-feature columns
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                       'dividends', 'stock_splits', 'target', 'next_return']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare features
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, y
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def _calculate_stochastic_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Stochastic RSI."""
        rsi = self._calculate_rsi(prices, window)
        stoch_rsi = (rsi - rsi.rolling(window=window).min()) / (
            rsi.rolling(window=window).max() - rsi.rolling(window=window).min()
        )
        return stoch_rsi
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        williams_r = (highest_high - df['close']) / (highest_high - lowest_low) * -100
        return williams_r
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of change
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # Relative performance vs moving average
        df['rel_to_sma_20'] = df['close'] / df['sma_20'] - 1
        df['rel_to_ema_20'] = df['close'] / df['ema_20'] - 1
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used in the model."""
        if not self.features:
            return []
        
        # Return feature column names
        sample_ticker = list(self.features.keys())[0]
        X, _ = self.prepare_ml_data(self.features[sample_ticker])
        return list(X.columns)


def load_and_prepare_data(config: Dict) -> Tuple[Dict, Dict]:
    """
    Convenience function to load and prepare data using configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (raw_data_dict, processed_features_dict)
    """
    loader = DataLoader(config)
    
    # Fetch raw data
    raw_data = loader.fetch_data(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Engineer features for each ticker
    processed_data = {}
    for ticker, df in raw_data.items():
        processed_data[ticker] = loader.engineer_features(df.copy())
    
    loader.features = processed_data
    
    return raw_data, processed_data


if __name__ == "__main__":
    # Example usage
    import json
    
    with open('../config.json', 'r') as f:
        config = json.load(f)
    
    raw_data, processed_data = load_and_prepare_data(config)
    print(f"Loaded data for {len(raw_data)} tickers")
    
    # Display sample features
    sample_ticker = list(processed_data.keys())[0]
    sample_df = processed_data[sample_ticker]
    print(f"\nSample features for {sample_ticker}:")
    print(sample_df.columns.tolist())
    print(f"Data shape: {sample_df.shape}")
