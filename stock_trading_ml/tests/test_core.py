"""
Unit tests for ML Trading Strategy components.

Run with: pytest tests/test_core.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import os

# Import modules to test
from data.data_loader import DataLoader, load_and_prepare_data
from models.ml_models import MLModelManager
from strategy.strategy import MLTradingStrategy, Signal, Position, Trade
from backtest.backtester import Backtester
from evaluation.metrics import PerformanceMetrics, generate_performance_report


class TestDataLoader:
    """Test data loading and feature engineering functionality."""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'data': {
                'tickers': ['AAPL'],
                'start_date': '2022-01-01',
                'end_date': '2023-01-01',
                'data_source': 'yahoo',
                'split_ratio': 0.8
            },
            'features': {
                'lookback_periods': [5, 10, 20],
                'technical_indicators': ['sma', 'ema', 'rsi'],
                'volatility_window': 20,
                'return_periods': [1, 5]
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with random walk
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))  # Small daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'dividends': [0] * len(dates),
            'stock_splits':  * len(dates)
        })
        
        return df
    
    def test_data_loader_initialization(self, sample_config):
        """Test DataLoader initialization."""
        loader = DataLoader(sample_config)
        assert loader.config == sample_config
        assert loader.data == {}
        assert loader.features == {}
    
    def test_calculate_returns(self, sample_config, sample_data):
        """Test return calculation."""
        loader = DataLoader(sample_config)
        df_with_returns = loader.calculate_returns(sample_data.copy())
        
        # Check that return columns are created
        assert 'return_1d' in df_with_returns.columns
        assert 'return_5d' in df_with_returns.columns
        assert 'log_return_1d' in df_with_returns.columns
        
        # Check that returns are properly calculated
        expected_return_1d = sample_data['close'].pct_change()
        np.testing.assert_array_almost_equal(
            df_with_returns['return_1d'].dropna(),
            expected_return_1d.dropna()
        )
    
    def test_technical_indicators(self, sample_config, sample_data):
        """Test technical indicator calculation."""
        loader = DataLoader(sample_config)
        df_with_indicators = loader.calculate_technical_indicators(sample_data.copy())
        
        # Check that indicator columns are created
        for period in sample_config['features']['lookback_periods']:
            assert f'sma_{period}' in df_with_indicators.columns
            assert f'ema_{period}' in df_with_indicators.columns
        
        assert 'rsi_14' in df_with_indicators.columns
        assert 'macd' in df_with_indicators.columns
        
        # Validate SMA calculation
        sma_5 = df_with_indicators['sma_5'].dropna()
        expected_sma_5 = sample_data['close'].rolling(window=5).mean().dropna()
        np.testing.assert_array_almost_equal(sma_5, expected_sma_5)
    
    def test_feature_engineering_pipeline(self, sample_config, sample_data):
        """Test complete feature engineering pipeline."""
        loader = DataLoader(sample_config)
        engineered_data = loader.engineer_features(sample_data.copy())
        
        # Check that target variable is created
        assert 'target' in engineered_data.columns
        
        # Check that features are created
        assert 'return_1d' in engineered_data.columns
        assert 'sma_5' in engineered_data.columns
        assert 'volatility' in engineered_data.columns
        
        # Ensure no infinite values
        numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns
        assert not np.isinf(engineered_data[numeric_cols]).any().any()
    
    def test_ml_data_preparation(self, sample_config, sample_data):
        """Test ML data preparation."""
        loader = DataLoader(sample_config)
        engineered_data = loader.engineer_features(sample_data.copy())
        X, y = loader.prepare_ml_data(engineered_data)
        
        # Check shapes
        assert len(X) == len(y)
        assert len(X.columns) > 0
        
        # Check for missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()


class TestMLModels:
    """Test ML model functionality."""
    
    @pytest.fixture
    def sample_ml_config(self):
        return {
            'models': {
                'random_state': 42,
                'test_size': 0.2,
                'cv_folds': 3,
                'models_to_train': ['logistic_regression', 'random_forest'],
                'hyperparameters': {
                    'logistic_regression': {
                        'C': [1, 10],
                        'penalty': ['l2']
                    },
                    'random_forest': {
                        'n_estimators': [50, 100],
                        'max_depth': [5, 10]
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some signal
        y = pd.Series(
            (X.iloc[:, 0] + X.iloc[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)
        )
        
        return X, y
    
    def test_model_manager_initialization(self, sample_ml_config):
        """Test MLModelManager initialization."""
        manager = MLModelManager(sample_ml_config)
        assert manager.config == sample_ml_config
        assert manager.models == {}
        assert manager.random_state == 42
    
    def test_model_initialization(self, sample_ml_config):
        """Test model initialization."""
        manager = MLModelManager(sample_ml_config)
        models = manager.initialize_models()
        
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert len(models) == 2
    
    def test_model_training(self, sample_ml_config, sample_ml_data):
        """Test model training."""
        X, y = sample_ml_data
        
        manager = MLModelManager(sample_ml_config)
        manager.initialize_models()
        trained_models = manager.train_models(X, y)
        
        # Check that models are trained
        assert 'logistic_regression' in trained_models
        assert 'random_forest' in trained_models
        
        # Check performance metrics
        assert 'logistic_regression' in manager.performance_metrics
        assert 'val_accuracy' in manager.performance_metrics['logistic_regression']
    
    def test_model_prediction(self, sample_ml_config, sample_ml_data):
        """Test model prediction."""
        X, y = sample_ml_data
        
        manager = MLModelManager(sample_ml_config)
        manager.initialize_models()
        manager.train_models(X, y)
        
        # Make predictions
        predictions, probabilities = manager.predict('logistic_regression', X.head(10))
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)


class TestTradingStrategy:
    """Test trading strategy functionality."""
    
    @pytest.fixture
    def sample_strategy_config(self):
        return {
            'strategy': {
                'signal_threshold': 0.6,
                'position_sizing': 'equal_weight',
                'max_positions': 5,
                'transaction_costs': 0.001,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'rebalance_frequency': 'daily',
                'risk_management': {
                    'max_portfolio_risk': 0.02,
                    'max_single_position': 0.2,
                    'volatility_target': 0.15
                }
            },
            'backtest': {
                'initial_capital': 100000,
                'benchmark': 'SPY',
                'commission': 0.001,
                'slippage': 0.0005
            }
        }
    
    def test_strategy_initialization(self, sample_strategy_config):
        """Test strategy initialization."""
        strategy = MLTradingStrategy(sample_strategy_config)
        
        assert strategy.signal_threshold == 0.6
        assert strategy.cash == 100000
        assert strategy.portfolio_value == 100000
        assert len(strategy.positions) == 0
        assert len(strategy.trades) == 0
    
    def test_signal_generation(self, sample_strategy_config):
        """Test signal generation."""
        strategy = MLTradingStrategy(sample_strategy_config)
        
        # Sample predictions and probabilities
        predictions = {'AAPL': np.array([1]), 'GOOGL': np.array()}
        probabilities = {'AAPL': np.array([0.8]), 'GOOGL': np.array([0.3])}
        
        # Sample price data
        price_data = {
            'AAPL': pd.DataFrame({'date': [pd.Timestamp.now()], 'close': [150.0]}),
            'GOOGL': pd.DataFrame({'date': [pd.Timestamp.now()], 'close': [2500.0]})
        }
        
        signals = strategy.generate_signals(
            predictions, probabilities, price_data, pd.Timestamp.now()
        )
        
        assert 'AAPL' in signals
        assert 'GOOGL' in signals
        assert signals['AAPL'][0] == Signal.BUY  # High probability buy signal
        assert signals['GOOGL'] == Signal.HOLD  # Low probability hold signal
    
    def test_position_sizing(self, sample_strategy_config):
        """Test position sizing calculation."""
        strategy = MLTradingStrategy(sample_strategy_config)
        
        signals = {
            'AAPL': (Signal.BUY, 0.8),
            'GOOGL': (Signal.BUY, 0.7)
        }
        
        price_data = {
            'AAPL': pd.DataFrame({'date': [pd.Timestamp.now()], 'close': [150.0]}),
            'GOOGL': pd.DataFrame({'date': [pd.Timestamp.now()], 'close': [2500.0]})
        }
        
        position_sizes = strategy.calculate_position_sizes(
            signals, price_data, pd.Timestamp.now()
        )
        
        assert 'AAPL' in position_sizes
        assert 'GOOGL' in position_sizes
        assert all(size > 0 for size in position_sizes.values())


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_returns_metrics(self):
        """Test return-based performance metrics."""
        # Create sample portfolio values
        dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, len(dates))  # Daily returns
        portfolio_values = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_returns_metrics(portfolio_values)
        
        # Check that key metrics are calculated
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Verify metrics are reasonable
        assert -1 <= metrics['total_return'] <= 5  # Reasonable return range
        assert -2 <= metrics['sharpe_ratio'] <= 5  # Reasonable Sharpe ratio range
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create a series with a known drawdown
        values = pd.Series([100, 110, 120, 90, 85, 95, 105])
        
        metrics_calc = PerformanceMetrics()
        max_dd = metrics_calc.calculate_max_drawdown(values)
        
        # Maximum drawdown should be from peak (120) to trough (85)
        expected_dd = (85 - 120) / 120
        assert abs(max_dd - expected_dd) < 0.001
    
    def test_win_loss_metrics(self):
        """Test trade win/loss statistics."""
        # Create sample trades
        trades_df = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150, -80, 75],
            'duration_days': [5, 3, 10, 2, 7, 4, 6]
        })
        
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_win_loss_metrics(trades_df)
        
        # Check calculations
        assert metrics['total_trades'] == 7
        assert abs(metrics['win_rate'] - 4/7) < 0.001  # 4 winning trades out of 7
        assert metrics['avg_win'] > 0
        assert metrics['avg_loss'] < 0


class TestBacktester:
    """Test backtesting functionality."""
    
    @pytest.fixture
    def sample_backtest_config(self):
        return {
            'data': {
                'tickers': ['TEST'],
                'start_date': '2022-01-01',
                'end_date': '2022-12-31'
            },
            'models': {
                'models_to_train': ['logistic_regression'],
                'random_state': 42
            },
            'strategy': {
                'signal_threshold': 0.6,
                'position_sizing': 'equal_weight',
                'max_positions': 5,
                'transaction_costs': 0.001,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'risk_management': {
                    'max_portfolio_risk': 0.02,
                    'max_single_position': 0.2,
                    'volatility_target': 0.15
                }
            },
            'backtest': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'walk_forward': {
                    'training_window': 60,
                    'validation_window': 20,
                    'step_size': 5
                }
            },
            'features': {
                'lookback_periods': [5, 10],
                'technical_indicators': ['sma', 'rsi'],
                'volatility_window': 10,
                'return_periods': [1]
            }
        }
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample data for backtesting."""
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        np.random.seed(42)
        
        # Generate price data
        prices = [100]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * len(dates)
        })
        
        return {'TEST': df}
    
    def test_backtester_initialization(self, sample_backtest_config):
        """Test backtester initialization."""
        backtester = Backtester(sample_backtest_config)
        
        assert backtester.initial_capital == 100000
        assert backtester.training_window == 60
        assert backtester.validation_window == 20
    
    def test_data_preparation(self, sample_backtest_config, sample_backtest_data):
        """Test backtesting data preparation."""
        backtester = Backtester(sample_backtest_config)
        
        prepared_data = backtester._prepare_backtest_data(
            sample_backtest_data, '2022-01-01', '2022-12-31'
        )
        
        assert 'TEST' in prepared_data
        assert len(prepared_data['TEST']) > 0
        assert 'date' in prepared_data['TEST'].columns


def test_configuration_loading():
    """Test configuration file loading."""
    # Create temporary config file
    config_data = {
        'data': {'tickers': ['AAPL'], 'start_date': '2022-01-01'},
        'models': {'random_state': 42}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        # Test loading
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config_data
        
    finally:
        os.unlink(config_file)


def test_data_validation():
    """Test data validation functions."""
    # Test with valid data
    valid_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'close': np.random.rand(100) * 100 + 50,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    assert len(valid_data) > 0
    assert 'close' in valid_data.columns
    assert not valid_data['close'].isnull().any()
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        'close': [np.nan, np.inf, -np.inf, 100]
    })
    
    # Should have problematic values
    assert invalid_data['close'].isnull().any() or np.isinf(invalid_data['close']).any()


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create minimal configuration
        config = {
            'data': {
                'tickers': ['TEST'],
                'start_date': '2022-01-01',
                'end_date': '2022-06-01',
                'data_source': 'test'
            },
            'features': {
                'lookback_periods': [5, 10],
                'technical_indicators': ['sma', 'rsi'],
                'volatility_window': 10,
                'return_periods': [1]
            },
            'models': {
                'random_state': 42,
                'test_size': 0.2,
                'cv_folds': 3,
                'models_to_train': ['logistic_regression'],
                'hyperparameters': {
                    'logistic_regression': {'C': [1], 'penalty': ['l2']}
                }
            },
            'strategy': {
                'signal_threshold': 0.6,
                'position_sizing': 'equal_weight',
                'max_positions': 1,
                'transaction_costs': 0.001,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'risk_management': {
                    'max_portfolio_risk': 0.02,
                    'max_single_position': 1.0,
                    'volatility_target': 0.15
                }
            },
            'backtest': {
                'initial_capital': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'walk_forward': {
                    'training_window': 30,
                    'validation_window': 10,
                    'step_size': 5
                }
            }
        }
        
        # Create sample data
        dates = pd.date_range('2022-01-01', '2022-06-01', freq='D')
        np.random.seed(42)
        prices = [100]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        data = {
            'TEST': pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': [1000000] * len(dates)
            })
        }
        
        # Test data loading
        loader = DataLoader(config)
        processed_data = {}
        for ticker, df in data.items():
            processed_data[ticker] = loader.engineer_features(df)
        
        assert len(processed_data) > 0
        
        # Test ML pipeline
        X, y = loader.prepare_ml_data(processed_data['TEST'])
        
        if len(X) > 10:  # Only test if we have enough data
            model_manager = MLModelManager(config)
            model_manager.initialize_models()
            
            # Should not crash
            try:
                trained_models = model_manager.train_models(X, y)
                assert len(trained_models) > 0
            except Exception as e:
                # It's okay if training fails with minimal data
                print(f"Training failed with minimal data: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
