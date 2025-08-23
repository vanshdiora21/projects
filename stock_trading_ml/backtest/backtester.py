"""
Backtesting engine for ML-driven trading strategy.

This module implements:
- Historical backtesting with walk-forward validation
- Performance tracking and metrics calculation
- Trade execution simulation
- Portfolio rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from strategy.strategy import MLTradingStrategy, Trade, Signal
from models.ml_models import MLModelManager
from data.data_loader import DataLoader


class Backtester:
    """
    Backtesting engine for ML trading strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config['backtest']
        
        # Initialize components
        self.strategy = MLTradingStrategy(config)
        self.model_manager = None
        self.data_loader = DataLoader(config)
        
        # Backtesting parameters
        self.initial_capital = self.backtest_config['initial_capital']
        self.commission = self.backtest_config['commission']
        self.slippage = self.backtest_config['slippage']
        
        # Walk-forward parameters
        self.training_window = self.backtest_config['walk_forward']['training_window']
        self.validation_window = self.backtest_config['walk_forward']['validation_window']
        self.step_size = self.backtest_config['walk_forward']['step_size']
        
        # Results storage
        self.results = {}
        self.trades = []
        self.portfolio_history = []
        self.model_performance = {}
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run complete backtesting process.
        
        Args:
            data: Dictionary mapping tickers to price DataFrames
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary containing backtest results
        """
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Using {len(data)} tickers: {list(data.keys())}")
        
        # Prepare data
        prepared_data = self._prepare_backtest_data(data, start_date, end_date)
        
        # Initialize model manager
        self.model_manager = MLModelManager(self.config)
        self.model_manager.initialize_models()
        
        # Get date range for backtesting
        date_range = self._get_date_range(prepared_data, start_date, end_date)
        
        if len(date_range) < self.training_window + self.validation_window:
            raise ValueError("Insufficient data for backtesting with current window sizes")
        
        # Run walk-forward backtesting
        results = self._walk_forward_backtest(prepared_data, date_range)
        
        # Calculate final performance metrics
        final_metrics = self._calculate_final_metrics()
        
        # Store results
        self.results = {
            'performance_metrics': final_metrics,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'model_performance': self.model_performance,
            'config': self.config
        }
        
        return self.results
    
    def _prepare_backtest_data(self, data: Dict[str, pd.DataFrame], 
                              start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for backtesting.
        
        Args:
            data: Raw price data
            start_date: Start date
            end_date: End date
            
        Returns:
            Prepared data with features
        """
        prepared_data = {}
        
        for ticker, df in data.items():
            # Ensure date column
            if 'date' not in df.columns:
                if df.index.name == 'Date' or 'date' in str(df.index.dtype):
                    df = df.reset_index()
                    df.columns = [col.lower() if col != 'Date' else 'date' for col in df.columns]
                else:
                    df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df[mask].copy()
            
            if len(df_filtered) < 50:  # Minimum data requirement
                print(f"Warning: Insufficient data for {ticker}, skipping")
                continue
            
            # Engineer features
            df_prepared = self.data_loader.engineer_features(df_filtered)
            prepared_data[ticker] = df_prepared
            
            print(f"Prepared {len(df_prepared)} records for {ticker}")
        
        return prepared_data
    
    def _get_date_range(self, data: Dict[str, pd.DataFrame], 
                       start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Get common date range across all tickers.
        
        Args:
            data: Prepared data
            start_date: Start date
            end_date: End date
            
        Returns:
            Common date range
        """
        # Find common date range
        all_dates = []
        for ticker, df in data.items():
            all_dates.extend(df['date'].tolist())
        
        unique_dates = sorted(list(set(all_dates)))
        
        # Filter to requested range
        date_range = [d for d in unique_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date]
        
        return pd.DatetimeIndex(date_range)
    
    def _walk_forward_backtest(self, data: Dict[str, pd.DataFrame], 
                              date_range: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Perform walk-forward backtesting.
        
        Args:
            data: Prepared data
            date_range: Date range for backtesting
            
        Returns:
            Backtesting results
        """
        print(f"Starting walk-forward backtest over {len(date_range)} days")
        
        # Initialize tracking variables
        current_step = 0
        total_steps = (len(date_range) - self.training_window - self.validation_window) // self.step_size
        
        for i in range(self.training_window, len(date_range) - self.validation_window, self.step_size):
            current_step += 1
            print(f"\nStep {current_step}/{total_steps}")
            
            # Define training and validation periods
            train_start_idx = max(0, i - self.training_window)
            train_end_idx = i
            val_start_idx = i
            val_end_idx = min(len(date_range), i + self.validation_window)
            
            train_start_date = date_range[train_start_idx]
            train_end_date = date_range[train_end_idx - 1]
            val_start_date = date_range[val_start_idx]
            val_end_date = date_range[val_end_idx - 1]
            
            print(f"Training: {train_start_date.date()} to {train_end_date.date()}")
            print(f"Validation: {val_start_date.date()} to {val_end_date.date()}")
            
            # Train models
            model_performance = self._train_models_for_period(
                data, train_start_date, train_end_date
            )
            
            # Run strategy for validation period
            self._run_strategy_for_period(
                data, val_start_date, val_end_date, model_performance
            )
        
        return {'status': 'completed'}
    
    def _train_models_for_period(self, data: Dict[str, pd.DataFrame], 
                                start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Train ML models for a specific time period.
        
        Args:
            data: Prepared data
            start_date: Training start date
            end_date: Training end date
            
        Returns:
            Model performance metrics
        """
        # Combine data from all tickers for training
        combined_features = []
        combined_targets = []
        
        for ticker, df in data.items():
            # Filter to training period
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            period_data = df[mask].copy()
            
            if len(period_data) < 20:  # Minimum training data
                continue
            
            # Prepare ML data
            X, y = self.data_loader.prepare_ml_data(period_data)
            
            # Remove rows with NaN targets
            valid_mask = ~y.isna() & ~X.isna().any(axis=1)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) > 10:  # Minimum samples
                combined_features.append(X_clean)
                combined_targets.append(y_clean)
        
        if not combined_features:
            print("Warning: No valid training data found")
            return {}
        
        # Combine all data
        X_train = pd.concat(combined_features, ignore_index=True)
        y_train = pd.concat(combined_targets, ignore_index=True)
        
        print(f"Training with {len(X_train)} samples, {len(X_train.columns)} features")
        
        # Train models
        trained_models = self.model_manager.train_models(X_train, y_train)
        
        # Store performance metrics
        performance = self.model_manager.performance_metrics
        
        return performance
    
    def _run_strategy_for_period(self, data: Dict[str, pd.DataFrame],
                                start_date: pd.Timestamp, end_date: pd.Timestamp,
                                model_performance: Dict[str, Any]) -> None:
        """
        Run trading strategy for a specific period.
        
        Args:
            data: Prepared data
            start_date: Strategy start date
            end_date: Strategy end date
            model_performance: Model performance metrics from training
        """
        # Get trading dates
        trading_dates = []
        for ticker, df in data.items():
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            period_dates = df[mask]['date'].tolist()
            trading_dates.extend(period_dates)
        
        trading_dates = sorted(list(set(trading_dates)))
        
        # Select best performing model
        best_model = self._select_best_model(model_performance)
        print(f"Using model: {best_model}")
        
        # Run strategy day by day
        for current_date in trading_dates:
            self._run_single_day(data, current_date, best_model)
    
    def _run_single_day(self, data: Dict[str, pd.DataFrame], 
                       current_date: pd.Timestamp, model_name: str) -> None:
        """
        Run strategy for a single trading day.
        
        Args:
            data: Prepared data
            current_date: Current trading date
            model_name: Name of model to use for predictions
        """
        # Get predictions for all tickers
        predictions = {}
        probabilities = {}
        prices = {}
        
        for ticker, df in data.items():
            # Get data up to current date
            mask = df['date'] <= current_date
            historical_data = df[mask]
            
            if len(historical_data) < 2:
                continue
            
            # Prepare features for prediction
            X, y = self.data_loader.prepare_ml_data(historical_data)
            
            if len(X) == 0:
                continue
            
            # Make prediction for the last row (current date)
            try:
                pred, prob = self.model_manager.predict(model_name, X.tail(1))
                predictions[ticker] = pred
                probabilities[ticker] = prob
                prices[ticker] = historical_data
            except Exception as e:
                print(f"Prediction error for {ticker}: {e}")
                continue
        
        if not predictions:
            return
        
        # Generate trading signals
        signals = self.strategy.generate_signals(predictions, probabilities, prices, current_date)
        
        # Calculate position sizes
        position_sizes = self.strategy.calculate_position_sizes(signals, prices, current_date)
        
        # Execute trades
        executed_trades = self.strategy.execute_trades(signals, position_sizes, prices, current_date)
        
        # Add executed trades to history
        self.trades.extend(executed_trades)
        
        # Update portfolio
        self.strategy.update_portfolio(current_date, prices)
        
        # Store portfolio history
        portfolio_state = self.strategy.get_portfolio_summary()
        portfolio_state['date'] = current_date
        self.portfolio_history.append(portfolio_state)
    
    def _select_best_model(self, model_performance: Dict[str, Any]) -> str:
        """
        Select best performing model based on validation metrics.
        
        Args:
            model_performance: Model performance metrics
            
        Returns:
            Name of best model
        """
        if not model_performance:
            # Default to first available model
            available_models = self.config['models']['models_to_train']
            return available_models[0] if available_models else 'logistic_regression'
        
        # Select model with highest validation accuracy
        best_model = None
        best_score = -1
        
        for model_name, metrics in model_performance.items():
            val_accuracy = metrics.get('val_accuracy', 0)
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_model = model_name
        
        return best_model or 'logistic_regression'
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """
        Calculate final performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        returns = portfolio_df['daily_return'].dropna()
        
        # Calculate metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1] if len(portfolio_df) > 0 else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized metrics
        trading_days = len(returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_portfolio_value': final_value,
            'trading_days': trading_days
        }
        
        return metrics
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Get detailed trade analysis.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'ticker': trade.ticker,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'commission': trade.commission,
                'signal_confidence': trade.signal_confidence,
                'trade_type': trade.trade_type,
                'duration_days': (trade.exit_date - trade.entry_date).days if pd.notna(trade.exit_date) else None
            })
        
        return pd.DataFrame(trade_data)
    
    def get_portfolio_timeline(self) -> pd.DataFrame:
        """
        Get portfolio value timeline.
        
        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.portfolio_history)
    
    def export_results(self, output_dir: str = 'backtest_results/') -> None:
        """
        Export backtest results to files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export trades
        if self.trades:
            trade_df = self.get_trade_analysis()
            trade_df.to_csv(f'{output_dir}/trades_{timestamp}.csv', index=False)
        
        # Export portfolio history
        if self.portfolio_history:
            portfolio_df = self.get_portfolio_timeline()
            portfolio_df.to_csv(f'{output_dir}/portfolio_history_{timestamp}.csv', index=False)
        
        # Export performance metrics
        if self.results:
            import json
            with open(f'{output_dir}/performance_metrics_{timestamp}.json', 'w') as f:
                json.dump(self.results['performance_metrics'], f, indent=4, default=str)
        
        print(f"Results exported to {output_dir}")


def run_backtest(config: Dict, data: Dict[str, pd.DataFrame], 
                start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Convenience function to run a complete backtest.
    
    Args:
        config: Configuration dictionary
        data: Price data
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        Backtest results
    """
    backtester = Backtester(config)
    results = backtester.run_backtest(data, start_date, end_date)
    return results


if __name__ == "__main__":
    print("Backtester module loaded successfully")
    print("Features:")
    print("- Walk-forward validation")
    print("- Model retraining")
    print("- Performance tracking")
    print("- Trade analysis")
    print("- Results export")
