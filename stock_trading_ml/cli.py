"""
Command Line Interface for ML Trading Strategy.

This module provides a comprehensive CLI for running different aspects of the trading strategy.
"""

import typer
import json
import pandas as pd
from typing import Optional, List
from pathlib import Path
import os

from main import run_complete_strategy, run_complete_strategy_with_config
from data.data_loader import load_and_prepare_data
from models.ml_models import MLModelManager


app = typer.Typer(help="ML Trading Strategy CLI")


@app.command()
def backtest(
    tickers: List[str] = typer.Option(["AAPL", "GOOGL", "MSFT"], "--ticker", "-t", help="Stock tickers to trade"),
    start_date: str = typer.Option("2020-01-01", "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option("2024-01-01", "--end", "-e", help="End date (YYYY-MM-DD)"),
    model: str = typer.Option("random_forest", "--model", "-m", help="ML model to use"),
    initial_capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital"),
    output_dir: str = typer.Option("results/", "--output", "-o", help="Output directory"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Custom config file")
):
    """Run backtesting with specified parameters."""
    
    typer.echo(f"🚀 Starting backtest for {', '.join(tickers)}")
    typer.echo(f"📅 Period: {start_date} to {end_date}")
    typer.echo(f"🤖 Model: {model}")
    typer.echo(f"💰 Initial Capital: ${initial_capital:,.2f}")
    
    if config_file:
        # Use existing config file
        results = run_complete_strategy(config_file, output_dir)
    else:
        # Create config from parameters
        config = {
            'data': {
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
                'data_source': 'yahoo',
                'split_ratio': 0.8
            },
            'features': {
                'lookback_periods': [5, 10, 20, 50],
                'technical_indicators': ['sma', 'ema', 'rsi', 'macd'],
                'volatility_window': 20,
                'return_periods': [1, 5, 10]
            },
            'models': {
                'random_state': 42,
                'test_size': 0.2,
                'cv_folds': 5,
                'models_to_train': [model],
                'hyperparameters': {
                    'logistic_regression': {'C': [1, 10], 'penalty': ['l2']},
                    'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                    'xgboost': {'n_estimators': [100, 200], 'max_depth': [6, 10]}
                }
            },
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
                'initial_capital': initial_capital,
                'benchmark': 'SPY',
                'commission': 0.001,
                'slippage': 0.0005,
                'walk_forward': {
                    'training_window': 252,
                    'validation_window': 63,
                    'step_size': 21
                }
            }
        }
        
        results = run_complete_strategy_with_config(config, output_dir)
    
    typer.echo(f"✅ Backtest completed! Results saved to {output_dir}")


@app.command()
def train(
    tickers: List[str] = typer.Option(["AAPL"], "--ticker", "-t", help="Stock tickers for training"),
    start_date: str = typer.Option("2020-01-01", "--start", "-s", help="Training start date"),
    end_date: str = typer.Option("2023-01-01", "--end", "-e", help="Training end date"),
    models: List[str] = typer.Option(["random_forest", "xgboost"], "--model", "-m", 
                                   help="Models to train"),
    output_dir: str = typer.Option("trained_models/", "--output", "-o", help="Output directory")
):
    """Train ML models on historical data."""
    
    typer.echo(f"🎯 Training models: {', '.join(models)}")
    typer.echo(f"📊 Data: {', '.join(tickers)} from {start_date} to {end_date}")
    
    # Create config for training
    config = {
        'data': {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'data_source': 'yahoo',
            'split_ratio': 0.8
        },
        'features': {
            'lookback_periods': [5, 10, 20, 50],
            'technical_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
            'volatility_window': 20,
            'return_periods': [1, 5, 10]
        },
        'models': {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'models_to_train': models,
            'hyperparameters': {
                'logistic_regression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
                'random_forest': {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None]},
                'xgboost': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.1, 0.2]}
            }
        }
    }
    
    # Load and prepare data
    typer.echo("📥 Loading data...")
    raw_data, processed_data = load_and_prepare_data(config)
    
    # Combine data from all tickers
    combined_features = []
    combined_targets = []
    
    for ticker, df in processed_data.items():
        from data.data_loader import DataLoader
        loader = DataLoader(config)
        X, y = loader.prepare_ml_data(df)
        
        # Remove NaN values
        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) > 10:
            combined_features.append(X_clean)
            combined_targets.append(y_clean)
    
    if not combined_features:
        typer.echo("❌ No valid training data found")
        return
    
    # Combine all data
    X_train = pd.concat(combined_features, ignore_index=True)
    y_train = pd.concat(combined_targets, ignore_index=True)
    
    typer.echo(f"🔄 Training on {len(X_train)} samples with {len(X_train.columns)} features...")
    
    # Train models
    model_manager = MLModelManager(config)
    model_manager.initialize_models()
    trained_models = model_manager.train_models(X_train, y_train)
    
    # Save models
    os.makedirs(output_dir, exist_ok=True)
    model_manager.save_models(output_dir)
    
    # Display results
    typer.echo("\n📈 Training Results:")
    for model_name, metrics in model_manager.performance_metrics.items():
        typer.echo(f"  {model_name}:")
        typer.echo(f"    Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
        if 'roc_auc' in metrics:
            typer.echo(f"    ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    
    typer.echo(f"✅ Models saved to {output_dir}")


@app.command()
def analyze(
    results_dir: str = typer.Argument("results/", help="Directory containing backtest results"),
    show_plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate plots")
):
    """Analyze backtest results."""
    
    typer.echo(f"📊 Analyzing results from {results_dir}")
    
    # Load results
    portfolio_file = os.path.join(results_dir, "portfolio_history.csv")
    trades_file = os.path.join(results_dir, "trades.csv")
    metrics_file = os.path.join(results_dir, "performance_metrics.json")
    
    if not os.path.exists(portfolio_file):
        typer.echo(f"❌ Portfolio history file not found: {portfolio_file}")
        return
    
    # Load data
    portfolio_df = pd.read_csv(portfolio_file)
    trades_df = pd.read_csv(trades_file) if os.path.exists(trades_file) else pd.DataFrame()
    
    # Display summary
    if not portfolio_df.empty:
        initial_value = portfolio_df['total_value'].iloc[0] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value'].iloc
        final_value = portfolio_df['total_value'].iloc[-1] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        typer.echo("\n📈 PERFORMANCE SUMMARY")
        typer.echo("-" * 30)
        typer.echo(f"Initial Value: ${initial_value:,.2f}")
        typer.echo(f"Final Value: ${final_value:,.2f}")
        typer.echo(f"Total Return: {total_return:.2%}")
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0]) if 'pnl' in trades_df.columns else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            typer.echo(f"Total Trades: {total_trades}")
            typer.echo(f"Win Rate: {win_rate:.2%}")
            
            if 'pnl' in trades_df.columns:
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
                avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
                typer.echo(f"Average Win: ${avg_win:.2f}")
                typer.echo(f"Average Loss: ${avg_loss:.2f}")
    
    # Generate plots if requested
    if show_plots:
        typer.echo("\n🎨 Generating plots...")
        
        from utils.plotting import TradingVisualizer
        visualizer = TradingVisualizer()
        
        plots_dir = os.path.join(results_dir, 'analysis_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        if not portfolio_df.empty:
            visualizer.plot_portfolio_performance(
                portfolio_df, 
                save_path=os.path.join(plots_dir, 'performance_analysis.png')
            )
            
            if not trades_df.empty:
                visualizer.plot_trade_analysis(
                    trades_df,
                    save_path=os.path.join(plots_dir, 'trade_analysis.png')
                )
        
        typer.echo(f"✅ Plots saved to {plots_dir}")


@app.command()
def quick_test(
    ticker: str = typer.Option("AAPL", "--ticker", "-t", help="Stock ticker to test"),
    model: str = typer.Option("random_forest", "--model", "-m", help="ML model to use"),
    days: int = typer.Option(365, "--days", "-d", help="Number of days to test")
):
    """Run a quick test with minimal configuration."""
    
    from datetime import datetime, timedelta
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    typer.echo(f"🔬 Quick test: {ticker} with {model}")
    typer.echo(f"📅 Testing {days} days: {start_date} to {end_date}")
    
    from main import quick_test as run_quick_test
    run_quick_test(ticker, model)


@app.command()
def config(
    output_file: str = typer.Option("custom_config.json", "--output", "-o", help="Output config file")
):
    """Generate a sample configuration file."""
    
    sample_config = {
        "data": {
            "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "data_source": "yahoo",
            "split_ratio": 0.8
        },
        "features": {
            "lookback_periods": [5, 10, 20, 50],
            "technical_indicators": [
                "sma", "ema", "rsi", "macd", "bollinger_bands",
                "atr", "stoch_rsi", "williams_r"
            ],
            "volatility_window": 20,
            "return_periods": [1, 5, 10]
        },
        "models": {
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 5,
            "models_to_train": [
                "logistic_regression",
                "random_forest",
                "xgboost"
            ],
            "hyperparameters": {
                "logistic_regression": {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                },
                "random_forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "xgboost": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0]
                }
            }
        },
        "strategy": {
            "signal_threshold": 0.6,
            "position_sizing": "equal_weight",
            "max_positions": 5,
            "transaction_costs": 0.001,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "rebalance_frequency": "daily",
            "risk_management": {
                "max_portfolio_risk": 0.02,
                "max_single_position": 0.2,
                "volatility_target": 0.15
            }
        },
        "backtest": {
            "initial_capital": 100000,
            "benchmark": "SPY",
            "commission": 0.001,
            "slippage": 0.0005,
            "walk_forward": {
                "training_window": 252,
                "validation_window": 63,
                "step_size": 21
            }
        },
        "plotting": {
            "figure_size": [12, 8],
            "style": "seaborn-v0_8",
            "save_plots": True,
            "plot_directory": "plots/",
            "formats": ["png", "pdf"]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    typer.echo(f"✅ Sample configuration saved to {output_file}")
    typer.echo("📝 Edit this file to customize your strategy parameters")


if __name__ == "__main__":
    app()
