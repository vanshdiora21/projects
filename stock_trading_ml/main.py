"""
Main execution script for ML Trading Strategy.

This script provides a command-line interface to run the complete trading strategy workflow.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

from data.data_loader import load_and_prepare_data
from models.ml_models import MLModelManager
from strategy.strategy import MLTradingStrategy
from backtest.backtester import Backtester
from evaluation.metrics import generate_performance_report
from utils.plotting import TradingVisualizer


def load_config(config_path: str = 'config.json') -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_complete_strategy(config_path: str = 'config.json', 
                         output_dir: str = 'results/') -> dict:
    """
    Run the complete ML trading strategy workflow.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save results
        
    Returns:
        Dictionary with strategy results
    """
    print("=" * 60)
    print("🚀 ML TRADING STRATEGY - COMPLETE WORKFLOW")
    print("=" * 60)
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config(config_path)
    print(f"✓ Configuration loaded from {config_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    raw_data, processed_data = load_and_prepare_data(config)
    
    if not processed_data:
        raise ValueError("No data loaded. Please check your configuration.")
    
    print(f"✓ Data loaded for {len(processed_data)} tickers")
    for ticker, df in processed_data.items():
        print(f"  - {ticker}: {len(df)} records")
    
    # Step 2: Run backtesting
    print("\n🔄 Step 2: Running backtesting...")
    backtester = Backtester(config)
    
    backtest_results = backtester.run_backtest(
        data=processed_data,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    print("✓ Backtesting completed")
    
    # Step 3: Performance analysis
    print("\n📈 Step 3: Analyzing performance...")
    
    # Get portfolio history
    portfolio_df = pd.DataFrame(backtester.portfolio_history)
    trades_df = backtester.get_trade_analysis()
    
    if not portfolio_df.empty:
        portfolio_values = portfolio_df['total_value'] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value']
        performance_report = generate_performance_report(portfolio_values, trades_df)
        
        # Print summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 40)
        
        metrics = performance_report['returns_metrics']
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {abs(metrics.get('max_drawdown', 0)):.2%}")
        print(f"Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        
        if 'trade_metrics' in performance_report:
            trade_metrics = performance_report['trade_metrics']
            print(f"Total Trades: {trade_metrics.get('total_trades', 0)}")
            print(f"Win Rate: {trade_metrics.get('win_rate', 0):.2%}")
    
    # Step 4: Generate visualizations
    print("\n🎨 Step 4: Generating visualizations...")
    visualizer = TradingVisualizer()
    
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if not portfolio_df.empty:
        # Portfolio performance plot
        visualizer.plot_portfolio_performance(
            portfolio_df, 
            save_path=os.path.join(plots_dir, 'portfolio_performance.png')
        )
        
        # Drawdown analysis
        visualizer.plot_drawdown_analysis(
            portfolio_df,
            save_path=os.path.join(plots_dir, 'drawdown_analysis.png')
        )
        
        # Trade analysis
        if not trades_df.empty:
            visualizer.plot_trade_analysis(
                trades_df,
                save_path=os.path.join(plots_dir, 'trade_analysis.png')
            )
    
    print(f"✓ Visualizations saved to {plots_dir}")
    
    # Step 5: Export results
    print("\n💾 Step 5: Exporting results...")
    
    # Export trades
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        print(f"✓ Trades exported to {output_dir}/trades.csv")
    
    # Export portfolio history
    if not portfolio_df.empty:
        portfolio_df.to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        print(f"✓ Portfolio history exported to {output_dir}/portfolio_history.csv")
    
    # Export performance metrics
    if 'performance_report' in locals():
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(performance_report['returns_metrics'], f, indent=4, default=str)
        print(f"✓ Performance metrics exported to {output_dir}/performance_metrics.json")
    
    print("\n🎉 STRATEGY EXECUTION COMPLETED!")
    print(f"📁 All results saved to: {output_dir}")
    
    return {
        'backtest_results': backtest_results,
        'portfolio_history': portfolio_df,
        'trades': trades_df,
        'performance_report': performance_report if 'performance_report' in locals() else {}
    }


def quick_test(ticker: str = 'AAPL', model: str = 'random_forest') -> None:
    """
    Run a quick test with a single ticker and model.
    
    Args:
        ticker: Stock ticker to test
        model: ML model to use
    """
    print(f"🔬 Quick Test: {ticker} with {model}")
    print("-" * 40)
    
    # Create minimal config
    config = {
        'data': {
            'tickers': [ticker],
            'start_date': '2022-01-01',
            'end_date': '2024-01-01',
            'data_source': 'yahoo',
            'split_ratio': 0.8
        },
        'features': {
            'lookback_periods': [5, 10, 20],
            'technical_indicators': ['sma', 'ema', 'rsi'],
            'volatility_window': 20,
            'return_periods': [1, 5]
        },
        'models': {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 3,
            'models_to_train': [model],
            'hyperparameters': {
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10]
                }
            }
        },
        'strategy': {
            'signal_threshold': 0.6,
            'position_sizing': 'equal_weight',
            'max_positions': 1,
            'transaction_costs': 0.001,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'rebalance_frequency': 'daily',
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_single_position': 1.0,
                'volatility_target': 0.15
            }
        },
        'backtest': {
            'initial_capital': 10000,
            'benchmark': 'SPY',
            'commission': 0.001,
            'slippage': 0.0005,
            'walk_forward': {
                'training_window': 126,
                'validation_window': 21,
                'step_size': 7
            }
        }
    }
    
    try:
        # Run strategy
        results = run_complete_strategy_with_config(config, f'results/quick_test_{ticker}_{model}/')
        print("✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")


def run_complete_strategy_with_config(config: dict, output_dir: str) -> dict:
    """Run strategy with provided config dictionary."""
    
    # Save config for reference
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config_used.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data
    raw_data, processed_data = load_and_prepare_data(config)
    
    if not processed_data:
        raise ValueError("No data loaded")
    
    # Run backtesting
    backtester = Backtester(config)
    backtest_results = backtester.run_backtest(
        data=processed_data,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Generate results
    portfolio_df = pd.DataFrame(backtester.portfolio_history)
    trades_df = backtester.get_trade_analysis()
    
    # Export results
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
    
    if not portfolio_df.empty:
        portfolio_df.to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
    
    return {
        'backtest_results': backtest_results,
        'portfolio_history': portfolio_df,
        'trades': trades_df
    }


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='ML Trading Strategy')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--output', '-o', default='results/', 
                       help='Output directory for results')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with default parameters')
    parser.add_argument('--ticker', default='AAPL',
                       help='Ticker for quick test (default: AAPL)')
    parser.add_argument('--model', default='random_forest',
                       help='Model for quick test (default: random_forest)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test(args.ticker, args.model)
    else:
        run_complete_strategy(args.config, args.output)


if __name__ == "__main__":
    main()
