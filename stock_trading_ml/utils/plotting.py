"""
Visualization utilities for ML trading strategy analysis.

This module provides:
- Performance plotting functions
- Chart generation for backtesting results
- Model analysis visualizations
- Risk analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class TradingVisualizer:
    """
    Comprehensive visualization tools for trading strategy analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        plt.style.use('default')  # Use default since seaborn-v0_8 may not be available
        sns.set_palette("husl")
        
        # Color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_portfolio_performance(self, portfolio_df: pd.DataFrame, 
                                  benchmark_df: Optional[pd.DataFrame] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio performance over time.
        
        Args:
            portfolio_df: DataFrame with portfolio history
            benchmark_df: Optional benchmark data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Portfolio value
        dates = pd.to_datetime(portfolio_df['date']) if 'date' in portfolio_df.columns else portfolio_df.index
        values = portfolio_df['total_value'] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value']
        
        ax1.plot(dates, values, linewidth=2, label='Portfolio', color=self.colors['primary'])
        
        if benchmark_df is not None:
            bench_dates = pd.to_datetime(benchmark_df['date']) if 'date' in benchmark_df.columns else benchmark_df.index
            bench_values = benchmark_df['value'] if 'value' in benchmark_df.columns else benchmark_df.iloc[:, 0]
            ax1.plot(bench_dates, bench_values, linewidth=2, label='Benchmark', color=self.colors['secondary'])
        
        ax1.set_title('Portfolio Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns distribution
        if 'daily_return' in portfolio_df.columns:
            returns = portfolio_df['daily_return'].dropna()
            ax2.hist(returns, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax2.axvline(returns.mean(), color=self.colors['danger'], linestyle='--', 
                       label=f'Mean: {returns.mean():.4f}')
            ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Daily Return', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown_analysis(self, portfolio_df: pd.DataFrame, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            portfolio_df: DataFrame with portfolio history
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        dates = pd.to_datetime(portfolio_df['date']) if 'date' in portfolio_df.columns else portfolio_df.index
        values = portfolio_df['total_value'] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value']
        
        # Calculate drawdown
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100
        
        # Plot 1: Portfolio value with peaks
        ax1.plot(dates, values, linewidth=2, label='Portfolio Value', color=self.colors['primary'])
        ax1.plot(dates, peak, linewidth=1, linestyle='--', label='Peak Value', color=self.colors['danger'])
        ax1.fill_between(dates, values, peak, alpha=0.3, color=self.colors['danger'])
        ax1.set_title('Portfolio Value and Drawdowns', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown series
        ax2.fill_between(dates, drawdown, 0, alpha=0.7, color=self.colors['danger'])
        ax2.plot(dates, drawdown, linewidth=1, color='darkred')
        ax2.axhline(drawdown.min(), color='black', linestyle='--', 
                   label=f'Max Drawdown: {drawdown.min():.2f}%')
        ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trade analysis charts.
        
        Args:
            trades_df: DataFrame with trade data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if trades_df.empty:
            print("No trades to plot")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: P&L distribution
        pnl = trades_df['pnl'] if 'pnl' in trades_df.columns else trades_df['pnl_pct']
        ax1.hist(pnl, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
        ax1.axvline(0, color=self.colors['danger'], linestyle='--', linewidth=2)
        ax1.axvline(pnl.mean(), color=self.colors['success'], linestyle='--', 
                   label=f'Mean P&L: ${pnl.mean():.2f}')
        ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative P&L
        cumulative_pnl = pnl.cumsum()
        trade_numbers = range(1, len(cumulative_pnl) + 1)
        ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color=self.colors['primary'])
        ax2.fill_between(trade_numbers, cumulative_pnl, 0, alpha=0.3, color=self.colors['primary'])
        ax2.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win/Loss analysis
        winning_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df.columns else pd.DataFrame()
        losing_trades = trades_df[trades_df['pnl'] <= 0] if 'pnl' in trades_df.columns else pd.DataFrame()
        
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [len(winning_trades), len(losing_trades)]
        colors = [self.colors['success'], self.colors['danger']]
        
        if sum(sizes) > 0:
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')
        
        # Plot 4: Trade duration analysis
        if 'duration_days' in trades_df.columns:
            duration = trades_df['duration_days'].dropna()
            if len(duration) > 0:
                ax4.hist(duration, bins=20, alpha=0.7, color=self.colors['info'], edgecolor='black')
                ax4.axvline(duration.mean(), color=self.colors['danger'], linestyle='--',
                           label=f'Mean Duration: {duration.mean():.1f} days')
                ax4.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Duration (Days)', fontsize=12)
                ax4.set_ylabel('Frequency', fontsize=12)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature names and importance scores
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if importance_df.empty:
            print("No feature importance data to plot")
            return None
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=self.colors['primary'], alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01 * max(top_features['importance']), bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, model_metrics: Dict[str, Dict], 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model performance comparison.
        
        Args:
            model_metrics: Dictionary of model performance metrics
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not model_metrics:
            print("No model metrics to plot")
            return None
        
        # Extract metrics for comparison
        metrics_to_compare = ['val_accuracy', 'roc_auc']  # Add more as needed
        
        fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(15, 6))
        if len(metrics_to_compare) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_compare):
            model_names = []
            metric_values = []
            
            for model_name, metrics in model_metrics.items():
                if metric in metrics:
                    model_names.append(model_name)
                    metric_values.append(metrics[metric])
            
            if metric_values:
                bars = axes[i].bar(model_names, metric_values, 
                                  color=[self.colors['primary'], self.colors['secondary'], 
                                        self.colors['success'], self.colors['warning']][:len(model_names)])
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, portfolio_df: pd.DataFrame, 
                                   trades_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            portfolio_df: Portfolio performance data
            trades_df: Trade data
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Performance', 'Drawdown Analysis',
                          'Trade P&L Distribution', 'Cumulative P&L'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        dates = pd.to_datetime(portfolio_df['date']) if 'date' in portfolio_df.columns else portfolio_df.index
        values = portfolio_df['total_value'] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value']
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(x=dates, y=values, mode='lines', name='Portfolio Value',
                      line=dict(color=self.colors['primary'], width=2)),
            row=1, col=1
        )
        
        # Drawdown
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, mode='lines', name='Drawdown',
                      fill='tonexty', fillcolor='rgba(214, 39, 40, 0.3)',
                      line=dict(color='darkred', width=1)),
            row=1, col=2
        )
        
        # Trade P&L distribution
        if not trades_df.empty and 'pnl' in trades_df.columns:
            fig.add_trace(
                go.Histogram(x=trades_df['pnl'], name='Trade P&L', 
                           marker=dict(color=self.colors['primary'], opacity=0.7)),
                row=2, col=1
            )
            
            # Cumulative P&L
            cumulative_pnl = trades_df['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(x=list(range(1, len(cumulative_pnl) + 1)), y=cumulative_pnl,
                          mode='lines', name='Cumulative P&L',
                          line=dict(color=self.colors['success'], width=2)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Trading Strategy Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        return fig
    
    def save_all_plots(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame,
                      feature_importance: Optional[pd.DataFrame] = None,
                      model_metrics: Optional[Dict] = None,
                      save_dir: str = 'plots/') -> None:
        """
        Save all visualization plots.
        
        Args:
            portfolio_df: Portfolio data
            trades_df: Trade data
            feature_importance: Feature importance data
            model_metrics: Model performance metrics
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Portfolio performance
        self.plot_portfolio_performance(portfolio_df, save_path=f'{save_dir}/portfolio_performance.png')
        plt.close()
        
        # Drawdown analysis
        self.plot_drawdown_analysis(portfolio_df, save_path=f'{save_dir}/drawdown_analysis.png')
        plt.close()
        
        # Trade analysis
        if not trades_df.empty:
            self.plot_trade_analysis(trades_df, save_path=f'{save_dir}/trade_analysis.png')
            plt.close()
        
        # Feature importance
        if feature_importance is not None and not feature_importance.empty:
            self.plot_feature_importance(feature_importance, save_path=f'{save_dir}/feature_importance.png')
            plt.close()
        
        # Model comparison
        if model_metrics:
            self.plot_model_comparison(model_metrics, save_path=f'{save_dir}/model_comparison.png')
            plt.close()
        
        print(f"All plots saved to {save_dir}")


def create_performance_summary_plot(metrics: Dict[str, float], 
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a summary plot of key performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Select key metrics to display
    key_metrics = {
        'Total Return': metrics.get('total_return', 0) * 100,
        'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
        'Max Drawdown': abs(metrics.get('max_drawdown', 0)) * 100,
        'Win Rate': metrics.get('win_rate', 0) * 100,
        'Volatility': metrics.get('annualized_volatility', 0) * 100
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.bar(key_metrics.keys(), key_metrics.values(), 
                 color=['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd'])
    
    # Customize plot
    ax.set_title('Strategy Performance Summary', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, (metric, value) in zip(bars, key_metrics.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{value:.2f}{"%" if metric != "Sharpe Ratio" else ""}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Trading Visualizer module loaded successfully")
    print("Available plots:")
    print("- Portfolio performance charts")
    print("- Drawdown analysis")
    print("- Trade analysis")
    print("- Feature importance")
    print("- Model comparison")
    print("- Interactive dashboards")
