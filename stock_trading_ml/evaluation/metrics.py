"""
Performance evaluation metrics for ML trading strategies.

This module provides:
- Financial performance metrics (Sharpe ratio, max drawdown, etc.)
- ML model evaluation metrics
- Risk analysis functions
- Benchmark comparison utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    Calculate and analyze trading strategy performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns_metrics(self, portfolio_values: pd.Series, benchmark_values: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate return-based performance metrics.
        """
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic return metrics - FIXED LINE
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

        
        # Annualized metrics
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        daily_rf_rate = self.risk_free_rate / 252
        excess_returns = returns - daily_rf_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'trading_days': trading_days
        }
        
        # Add benchmark comparison if provided
        if benchmark_values is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_values)
            metrics.update(benchmark_metrics)
        
        return metrics

        
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Maximum drawdown as a negative percentage
        """
        # Calculate running maximum (peak)
        peak = portfolio_values.expanding().max()
        
        # Calculate drawdown from peak
        drawdown = (portfolio_values - peak) / peak
        
        # Return maximum drawdown (most negative value)
        return drawdown.min()

    def calculate_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate drawdown time series.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Time series of drawdown percentages
        """
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            returns: Return time series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Sort returns
        sorted_returns = returns.sort_values()
        
        # Calculate VaR
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns.iloc[var_index] if var_index < len(sorted_returns) else sorted_returns.iloc[0]
        
        # Calculate CVaR (expected return given return is worse than VaR)
        cvar = sorted_returns[sorted_returns <= var].mean()
        
        return var, cvar
    
    def calculate_win_loss_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate win/loss statistics from trades.
        
        Args:
            trades_df: DataFrame with trade data including 'pnl' column
            
        Returns:
            Dictionary of win/loss metrics
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return {}
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_trades = len(trades_df)
        if total_trades == 0:
            return {}
        
        # Win rate
        win_rate = len(winning_trades) / total_trades
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Largest win/loss
        largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        
        # Average trade duration
        if 'duration_days' in trades_df.columns:
            avg_duration = trades_df['duration_days'].mean()
        else:
            avg_duration = None
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration': avg_duration
        }
        
        return metrics
    
    def calculate_ml_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate ML model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of ML metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC if probabilities are provided
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk-related metrics.
        
        Args:
            returns: Return time series
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) < 2:
            return {}
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR and CVaR
        var_95, cvar_95 = self.calculate_var_cvar(returns, 0.05)
        var_99, cvar_99 = self.calculate_var_cvar(returns, 0.01)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Positive/negative return statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        metrics = {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_99': var_99,
            'cvar_99': cvar_99,
            'positive_return_ratio': len(positive_returns) / len(returns),
            'avg_positive_return': positive_returns.mean() if len(positive_returns) > 0 else 0,
            'avg_negative_return': negative_returns.mean() if len(negative_returns) > 0 else 0
        }
        
        return metrics
    
    def _calculate_benchmark_metrics(self, portfolio_returns: pd.Series, 
                                   benchmark_values: pd.Series) -> Dict[str, float]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_values: Benchmark value series
            
        Returns:
            Dictionary of benchmark comparison metrics
        """
        # Align dates
        benchmark_returns = benchmark_values.pct_change().dropna()
        
        # Find common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        port_aligned = portfolio_returns[common_dates]
        bench_aligned = benchmark_returns[common_dates]
        
        # Calculate metrics
        correlation = port_aligned.corr(bench_aligned)
        
        # Beta
        covariance = np.cov(port_aligned, bench_aligned)[0][asset:1]
        benchmark_variance = bench_aligned.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Alpha (Jensen's alpha)
        port_annual_return = (1 + port_aligned.mean()) ** 252 - 1
        bench_annual_return = (1 + bench_aligned.mean()) ** 252 - 1
        alpha = port_annual_return - (self.risk_free_rate + beta * (bench_annual_return - self.risk_free_rate))
        
        # Information ratio
        excess_returns = port_aligned - bench_aligned
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Treynor ratio
        treynor_ratio = (port_annual_return - self.risk_free_rate) / beta if beta != 0 else 0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation_with_benchmark': correlation,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'benchmark_annual_return': bench_annual_return
        }


class RiskAnalyzer:
    """
    Advanced risk analysis tools.
    """
    
    def __init__(self):
        self.metrics_calculator = PerformanceMetrics()
    
    def rolling_metrics(self, portfolio_values: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Portfolio value time series
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = portfolio_values.pct_change().dropna()
        
        results = []
        
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            period_values = portfolio_values.iloc[i-window:i]
            
            metrics = self.metrics_calculator.calculate_returns_metrics(period_values)
            metrics['date'] = returns.index[i-1]
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def stress_test_analysis(self, returns: pd.Series, scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing on portfolio returns.
        
        Args:
            returns: Portfolio return series
            scenarios: Dictionary of stress scenarios (name -> return shock)
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply shock to returns
            stressed_returns = returns + shock
            
            # Calculate metrics under stress
            stressed_values = (1 + stressed_returns).cumprod()
            stressed_metrics = self.metrics_calculator.calculate_returns_metrics(stressed_values)
            
            results[scenario_name] = stressed_metrics
        
        return results
    
    def monte_carlo_simulation(self, returns: pd.Series, 
                              n_simulations: int = 1000, 
                              time_horizon: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio projections.
        
        Args:
            returns: Historical return series
            n_simulations: Number of simulation runs
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with simulation results
        """
        # Calculate return statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Run simulations
        simulated_paths = []
        final_returns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, time_horizon)
            
            # Calculate cumulative path
            cumulative_returns = (1 + random_returns).cumprod()
            simulated_paths.append(cumulative_returns)
            final_returns.append(cumulative_returns[-1] - 1)
        
        # Calculate statistics
        final_returns = np.array(final_returns)
        
        results = {
            'mean_final_return': np.mean(final_returns),
            'std_final_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_25': np.percentile(final_returns, 25),
            'percentile_50': np.percentile(final_returns, 50),
            'percentile_75': np.percentile(final_returns, 75),
            'percentile_95': np.percentile(final_returns, 95),
            'probability_of_loss': np.sum(final_returns < 0) / n_simulations,
            'simulated_paths': simulated_paths
        }
        
        return results


def generate_performance_report(portfolio_values: pd.Series, 
                              trades_df: pd.DataFrame,
                              benchmark_values: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Generate comprehensive performance report.
    
    Args:
        portfolio_values: Portfolio value time series
        trades_df: DataFrame with trade information
        benchmark_values: Optional benchmark comparison
        
    Returns:
        Complete performance report
    """
    metrics_calc = PerformanceMetrics()
    
    # Calculate all metrics
    returns_metrics = metrics_calc.calculate_returns_metrics(portfolio_values, benchmark_values)
    
    portfolio_returns = portfolio_values.pct_change().dropna()
    risk_metrics = metrics_calc.calculate_risk_metrics(portfolio_returns)
    
    trade_metrics = metrics_calc.calculate_win_loss_metrics(trades_df)
    
    # Drawdown analysis
    drawdown_series = metrics_calc.calculate_drawdown_series(portfolio_values)
    
    report = {
        'summary': {
            'start_value': portfolio_values.iloc[0],
            'end_value': portfolio_values.iloc[-1],
            'start_date': portfolio_values.index,
            'end_date': portfolio_values.index[-1]
        },
        'returns_metrics': returns_metrics,
        'risk_metrics': risk_metrics,
        'trade_metrics': trade_metrics,
        'drawdown_analysis': {
            'max_drawdown': drawdown_series.min(),
            'current_drawdown': drawdown_series.iloc[-1],
            'drawdown_series': drawdown_series
        }
    }
    
    return report


if __name__ == "__main__":
    print("Performance Metrics module loaded successfully")
    print("Available metrics:")
    print("- Return-based metrics (Sharpe, Sortino, Calmar ratios)")
    print("- Risk metrics (VaR, CVaR, volatility)")
    print("- Trade analysis (win rate, profit factor)")
    print("- Benchmark comparison (alpha, beta, information ratio)")
    print("- Monte Carlo simulation")
    print("- Stress testing")
