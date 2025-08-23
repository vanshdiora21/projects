"""
Clean ML Trading Strategy Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="ML Trading Strategy",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("🚀 ML Trading Strategy Dashboard")
    st.markdown("Professional AI-powered trading system")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Select Page", ["🏠 Home", "⚙️ Run Strategy", "📈 Results"])
    
    if page == "🏠 Home":
        show_home()
    elif page == "⚙️ Run Strategy":
        show_strategy_runner()
    elif page == "📈 Results":
        show_results()

def show_home():
    """Home page with overview."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center'>
            <h1>🎯 ML Trading Strategy</h1>
            <h3>Professional AI-Powered Trading System</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI Models
        - **Random Forest**
        - **XGBoost** 
        - **Logistic Regression**
        - Walk-forward validation
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Features
        - Technical indicators
        - Risk management
        - Performance analytics
        - Professional backtesting
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Results
        - Interactive charts
        - Trade analysis
        - Performance metrics
        - Export functionality
        """)
    
    st.markdown("---")
    
    # Show latest results if available
    if os.path.exists('results/portfolio_history.csv'):
        st.subheader("📈 Latest Strategy Performance")
        
        df = pd.read_csv('results/portfolio_history.csv')
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            initial_value = 10000
            final_value = df['total_value'].iloc[-1] if 'total_value' in df.columns else df['portfolio_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            col1.metric("💰 Total Return", f"{total_return:.2f}%")
            col2.metric("📊 Final Value", f"${final_value:,.2f}")
            col3.metric("🎯 Trading Days", len(df))
            
            # Calculate win rate if trades exist
            if os.path.exists('results/trades.csv'):
                trades_df = pd.read_csv('results/trades.csv')
                if not trades_df.empty and 'pnl' in trades_df.columns:
                    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
                    col4.metric("🏆 Win Rate", f"{win_rate:.1f}%")
            
            # Quick chart
            dates = pd.to_datetime(df['date'])
            values = df['total_value'] if 'total_value' in df.columns else df['portfolio_value']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.update_layout(
                title="Portfolio Performance Overview",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🚀 No results yet. Run your first strategy to see performance here!")

def show_strategy_runner():
    """Strategy configuration and execution."""
    st.subheader("⚙️ Configure Your Trading Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Data Settings")
        
        ticker = st.selectbox(
            "Select Stock",
            ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            help="Choose the stock to trade"
        )
        
        start_date = st.date_input(
            "Start Date", 
            datetime(2023, 1, 1),
            help="Strategy start date"
        )
        
        end_date = st.date_input(
            "End Date", 
            datetime(2024, 1, 1),
            help="Strategy end date"
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=10000,
            min_value=1000,
            max_value=100000,
            step=1000,
            help="Starting portfolio value"
        )
    
    with col2:
        st.markdown("#### 🤖 Model Settings")
        
        models = st.multiselect(
            "ML Models",
            ['random_forest', 'xgboost'],
            default=['random_forest', 'xgboost'],
            help="Select machine learning models to use"
        )
        
        signal_threshold = st.slider(
            "Signal Threshold",
            0.5, 0.9, 0.6,
            help="Minimum confidence required for trading"
        )
        
        stop_loss = st.slider(
            "Stop Loss (%)",
            1.0, 15.0, 5.0,
            help="Maximum loss before closing position"
        )
        
        take_profit = st.slider(
            "Take Profit (%)", 
            5.0, 30.0, 15.0,
            help="Target profit before closing position"
        )
    
    st.markdown("---")
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest(ticker, start_date, end_date, initial_capital, models, 
                        signal_threshold, stop_loss, take_profit)

def run_backtest(ticker, start_date, end_date, initial_capital, models, 
                signal_threshold, stop_loss, take_profit):
    """Execute the backtest with given parameters."""
    
    if not models:
        st.error("❌ Please select at least one model")
        return
    
    # Create configuration
    config = {
        "data": {
            "tickers": [ticker],
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "data_source": "yahoo",
            "split_ratio": 0.8
        },
        "features": {
            "lookback_periods": [5, 10, 20],
            "technical_indicators": ["sma", "ema", "rsi", "macd"],
            "volatility_window": 20,
            "return_periods": [1, 5]
        },
        "models": {
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 3,
            "models_to_train": models,
            "hyperparameters": {
                "random_forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20]
                },
                "xgboost": {
                    "n_estimators": [100, 200], 
                    "max_depth": [6, 10]
                }
            }
        },
        "strategy": {
            "signal_threshold": signal_threshold,
            "position_sizing": "equal_weight",
            "max_positions": 1,
            "transaction_costs": 0.001,
            "stop_loss": stop_loss / 100,
            "take_profit": take_profit / 100,
            "rebalance_frequency": "daily",
            "risk_management": {
                "max_portfolio_risk": 0.02,
                "max_single_position": 1.0,
                "volatility_target": 0.15
            }
        },
        "backtest": {
            "initial_capital": initial_capital,
            "benchmark": "SPY",
            "commission": 0.001,
            "slippage": 0.0005,
            "walk_forward": {
                "training_window": 60,
                "validation_window": 20,
                "step_size": 7
            }
        },
        "plotting": {
            "figure_size": [12, 8],
            "style": "default",
            "save_plots": True,
            "plot_directory": "plots/",
            "formats": ["png"]
        }
    }
    
    # Save configuration
    with open('dashboard_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run the backtest
    with st.spinner("🔄 Running backtest... This may take a few minutes."):
        try:
            # Use subprocess to run main.py
            result = subprocess.run([
                'python', 'main.py', 
                '--config', 'dashboard_config.json',
                '--output', 'results/'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                st.success("✅ Backtest completed successfully!")
                st.balloons()
                
                # Show execution summary
                with st.expander("📜 Execution Summary"):
                    if result.stdout:
                        st.text(result.stdout)
                
                # Show quick results
                show_quick_results()
                
            else:
                st.error("❌ Backtest failed!")
                st.error("Please check your configuration and try again.")
                
                if result.stderr:
                    with st.expander("❌ Error Details"):
                        st.code(result.stderr)
                        
        except Exception as e:
            st.error(f"❌ Error running backtest: {str(e)}")

def show_quick_results():
    """Show quick results after backtest completion."""
    if os.path.exists('results/portfolio_history.csv'):
        df = pd.read_csv('results/portfolio_history.csv')
        
        if not df.empty:
            initial_value = 10000
            final_value = df['total_value'].iloc[-1] if 'total_value' in df.columns else df['portfolio_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            st.success(f"🎉 Strategy completed with {total_return:.2f}% total return!")
            
            # Quick metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{total_return:.2f}%")
            col2.metric("Final Value", f"${final_value:,.2f}")
            col3.metric("Days", len(df))

def show_results():
    """Display detailed backtest results."""
    st.subheader("📈 Detailed Results Analysis")
    
    if not os.path.exists('results/portfolio_history.csv'):
        st.warning("⚠️ No results found. Please run a backtest first.")
        return
    
    # Load portfolio data
    portfolio_df = pd.read_csv('results/portfolio_history.csv')
    
    if portfolio_df.empty:
        st.error("❌ Portfolio data is empty")
        return
    
    # Performance Summary
    st.subheader("📊 Performance Summary")
    
    initial_value = 10000
    final_value = portfolio_df['total_value'].iloc[-1] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate additional metrics
    values = portfolio_df['total_value'] if 'total_value' in portfolio_df.columns else portfolio_df['portfolio_value']
    returns = pd.Series(values).pct_change().dropna()
    
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Max drawdown
    peak = values.expanding().max()
    drawdown = ((values - peak) / peak * 100).min()
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("💰 Total Return", f"{total_return:.2f}%", delta=f"${final_value - initial_value:.2f}")
    col2.metric("📊 Final Value", f"${final_value:,.2f}", delta=f"from ${initial_value:,.2f}")
    col3.metric("⚡ Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("📉 Max Drawdown", f"{abs(drawdown):.2f}%")
    
    # Portfolio Performance Chart
    st.subheader("📈 Portfolio Performance")
    
    dates = pd.to_datetime(portfolio_df['date'])
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Initial value reference line
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: ${initial_value:,}"
    )
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown Chart
    st.subheader("📉 Drawdown Analysis")
    
    drawdown_series = (values - peak) / peak * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dates,
        y=drawdown_series,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='red', width=2),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    fig_dd.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Trade Analysis
    if os.path.exists('results/trades.csv'):
        trades_df = pd.read_csv('results/trades.csv')
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            st.subheader("💼 Trade Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Trade statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = winning_trades / total_trades * 100
            
            col1.metric("🔢 Total Trades", total_trades)
            col2.metric("🏆 Win Rate", f"{win_rate:.1f}%")
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
            
            col3.metric("📈 Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "$0.00")
            col4.metric("📉 Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "$0.00")
            
            # Trade distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_trades = go.Figure()
                fig_trades.add_trace(go.Histogram(
                    x=trades_df['pnl'],
                    nbinsx=20,
                    name='Trade P&L',
                    marker_color='lightblue'
                ))
                fig_trades.add_vline(x=0, line_dash="dash", line_color="red")
                fig_trades.update_layout(title="Trade P&L Distribution", height=400)
                st.plotly_chart(fig_trades, use_container_width=True)
            
            with col2:
                # Cumulative P&L
                cumulative_pnl = trades_df['pnl'].cumsum()
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=list(range(len(cumulative_pnl))),
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green', width=3)
                ))
                fig_cum.update_layout(title="Cumulative Trade P&L", height=400)
                st.plotly_chart(fig_cum, use_container_width=True)
    
    # Download Section
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = portfolio_df.to_csv(index=False)
        st.download_button(
            "📊 Portfolio Data",
            csv_data,
            "portfolio_results.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if os.path.exists('results/trades.csv'):
            trades_data = pd.read_csv('results/trades.csv').to_csv(index=False)
            st.download_button(
                "💼 Trades Data",
                trades_data,
                "trades_results.csv", 
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if os.path.exists('results/performance_metrics.json'):
            with open('results/performance_metrics.json', 'r') as f:
                metrics_data = f.read()
            st.download_button(
                "📈 Performance Metrics",
                metrics_data,
                "performance_metrics.json",
                "application/json",
                use_container_width=True
            )

if __name__ == '__main__':
    main()
