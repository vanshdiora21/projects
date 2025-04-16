
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import fetch_data
from src.strategy import get_hedge_ratio, compute_spread, generate_zscore_signals
from src.backtester import compute_returns, backtest_pairs, compute_pnl_metrics
from src.evaluator import evaluate_strategy
from src.trade_log import log_trades
from src.plotter import plot_trade_signals

st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Strategy Dashboard")

# Sidebar: User Inputs
st.sidebar.header("Strategy Configuration")
ticker1 = st.sidebar.text_input("Ticker 1", "AAPL")
ticker2 = st.sidebar.text_input("Ticker 2", "MSFT")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
entry_z = st.sidebar.slider("Entry Z-Score", 0.5, 3.0, 1.0, 0.1)
exit_z = st.sidebar.slider("Exit Z-Score", 0.0, 2.0, 0.0, 0.1)

run_button = st.sidebar.button("Run Backtest")

# Run strategy if user clicks button
if run_button:
    try:
        df1 = fetch_data(ticker1, str(start_date), str(end_date))
        df2 = fetch_data(ticker2, str(start_date), str(end_date))

        st.write("DF1 Preview:", df1.head())
        st.write("DF2 Preview:", df2.head())


        combined = pd.concat([df1['Close'], df2['Close']], axis=1).dropna()
        combined.columns = [ticker1, ticker2]

        hedge_ratio = get_hedge_ratio(combined[ticker1], combined[ticker2])
        spread = compute_spread(combined[ticker1], combined[ticker2], hedge_ratio)
        signals, zscore = generate_zscore_signals(spread, entry_z, exit_z)

        returns = compute_returns(combined)
        strat_returns = backtest_pairs(signals, returns, hedge_ratio)
        pnl_df = compute_pnl_metrics(strat_returns)
        metrics = evaluate_strategy(pnl_df)
        trade_log = log_trades(signals, combined, ticker1, ticker2)

        # Metrics
        st.subheader("📊 Performance Metrics")
        st.dataframe(pd.DataFrame(metrics, index=[f"{ticker1}/{ticker2}"]))

        # Plots
        st.subheader("📉 Price Chart with Signals")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(combined[ticker1], label='Price', alpha=0.6)
        ax1.scatter(signals[signals == 1].index, combined[ticker1][signals == 1], label='Buy', marker='^', color='green')
        ax1.scatter(signals[signals == -1].index, combined[ticker1][signals == -1], label='Sell', marker='v', color='red')
        ax1.legend()
        ax1.set_title("Price & Trade Signals")
        st.pyplot(fig1)

        st.subheader("📐 Z-Score Plot")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(zscore, label='Z-score')
        ax2.axhline(entry_z, color='red', linestyle='--', label='Entry Short')
        ax2.axhline(-entry_z, color='green', linestyle='--', label='Entry Long')
        ax2.axhline(0, color='black')
        ax2.set_title("Z-Score with Thresholds")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("📈 Cumulative Returns")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(pnl_df['Cumulative Return'], label='Cumulative Return')
        ax3.set_title("Cumulative Strategy Return")
        ax3.legend()
        st.pyplot(fig3)

        # Trade Log
        st.subheader("📋 Trade Log")
        st.dataframe(trade_log)

        # Download CSV
        csv = trade_log.to_csv(index=False).encode('utf-8')
        st.download_button("Download Trade Log as CSV", csv, "trade_log.csv", "text/csv")

    except Exception as e:
        st.error(f"Error running strategy: {e}")
