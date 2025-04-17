import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import fetch_price_data, compute_log_returns
from src.garch_model import fit_garch_model, forecast_volatility
from src.plotter import plot_volatility_forecast

st.set_page_config(page_title="📈 Volatility Forecasting Dashboard", layout="wide")

st.title("📈 Volatility Forecasting Dashboard")

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=1, max_value=10, value=5)
    run_forecast = st.button("Run Forecast")

if run_forecast:
    try:
        # Load data
        df = fetch_price_data(ticker, start_date, end_date)
        close = df.loc[:, ('Close', ticker)]
        log_returns = compute_log_returns(close)
        if log_returns.empty:
            st.error("Log returns data is empty. Please check your ticker symbol or date range.")
            st.stop()

        # Fit GARCH model
        garch_result = fit_garch_model(log_returns)

        # Forecast
        forecast = forecast_volatility(garch_result, horizon=forecast_days)

        # Display summary
        st.subheader("Model Summary")
        st.text(garch_result.summary())

        # Plot
        st.subheader("Predicted vs Realized Volatility")
        fig = plot_volatility_forecast(log_returns, forecast)
        st.pyplot(fig)

        # Show forecast table
        st.subheader("📊 Forecast Table")
        st.dataframe(pd.DataFrame({"Predicted Volatility": forecast}, index=pd.date_range(end=log_returns.index[-1], periods=len(forecast), freq='B')))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
