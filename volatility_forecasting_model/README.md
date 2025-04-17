# 📊 Volatility Forecasting Model (GARCH)

This project implements a volatility forecasting model for financial time series using the GARCH(𝑝,𝑞) framework. The goal is to predict the future volatility of a stock's returns, compare it against realized volatility, and analyze the model's sensitivity to parameter changes.

---

## 🚀 Project Overview

Volatility is a key metric in risk management, options pricing, and portfolio optimization. This model aims to forecast short-term volatility for a stock (e.g., AAPL) using a GARCH(1,1) model and evaluate its effectiveness by:

- Computing log returns from historical price data
- Fitting a GARCH model using maximum likelihood estimation
- Forecasting volatility over a 5-day horizon
- Comparing predicted volatility with realized volatility
- Performing a parameter sensitivity analysis on GARCH(p,q)

---

## 📂 Project Structure

volatility_forecasting_model/ │ ├── notebook/ │ └── volatility_forecasting_garch.ipynb # Full exploratory notebook │ ├── src/ │ ├── data_loader.py # Data fetching + log return computation │ ├── garch_model.py # GARCH model fitting & forecasting │ └── plotter.py # Plotting functions │ ├── requirements.txt # Dependencies ├── README.md └── .gitignore


---

## 🧪 Methodology

1. **Data Loading**  
   Uses `yfinance` to download historical price data for a given ticker.

2. **Log Return Calculation**  
   Computes daily log returns from adjusted close prices.

3. **GARCH Model Fitting**  
   A GARCH(1,1) model is fit using the `arch` library.

4. **Volatility Forecasting**  
   5-day out-of-sample volatility forecasts are generated.

5. **Visualization**  
   Compares predicted vs realized volatility and visualizes parameter sensitivity.

---

## 📊 Example Output

- ✔️ GARCH Model Summary
- ✔️ 5-Day Forecasted Volatility:  


- 📈 Predicted vs Realized Volatility  
![Predicted vs Realized](notebook/garch_plot.png)

- 🔍 Sensitivity Analysis  
![Sensitivity Analysis](notebook/sensitivity_plot.png)

---

## 💡 How to Run

1. Clone the repository:
 ```bash
 git clone https://github.com/yourusername/volatility_forecasting_model.git
 cd volatility_forecasting_model


