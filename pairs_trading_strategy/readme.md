# 📈 Pairs Trading Strategy

This project implements a complete pairs trading pipeline using Python and Streamlit.

## 🔧 Features
- Fetches historical stock data from Yahoo Finance
- Computes hedge ratios using linear regression
- Generates trading signals from Z-score of spread
- Backtests strategies and logs trades
- Optimizes strategy parameters
- Fully interactive Streamlit dashboard

## 🗂️ Structure

```
.
├── README.md
├── requirements.txt
├── streamlit_app.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── strategy.py
│   ├── backtester.py
│   ├── evaluator.py
│   ├── trade_log.py
│   └── plotter.py
├── notebooks/
│   └── analysis.ipynb
└── outputs/
```

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run dashboard:
```bash
streamlit run streamlit_app.py
```

3. Or run notebook from `notebooks/` for step-by-step dev

---

Made by Vansh Diora and Vivaan Mehta
