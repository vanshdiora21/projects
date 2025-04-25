# Market-Making Simulation Project

## Overview
This project simulates a market-making strategy in a dynamic market environment. It features:
- Two competing market makers (MM1 and MM2) quoting independently.
- Mean-reverting and trending market regimes.
- External participants and noise traders submitting market and limit orders.
- Liquidity shocks that adapt to market maker inventory.
- Full analytics including PnL attribution, inventory trends, market impact analysis, and risk metrics.
- A Streamlit dashboard for real-time visualization.

---

## Key Features
- **Market Regimes**: Supports both *mean-reverting* and *trending* price dynamics.
- **Multiple Market Makers**: MM1 and MM2 quote independently, compete in the order book, and manage separate inventories.
- **Order Flow Skewing**: MM1 adapts quotes based on recent order flow.
- **Liquidity Shocks**: Periodically introduces large market orders, with direction based on MM1's inventory.
- **Noise Traders**: Submit random market orders to simulate external volatility.
- **Risk Metrics**: Max drawdown, inventory variance, Sharpe ratio.
- **PnL Attribution Plots**: Realized, unrealized, total PnL, and inventory levels visualized.
- **Market Impact Analysis**: Scatter plot of order size vs. price impact.
- **Live Dashboard**: Powered by Streamlit for interactive visualization.

---

## File Structure
```
/core
  market.py          # Market simulator core logic
  market_maker.py    # Market maker behavior
  order_book.py      # Order book logic
/analysis
  evaluator.py       # Analytics, risk metrics, plots
exports/             # Contains CSV exports of metrics
streamlit_dashboard.py  # Dashboard visualization
config.json          # Simulation parameters
main.py              # Entry point for running simulations
```

---

## Setup Instructions
1. **Clone the repository**.
2. **Install dependencies**:
```bash
pip install streamlit matplotlib pandas
```
3. **Run the simulation**:
```bash
python main.py --regime mean-reverting --steps 20 --mm1_limit 20 --mm2_limit 10
```
4. **Launch the dashboard**:
```bash
streamlit run streamlit_dashboard.py
```

---

## Configuration
Modify **`config.json`** to adjust:
- Inventory limits
- Base spread and order sizes
- Order flow bias scaling
- Noise trader behavior

You can also override key settings using CLI flags (e.g., regime, steps).

---

## Key Outputs
- **PnL Metrics**: Exports to `exports/MM1_metrics.csv` and `exports/MM2_metrics.csv`.
- **Market Impact**: Exports to `exports/MM1_market_impact.csv` and `exports/MM2_market_impact.csv`.
- **Plots**: Rendered live in the dashboard.

---

## Contact
For any questions or improvements, feel free to reach out!

