# ML Trading Strategy Methodology

## Abstract
This project implements a sophisticated machine learning-driven trading strategy using ensemble methods, walk-forward validation, and institutional-grade risk management.

## Research Question
Can machine learning models consistently predict short-term price movements and generate alpha in equity markets?

## Methodology

### Data Sources
- Yahoo Finance API for historical price data
- 50+ engineered features including technical indicators
- Business days only (252 trading days per year)

### Feature Engineering
- **Price-based**: Returns (1d, 5d), log returns, volatility
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Market Microstructure**: High-low ratios, volume indicators

### Machine Learning Models
1. **Random Forest**: Ensemble method handling non-linear relationships
2. **XGBoost**: Gradient boosting for high-performance predictions  
3. **Logistic Regression**: Linear baseline with regularization

### Backtesting Framework
- **Walk-Forward Validation**: 60-day training, 20-day validation windows
- **Model Retraining**: Every 7 days to adapt to market changes
- **Transaction Costs**: 0.1% commission + 0.05% slippage
- **Risk Management**: 5% stop-loss, 15% take-profit, position sizing

### Performance Metrics
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Volatility
- Win Rate, Profit Factor, Expectancy

## Academic References
- Jansen, M. (2020). Machine Learning for Algorithmic Trading
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio
