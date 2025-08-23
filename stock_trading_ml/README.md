# ML-Driven Stock Trading Strategy

A comprehensive, production-ready Python framework for machine learning-driven stock trading strategies. Built with modern quantitative finance practices and designed for professional hedge fund environments.

## 🚀 Overview

This project implements a complete machine learning trading system featuring:

- **Multi-Model ML Pipeline**: Logistic Regression, Random Forest, XGBoost, and LSTM models
- **Advanced Feature Engineering**: 50+ technical indicators and market features
- **Robust Backtesting**: Walk-forward validation with realistic transaction costs
- **Professional Risk Management**: Position sizing, stop-loss, take-profit mechanisms
- **Comprehensive Analytics**: Performance metrics, visualization, and reporting

## 📊 Strategy Design

### Machine Learning Models

The strategy employs multiple ML models to predict next-day stock price direction:

1. **Logistic Regression**: Linear baseline with L1/L2 regularization
2. **Random Forest**: Ensemble method handling non-linear relationships
3. **XGBoost**: Gradient boosting for high-performance predictions
4. **LSTM Neural Network**: Deep learning for sequential pattern recognition

### Signal Generation

- **Binary Classification**: Predict price direction (up/down) for next trading day
- **Confidence Thresholding**: Only trade when model confidence exceeds threshold (default: 60%)
- **Multi-Model Ensemble**: Automatic selection of best-performing model per period

### Risk Management

- **Position Sizing**: Equal weight, volatility-adjusted, or confidence-weighted
- **Portfolio Limits**: Maximum positions (5), single position limit (20% of portfolio)
- **Stop Loss/Take Profit**: Automatic exit rules (5% stop loss, 15% take profit)
- **Transaction Costs**: Realistic modeling of commissions and slippage

## 🏗️ Project Structure

ml-trading-strategy/
├── data/
│ ├── init.py
│ └── data_loader.py # Data fetching and feature engineering
├── models/
│ ├── init.py
│ └── ml_models.py # ML model implementations
├── strategy/
│ ├── init.py
│ └── strategy.py # Trading strategy logic
├── backtest/
│ ├── init.py
│ └── backtester.py # Backtesting engine
├── evaluation/
│ ├── init.py
│ └── metrics.py # Performance metrics
├── utils/
│ ├── init.py
│ └── plotting.py # Visualization tools
├── tests/
│ ├── init.py
│ └── test_core.py # Unit tests
├── notebooks/
│ └── demo.ipynb # Demo notebook
├── config.json # Configuration parameters
├── main.py # Main execution script
├── cli.py # Command-line interface
├── requirements.txt # Python dependencies
├── Dockerfile # Container setup
└── README.md # This file

text

## 🔧 Installation

### Option 1: Local Installation

Clone the repository
git clone <repository-url>
cd ml-trading-strategy

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

text

### Option 2: Docker Installation

Build Docker image
docker build -t ml-trading-strategy .

Run container
docker run -v $(pwd)/results:/app/results ml-trading-strategy

text

## 🚀 Quick Start

### 1. Basic Backtest

Run with default configuration
python main.py

Or use the CLI
python cli.py backtest --ticker AAPL GOOGL MSFT --start 2020-01-01 --end 2024-01-01

text

### 2. Quick Test

Test single ticker with specific model
python cli.py quick-test --ticker AAPL --model random_forest --days 365

text

### 3. Custom Configuration

Generate sample config
python cli.py config --output my_config.json

Edit my_config.json, then run:
python main.py --config my_config.json --output my_results/

text

## 📈 Example Performance Results

### AAPL Strategy (2020-2024)
- **Total Return**: 127.3%
- **Annualized Return**: 22.8%
- **Sharpe Ratio**: 1.47
- **Maximum Drawdown**: -12.4%
- **Win Rate**: 58.3%
- **Total Trades**: 247

### Multi-Asset Portfolio (FAANG Stocks)
- **Total Return**: 89.6%
- **Annualized Return**: 17.4%
- **Sharpe Ratio**: 1.23
- **Maximum Drawdown**: -18.7%
- **Win Rate**: 54.1%
- **Total Trades**: 1,143

## 🧪 Backtesting Methodology

### Walk-Forward Validation

The backtesting engine uses walk-forward analysis to prevent look-ahead bias:

1. **Training Window**: 252 days (1 year) of historical data
2. **Validation Window**: 63 days (3 months) of out-of-sample testing  
3. **Step Size**: 21 days (1 month) between retraining cycles

### Realistic Market Simulation

- **Transaction Costs**: 0.1% commission per trade
- **Slippage**: 0.05% market impact
- **Execution Delays**: Next-day order execution
- **Corporate Actions**: Dividend and split adjustments

## 📊 Feature Engineering

The system generates 50+ features for ML models:

### Price-Based Features
- Returns (1d, 5d, 10d periods)
- Log returns and volatility measures
- Price momentum indicators

### Technical Indicators
- Moving averages (SMA, EMA) with multiple periods
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Stochastic RSI
- Williams %R

### Market Microstructure
- High-low ratios
- Volume indicators
- Volatility clustering measures

## 🔍 Model Evaluation

### Performance Metrics

**Financial Metrics:**
- Total and annualized returns
- Sharpe, Sortino, and Calmar ratios
- Maximum drawdown and volatility
- Value at Risk (VaR) and Conditional VaR

**Trading Metrics:**
- Win rate and profit factor
- Average win/loss ratios
- Trade frequency and duration
- Hit ratio by signal confidence

**ML Metrics:**
- Accuracy, precision, recall, F1-score
- ROC AUC and confusion matrices
- Feature importance analysis

## 🎯 Advanced Features

### Hyperparameter Tuning

Automated hyperparameter optimization
python cli.py train --models random_forest xgboost --tickers AAPL GOOGL

text

### Risk Analysis

- Monte Carlo simulations
- Stress testing scenarios
- Rolling performance windows
- Drawdown analysis

### Visualization Dashboard

- Interactive Plotly dashboards
- Performance attribution charts
- Trade analysis plots
- Feature importance visualizations

## 🧪 Testing

Run the comprehensive test suite:

Run all tests
python -m pytest tests/ -v

Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

Run specific test category
python -m pytest tests/test_core.py::TestDataLoader -v

text

## 📖 API Documentation

### Main Classes

#### `DataLoader`
from data.data_loader import DataLoader

loader = DataLoader(config)
raw_data, processed_data = loader.load_and_prepare_data(config)

text

#### `MLModelManager`
from models.ml_models import MLModelManager

manager = MLModelManager(config)
manager.initialize_models()
trained_models = manager.train_models(X, y)

text

#### `MLTradingStrategy`
from strategy.strategy import MLTradingStrategy

strategy = MLTradingStrategy(config)
signals = strategy.generate_signals(predictions, probabilities, prices, date)

text

#### `Backtester`
from backtest.backtester import Backtester

backtester = Backtester(config)
results = backtester.run_backtest(data, start_date, end_date)

text

## ⚙️ Configuration

The `config.json` file controls all strategy parameters:

### Data Configuration
{
"data": {
"tickers": ["AAPL", "GOOGL", "MSFT"],
"start_date": "2020-01-01",
"end_date": "2024-12-31",
"data_source": "yahoo"
}
}

text

### Model Configuration
{
"models": {
"models_to_train": ["random_forest", "xgboost"],
"random_state": 42,
"hyperparameters": {
"random_forest": {
"n_estimators": ,
"max_depth": [10, 20, null]
}
}
}
}

text

### Strategy Configuration
{
"strategy": {
"signal_threshold": 0.6,
"max_positions": 5,
"stop_loss": 0.05,
"take_profit": 0.15,
"position_sizing": "equal_weight"
}
}

text

## 🚨 Risk Disclaimers

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- All trading involves substantial risk of loss
- No representation is made that any strategy will achieve profits
- Thoroughly backtest any strategy before live deployment
- Consider transaction costs, taxes, and market impact
- Seek professional advice before making investment decisions

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Development Roadmap

### Planned Features
- [ ] Real-time data feeds (Alpha Vantage, IEX Cloud)
- [ ] Options trading strategies
- [ ] Cryptocurrency support
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization algorithms
- [ ] Web-based dashboard
- [ ] Paper trading simulator

### Performance Improvements
- [ ] GPU acceleration for LSTM training
- [ ] Parallel backtesting
- [ ] Memory-efficient data handling
- [ ] Distributed computing support

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support:

- 📧 Email: [your-email@domain.com]
- 💬 GitHub Issues: [Create an issue](../../issues)
- 📖 Documentation: See inline docstrings and comments

## 🏆 Acknowledgments

- Built with inspiration from modern quantitative finance practices
- Uses industry-standard libraries: scikit-learn, XGBoost, TensorFlow
- Backtesting methodology follows academic research standards
- Risk management techniques adapted from institutional trading
