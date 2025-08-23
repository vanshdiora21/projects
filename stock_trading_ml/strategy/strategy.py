"""
Trading strategy implementation for ML-driven stock trading.

This module implements:
- Signal generation from ML model predictions
- Position sizing and risk management
- Portfolio construction rules
- Transaction cost modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class Signal(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Position:
    """Represents a trading position."""
    ticker: str
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    signal_confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl: float
    pnl_pct: float
    commission: float
    signal_confidence: float
    trade_type: str  # 'LONG' or 'SHORT'


class MLTradingStrategy:
    """
    Machine Learning driven trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trading strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        self.config = config
        self.strategy_config = config['strategy']
        self.backtest_config = config['backtest']
        
        # Strategy parameters
        self.signal_threshold = self.strategy_config['signal_threshold']
        self.max_positions = self.strategy_config['max_positions']
        self.transaction_costs = self.strategy_config['transaction_costs']
        self.stop_loss = self.strategy_config['stop_loss']
        self.take_profit = self.strategy_config['take_profit']
        
        # Risk management parameters
        self.risk_params = self.strategy_config['risk_management']
        self.max_portfolio_risk = self.risk_params['max_portfolio_risk']
        self.max_single_position = self.risk_params['max_single_position']
        self.volatility_target = self.risk_params['volatility_target']
        
        # Portfolio state
        self.positions = {}  # ticker -> Position
        self.cash = self.backtest_config['initial_capital']
        self.portfolio_value = self.backtest_config['initial_capital']
        self.trades = []
        
        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.drawdown_history = []
        
    def generate_signals(self, predictions: Dict[str, np.ndarray], 
                        probabilities: Dict[str, np.ndarray],
                        prices: Dict[str, pd.DataFrame],
                        current_date: pd.Timestamp) -> Dict[str, Tuple[Signal, float]]:
        """
        Generate trading signals from ML model predictions.
        
        Args:
            predictions: Dictionary mapping tickers to prediction arrays
            probabilities: Dictionary mapping tickers to probability arrays
            prices: Dictionary mapping tickers to price DataFrames
            current_date: Current trading date
            
        Returns:
            Dictionary mapping tickers to (signal, confidence) tuples
        """
        signals = {}
        
        for ticker in predictions.keys():
            if ticker not in probabilities or ticker not in prices:
                continue
            
            # Get current prediction and probability
            pred = predictions[ticker]
            prob = probabilities[ticker]
            
            # Handle case where we have arrays
            if isinstance(pred, np.ndarray) and len(pred) > 0:
                current_pred = pred[-1]
                current_prob = prob[-1]
            else:
                current_pred = pred
                current_prob = prob
            
            # Generate signal based on probability threshold
            if current_prob > self.signal_threshold:
                signal = Signal.BUY
                confidence = current_prob
            elif current_prob < (1 - self.signal_threshold):
                signal = Signal.SELL  
                confidence = 1 - current_prob
            else:
                signal = Signal.HOLD
                confidence = 0.5
            
            # Additional filters
            signal = self._apply_signal_filters(ticker, signal, confidence, prices[ticker], current_date)
            
            signals[ticker] = (signal, confidence)
        
        return signals
    
    def calculate_position_sizes(self, signals: Dict[str, Tuple[Signal, float]], 
                               prices: Dict[str, pd.DataFrame],
                               current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate position sizes based on signals and risk management rules.
        
        Args:
            signals: Dictionary of trading signals
            prices: Dictionary of price data
            current_date: Current trading date
            
        Returns:
            Dictionary mapping tickers to position sizes
        """
        position_sizes = {}
        
        # Count number of signals
        buy_signals = [(ticker, conf) for ticker, (sig, conf) in signals.items() if sig == Signal.BUY]
        sell_signals = [(ticker, conf) for ticker, (sig, conf) in signals.items() if sig == Signal.SELL]
        
        # Calculate available capital for new positions
        available_capital = self._calculate_available_capital()
        
        # Position sizing based on configuration
        sizing_method = self.strategy_config['position_sizing']
        
        if sizing_method == 'equal_weight':
            position_sizes = self._equal_weight_sizing(buy_signals, sell_signals, available_capital, prices, current_date)
        elif sizing_method == 'volatility_adjusted':
            position_sizes = self._volatility_adjusted_sizing(buy_signals, sell_signals, available_capital, prices, current_date)
        elif sizing_method == 'confidence_weighted':
            position_sizes = self._confidence_weighted_sizing(buy_signals, sell_signals, available_capital, prices, current_date)
        else:
            # Default to equal weight
            position_sizes = self._equal_weight_sizing(buy_signals, sell_signals, available_capital, prices, current_date)
        
        # Apply position size limits
        position_sizes = self._apply_position_limits(position_sizes, prices, current_date)
        
        return position_sizes
    
    def execute_trades(self, signals: Dict[str, Tuple[Signal, float]],
                      position_sizes: Dict[str, float],
                      prices: Dict[str, pd.DataFrame],
                      current_date: pd.Timestamp) -> List[Trade]:
        """
        Execute trades based on signals and position sizes.
        
        Args:
            signals: Trading signals
            position_sizes: Calculated position sizes
            prices: Price data
            current_date: Current trading date
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for ticker, (signal, confidence) in signals.items():
            if ticker not in prices:
                continue
            
            current_price = self._get_current_price(prices[ticker], current_date)
            if current_price is None:
                continue
            
            if signal == Signal.BUY:
                trade = self._execute_buy_order(ticker, position_sizes.get(ticker, 0), 
                                              current_price, current_date, confidence)
                if trade:
                    executed_trades.append(trade)
                    
            elif signal == Signal.SELL:
                trade = self._execute_sell_order(ticker, position_sizes.get(ticker, 0),
                                               current_price, current_date, confidence)
                if trade:
                    executed_trades.append(trade)
            
            # Check for stop loss / take profit triggers
            if ticker in self.positions:
                sl_tp_trade = self._check_stop_loss_take_profit(ticker, current_price, current_date)
                if sl_tp_trade:
                    executed_trades.append(sl_tp_trade)
        
        return executed_trades
    
    def update_portfolio(self, current_date: pd.Timestamp, prices: Dict[str, pd.DataFrame]) -> None:
        """
        Update portfolio value and performance metrics.
        
        Args:
            current_date: Current date
            prices: Current price data
        """
        # Calculate current portfolio value
        portfolio_value = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in prices:
                current_price = self._get_current_price(prices[ticker], current_date)
                if current_price:
                    position_value = position.quantity * current_price
                    portfolio_value += position_value
        
        # Update portfolio metrics
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.returns_history.append(daily_return)
            
            # Calculate drawdown
            peak_value = max([record['portfolio_value'] for record in self.portfolio_history])
            current_drawdown = (peak_value - portfolio_value) / peak_value
            self.drawdown_history.append(current_drawdown)
        else:
            self.returns_history.append(0.0)
            self.drawdown_history.append(0.0)
        
        # Record portfolio state
        self.portfolio_value = portfolio_value
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'daily_return': self.returns_history[-1],
            'drawdown': self.drawdown_history[-1]
        })
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Dictionary with portfolio statistics
        """
        if not self.portfolio_history:
            return {}
        
        returns = np.array(self.returns_history)
        
        summary = {
            'total_value': self.portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'total_return': (self.portfolio_value - self.backtest_config['initial_capital']) / self.backtest_config['initial_capital'],
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0.0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        }
        
        return summary
    
    def _apply_signal_filters(self, ticker: str, signal: Signal, confidence: float,
                             price_data: pd.DataFrame, current_date: pd.Timestamp) -> Signal:
        """
        Apply additional filters to trading signals.
        
        Args:
            ticker: Stock ticker
            signal: Raw signal from ML model
            confidence: Signal confidence
            price_data: Price data for the ticker
            current_date: Current date
            
        Returns:
            Filtered signal
        """
        # Filter 1: Don't trade if we already have max positions
        if signal == Signal.BUY and len(self.positions) >= self.max_positions:
            return Signal.HOLD
        
        # Filter 2: Don't buy if we already have a position in this ticker
        if signal == Signal.BUY and ticker in self.positions:
            return Signal.HOLD
        
        # Filter 3: Don't sell if we don't have a position
        if signal == Signal.SELL and ticker not in self.positions:
            return Signal.HOLD
        
        # Filter 4: Minimum confidence threshold
        if confidence < 0.6:  # Require minimum confidence
            return Signal.HOLD
        
        # Filter 5: Liquidity check (placeholder)
        # In real implementation, check average volume, bid-ask spread, etc.
        
        return signal
    
    def _calculate_available_capital(self) -> float:
        """
        Calculate available capital for new positions.
        
        Returns:
            Available capital amount
        """
        # Reserve some cash for transaction costs and margin
        reserved_cash = self.portfolio_value * 0.05  # Reserve 5%
        available = max(0, self.cash - reserved_cash)
        
        return available
    
    def _equal_weight_sizing(self, buy_signals: List[Tuple[str, float]], 
                            sell_signals: List[Tuple[str, float]],
                            available_capital: float,
                            prices: Dict[str, pd.DataFrame],
                            current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Equal weight position sizing.
        
        Args:
            buy_signals: List of buy signals with confidence
            sell_signals: List of sell signals with confidence  
            available_capital: Available capital
            prices: Price data
            current_date: Current date
            
        Returns:
            Dictionary of position sizes
        """
        position_sizes = {}
        
        if buy_signals:
            # Equal weight among buy signals
            capital_per_position = available_capital / len(buy_signals)
            
            for ticker, confidence in buy_signals:
                current_price = self._get_current_price(prices[ticker], current_date)
                if current_price:
                    shares = capital_per_position / current_price
                    position_sizes[ticker] = shares
        
        # For sell signals, sell entire position
        for ticker, confidence in sell_signals:
            if ticker in self.positions:
                position_sizes[ticker] = -self.positions[ticker].quantity
        
        return position_sizes
    
    def _volatility_adjusted_sizing(self, buy_signals: List[Tuple[str, float]], 
                                   sell_signals: List[Tuple[str, float]],
                                   available_capital: float,
                                   prices: Dict[str, pd.DataFrame],
                                   current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Volatility-adjusted position sizing.
        
        Args:
            buy_signals: List of buy signals
            sell_signals: List of sell signals
            available_capital: Available capital
            prices: Price data
            current_date: Current date
            
        Returns:
            Dictionary of position sizes
        """
        position_sizes = {}
        
        if buy_signals:
            # Calculate volatility for each ticker
            volatilities = {}
            for ticker, confidence in buy_signals:
                if ticker in prices:
                    returns = prices[ticker]['close'].pct_change().dropna()
                    if len(returns) > 20:
                        vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
                        volatilities[ticker] = vol if vol > 0 else 0.1  # Minimum volatility
            
            # Inverse volatility weighting
            if volatilities:
                inv_vols = {ticker: 1/vol if vol > 0 else 0 for ticker, vol in volatilities.items()}
                total_inv_vol = sum(inv_vols.values())
                
                for ticker, confidence in buy_signals:
                    if ticker in inv_vols and total_inv_vol > 0:
                        weight = inv_vols[ticker] / total_inv_vol
                        capital_allocation = available_capital * weight
                        current_price = self._get_current_price(prices[ticker], current_date)
                        if current_price:
                            shares = capital_allocation / current_price
                            position_sizes[ticker] = shares
        
        # Handle sell signals
        for ticker, confidence in sell_signals:
            if ticker in self.positions:
                position_sizes[ticker] = -self.positions[ticker].quantity
        
        return position_sizes
    
    def _confidence_weighted_sizing(self, buy_signals: List[Tuple[str, float]], 
                                   sell_signals: List[Tuple[str, float]],
                                   available_capital: float,
                                   prices: Dict[str, pd.DataFrame],
                                   current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Confidence-weighted position sizing.
        
        Args:
            buy_signals: List of buy signals with confidence
            sell_signals: List of sell signals with confidence
            available_capital: Available capital
            prices: Price data
            current_date: Current date
            
        Returns:
            Dictionary of position sizes
        """
        position_sizes = {}
        
        if buy_signals:
            # Weight by confidence
            total_confidence = sum([conf for _, conf in buy_signals])
            
            for ticker, confidence in buy_signals:
                weight = confidence / total_confidence if total_confidence > 0 else 1/len(buy_signals)
                capital_allocation = available_capital * weight
                current_price = self._get_current_price(prices[ticker], current_date)
                if current_price:
                    shares = capital_allocation / current_price
                    position_sizes[ticker] = shares
        
        # Handle sell signals
        for ticker, confidence in sell_signals:
            if ticker in self.positions:
                position_sizes[ticker] = -self.positions[ticker].quantity
        
        return position_sizes
    
    def _apply_position_limits(self, position_sizes: Dict[str, float],
                              prices: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Apply position size limits based on risk management rules.
        
        Args:
            position_sizes: Raw position sizes
            prices: Price data
            current_date: Current date
            
        Returns:
            Limited position sizes
        """
        limited_sizes = {}
        
        for ticker, size in position_sizes.items():
            if ticker not in prices:
                continue
            
            current_price = self._get_current_price(prices[ticker], current_date)
            if not current_price:
                continue
            
            # Calculate position value
            position_value = abs(size) * current_price
            
            # Apply maximum single position limit
            max_position_value = self.portfolio_value * self.max_single_position
            if position_value > max_position_value:
                # Scale down the position
                scale_factor = max_position_value / position_value
                size = size * scale_factor
            
            limited_sizes[ticker] = size
        
        return limited_sizes
    
    def _execute_buy_order(self, ticker: str, quantity: float, price: float,
                          date: pd.Timestamp, confidence: float) -> Optional[Trade]:
        """
        Execute a buy order.
        
        Args:
            ticker: Stock ticker
            quantity: Number of shares to buy
            price: Execution price
            date: Trade date
            confidence: Signal confidence
            
        Returns:
            Trade object if executed, None otherwise
        """
        if quantity <= 0:
            return None
        
        # Calculate total cost including commission
        gross_cost = quantity * price
        commission = gross_cost * self.transaction_costs
        total_cost = gross_cost + commission
        
        if total_cost > self.cash:
            # Not enough cash - reduce quantity
            available_cash = self.cash * 0.99  # Leave small buffer
            max_quantity = available_cash / (price * (1 + self.transaction_costs))
            if max_quantity < 1:  # Can't buy even 1 share
                return None
            quantity = max_quantity
            gross_cost = quantity * price  
            commission = gross_cost * self.transaction_costs
            total_cost = gross_cost + commission
        
        # Execute trade
        self.cash -= total_cost
        
        # Create position
        position = Position(
            ticker=ticker,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            signal_confidence=confidence,
            stop_loss=price * (1 - self.stop_loss),
            take_profit=price * (1 + self.take_profit)
        )
        
        self.positions[ticker] = position
        
        # Create trade record (for buy, no PnL yet)
        trade = Trade(
            ticker=ticker,
            quantity=quantity,
            entry_price=price,
            exit_price=0.0,  # Will be filled when position is closed
            entry_date=date,
            exit_date=pd.NaT,
            pnl=0.0,
            pnl_pct=0.0,
            commission=commission,
            signal_confidence=confidence,
            trade_type='LONG'
        )
        
        return trade
    
    def _execute_sell_order(self, ticker: str, quantity: float, price: float,
                           date: pd.Timestamp, confidence: float) -> Optional[Trade]:
        """
        Execute a sell order (close position).
        
        Args:
            ticker: Stock ticker
            quantity: Number of shares to sell (should be negative for full position close)
            price: Execution price
            date: Trade date
            confidence: Signal confidence
            
        Returns:
            Trade object if executed, None otherwise
        """
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        
        # For simplicity, sell entire position
        sell_quantity = position.quantity
        
        # Calculate proceeds
        gross_proceeds = sell_quantity * price
        commission = gross_proceeds * self.transaction_costs
        net_proceeds = gross_proceeds - commission
        
        # Calculate PnL
        total_cost = position.quantity * position.entry_price
        entry_commission = total_cost * self.transaction_costs
        pnl = net_proceeds - total_cost - entry_commission
        pnl_pct = pnl / (total_cost + entry_commission) if (total_cost + entry_commission) > 0 else 0.0
        
        # Update cash
        self.cash += net_proceeds
        
        # Remove position
        del self.positions[ticker]
        
        # Create completed trade record
        trade = Trade(
            ticker=ticker,
            quantity=sell_quantity,
            entry_price=position.entry_price,
            exit_price=price,
            entry_date=position.entry_date,
            exit_date=date,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=entry_commission + commission,  # Entry + exit commission
            signal_confidence=confidence,
            trade_type='LONG'
        )
        
        self.trades.append(trade)
        
        return trade
    
    def _check_stop_loss_take_profit(self, ticker: str, current_price: float,
                                    current_date: pd.Timestamp) -> Optional[Trade]:
        """
        Check if stop loss or take profit should be triggered.
        
        Args:
            ticker: Stock ticker
            current_price: Current price
            current_date: Current date
            
        Returns:
            Trade object if stop loss/take profit triggered, None otherwise
        """
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        
        # Check stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            return self._execute_sell_order(ticker, -position.quantity, current_price, 
                                          current_date, 0.0)  # 0 confidence for forced exit
        
        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            return self._execute_sell_order(ticker, -position.quantity, current_price,
                                          current_date, 1.0)  # 1.0 confidence for profitable exit
        
        return None
    
    def _get_current_price(self, price_df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        """
        Get current price for a given date.
        
        Args:
            price_df: Price DataFrame
            date: Target date
            
        Returns:
            Current price or None if not available
        """
        # Find the price for the given date or the closest previous date
        try:
            if 'date' in price_df.columns:
                price_df_indexed = price_df.set_index('date')
            else:
                price_df_indexed = price_df
            
            # Get the last available price on or before the date
            available_dates = price_df_indexed.index[price_df_indexed.index <= date]
            if len(available_dates) > 0:
                latest_date = available_dates.max()
                return float(price_df_indexed.loc[latest_date, 'close'])
        except Exception as e:
            print(f"Error getting price for {date}: {e}")
        
        return None
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades."""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0


if __name__ == "__main__":
    print("Trading Strategy module loaded successfully")
    print("Features:")
    print("- ML signal generation")
    print("- Multiple position sizing methods")
    print("- Risk management")
    print("- Stop loss / take profit")
    print("- Portfolio tracking")
