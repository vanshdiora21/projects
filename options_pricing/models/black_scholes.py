"""
Black-Scholes Options Pricing Model

This module implements the classic Black-Scholes formula for European options pricing.
The model assumes constant volatility and interest rates.

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. 
  Journal of Political Economy, 81(3), 637-654.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import warnings


class BlackScholesModel:
    """
    Black-Scholes options pricing model for European options.
    
    The model assumes:
    - Constant risk-free rate
    - Constant volatility
    - No dividends
    - European exercise
    """
    
    def __init__(self, S0, K, T, r, sigma, option_type='call'):
        """
        Initialize Black-Scholes model parameters.
        
        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of underlying asset
        option_type : str
            'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def _d1(self):
        """Calculate d1 parameter."""
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self):
        """Calculate d2 parameter."""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        """
        Calculate option price using Black-Scholes formula.
        
        Returns:
        --------
        float
            Option price
        """
        if self.T <= 0:
            if self.option_type == 'call':
                return max(self.S0 - self.K, 0)
            else:
                return max(self.K - self.S0, 0)
        
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        
        return price
    
    def delta(self):
        """
        Calculate Delta (sensitivity to underlying price).
        
        Returns:
        --------
        float
            Delta value
        """
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S0 > self.K else 0.0
            else:
                return -1.0 if self.S0 < self.K else 0.0
        
        d1 = self._d1()
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self):
        """
        Calculate Gamma (sensitivity of Delta to underlying price).
        
        Returns:
        --------
        float
            Gamma value
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
    
    def theta(self):
        """
        Calculate Theta (sensitivity to time decay).
        
        Returns:
        --------
        float
            Theta value (per day)
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            theta = (-self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            theta = (-self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return theta / 365  # Convert to per day
    
    def vega(self):
        """
        Calculate Vega (sensitivity to volatility).
        
        Returns:
        --------
        float
            Vega value
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T) / 100  # Per 1% vol change
    
    def rho(self):
        """
        Calculate Rho (sensitivity to interest rate).
        
        Returns:
        --------
        float
            Rho value
        """
        if self.T <= 0:
            return 0.0
        
        d2 = self._d2()
        
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    def implied_volatility(self, market_price, max_iterations=100, tolerance=1e-6):
        """
        Calculate implied volatility from market price using Newton-Raphson method.
        
        Parameters:
        -----------
        market_price : float
            Market price of the option
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        float
            Implied volatility
        """
        def objective(vol):
            self.sigma = vol
            return self.price() - market_price
        
        def vega_derivative(vol):
            self.sigma = vol
            return self.vega() * 100  # Convert back from percentage
        
        try:
            # Initial guess
            sigma_guess = 0.2
            
            for i in range(max_iterations):
                self.sigma = sigma_guess
                price_diff = self.price() - market_price
                vega_val = self.vega() * 100
                
                if abs(price_diff) < tolerance:
                    return sigma_guess
                
                if vega_val == 0:
                    raise ValueError("Vega is zero - cannot compute implied volatility")
                
                sigma_guess = sigma_guess - price_diff / vega_val
                
                if sigma_guess <= 0:
                    sigma_guess = 0.01
            
            warnings.warn(f"Implied volatility did not converge after {max_iterations} iterations")
            return sigma_guess
            
        except Exception as e:
            raise ValueError(f"Failed to calculate implied volatility: {str(e)}")


def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """
    Convenience function for Black-Scholes pricing.
    
    Parameters:
    -----------
    S0 : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    model = BlackScholesModel(S0, K, T, r, sigma, option_type)
    return model.price()


def black_scholes_greeks(S0, K, T, r, sigma, option_type='call'):
    """
    Calculate all Greeks for a Black-Scholes option.
    
    Returns:
    --------
    dict
        Dictionary containing all Greeks
    """
    model = BlackScholesModel(S0, K, T, r, sigma, option_type)
    
    return {
        'price': model.price(),
        'delta': model.delta(),
        'gamma': model.gamma(),
        'theta': model.theta(),
        'vega': model.vega(),
        'rho': model.rho()
    }
