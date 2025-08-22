"""
SABR Stochastic Volatility Model

This module implements the SABR (Stochastic Alpha, Beta, Rho) model for option pricing.
The model uses the lognormal approximation for European option pricing.

References:
- Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). 
  Managing smile risk. Wilmott Magazine, 1, 84-108.
"""

import numpy as np
from scipy.optimize import minimize, newton
from scipy.stats import norm
import warnings


class SABRModel:
    """
    SABR stochastic volatility model for European options pricing.
    """
    
    def __init__(self, F0, alpha, beta, rho, sigma0=None):
        """
        Initialize SABR model parameters.
        """
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigma0 = sigma0 if sigma0 is not None else alpha
        
        # Parameter validation
        if not 0 <= beta <= 1:
            raise ValueError("Beta must be between 0 and 1")
        if not -1 <= rho <= 1:
            raise ValueError("Rho must be between -1 and 1")
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
    
    def implied_volatility(self, K, T):
        """
        Calculate implied Black volatility using SABR formula with improved numerical stability.
        """
        F, alpha, beta, rho = self.F0, self.alpha, self.beta, self.rho
        
        # Handle scalar and array inputs
        K = np.atleast_1d(K)
        is_scalar = len(K) == 1
        
        # Initialize result array
        iv = np.zeros_like(K, dtype=float)
        
        for i, strike in enumerate(K):
            if T <= 0:
                iv[i] = 0.0
                continue
            
            # Handle ATM case
            if abs(strike - F) < 1e-8:
                if beta == 1:
                    iv[i] = alpha
                elif beta == 0:
                    iv[i] = alpha / F
                else:
                    iv[i] = alpha / (F**(1 - beta))
                continue
            
            try:
                # General SABR volatility calculation
                if beta == 0:
                    # Normal SABR (beta = 0)
                    iv[i] = self._sabr_normal_volatility(F, strike, T, alpha, rho)
                elif beta == 1:
                    # Lognormal SABR (beta = 1)
                    iv[i] = self._sabr_lognormal_volatility(F, strike, T, alpha, rho)
                else:
                    # General SABR
                    iv[i] = self._sabr_general_volatility(F, strike, T, alpha, beta, rho)
                
                # Ensure positive volatility
                iv[i] = max(iv[i], 1e-8)
                
            except Exception as e:
                warnings.warn(f"SABR volatility calculation failed for K={strike}: {str(e)}")
                iv[i] = 0.2  # Fallback volatility
        
        return iv[0] if is_scalar else iv
    
    def _sabr_normal_volatility(self, F, K, T, alpha, rho):
        """Calculate volatility for normal SABR (beta=0)."""
        log_FK = np.log(F / K)
        
        if abs(log_FK) < 1e-8:
            return alpha / F
        
        z = alpha * log_FK / np.sqrt(T) / F
        x = self._calculate_x(z, rho)
        
        vol = (alpha / F) * (log_FK / x) * (1 + self._time_correction_normal(T, alpha, F, rho))
        return vol
    
    def _sabr_lognormal_volatility(self, F, K, T, alpha, rho):
        """Calculate volatility for lognormal SABR (beta=1)."""
        log_FK = np.log(F / K)
        
        if abs(log_FK) < 1e-8:
            return alpha * (1 + self._time_correction_lognormal(T, alpha, rho))
        
        z = alpha * log_FK / np.sqrt(T)
        x = self._calculate_x(z, rho)
        
        vol = alpha * (log_FK / x) * (1 + self._time_correction_lognormal(T, alpha, rho))
        return vol
    
    def _sabr_general_volatility(self, F, K, T, alpha, beta, rho):
        """Calculate volatility for general SABR."""
        if abs(F - K) < 1e-8:
            return (alpha / (F**(1 - beta))) * (1 + self._time_correction_general(T, F, alpha, beta, rho))
        
        # Calculate intermediate values
        FK_avg = (F * K)**((1 - beta) / 2)
        log_FK = np.log(F / K)
        
        # Calculate z
        if beta == 0:
            z = alpha * (F - K) / np.sqrt(T)
        else:
            z = (alpha / ((1 - beta) * FK_avg)) * log_FK
        
        z = z / np.sqrt(T)
        x = self._calculate_x(z, rho)
        
        # Main SABR formula
        numerator = alpha
        denominator = FK_avg * (1 - beta) * x
        
        if abs(denominator) < 1e-12:
            return alpha / (F**(1 - beta))
        
        vol = (numerator / denominator) * (log_FK) * (1 + self._time_correction_general(T, F, alpha, beta, rho))
        return vol
    
    def _calculate_x(self, z, rho):
        """Calculate the x(z) function with numerical stability."""
        if abs(z) < 1e-7:
            # Taylor expansion for small z
            return 1 - rho * z / 2 + (rho**2 - 1) * z**2 / 12
        
        # Avoid numerical issues
        discriminant = 1 - 2 * rho * z + z**2
        if discriminant <= 0:
            return 1.0  # Fallback
        
        sqrt_discriminant = np.sqrt(discriminant)
        numerator = sqrt_discriminant + z - rho
        denominator = 1 - rho
        
        if abs(denominator) < 1e-12:
            return 1.0  # Fallback
        
        arg = numerator / denominator
        if arg <= 0:
            return 1.0  # Fallback
        
        return z / np.log(arg)
    
    def _time_correction_normal(self, T, alpha, F, rho):
        """Time-dependent correction for normal SABR."""
        term1 = rho * alpha / (4 * F)
        term2 = (2 - 3 * rho**2) * alpha**2 / (24 * F**2)
        return (term1 + term2) * T
    
    def _time_correction_lognormal(self, T, alpha, rho):
        """Time-dependent correction for lognormal SABR."""
        term1 = rho * alpha / 4
        term2 = (2 - 3 * rho**2) * alpha**2 / 24
        return (term1 + term2) * T
    
    def _time_correction_general(self, T, F, alpha, beta, rho):
        """Time-dependent correction for general SABR."""
        F_power = F**(1 - beta)
        
        term1 = (1 - beta)**2 / (24 * F_power**2)
        term2 = rho * alpha / (4 * F_power)
        term3 = (2 - 3 * rho**2) * alpha**2 / 24
        
        return (term1 + term2 + term3) * T
    
    def price(self, K, T, r, option_type='call'):
        """
        Price European option using SABR implied volatility and Black formula.
        """
        # Get SABR implied volatility
        sabr_vol = self.implied_volatility(K, T)
        
        # Use Black formula for forward prices
        return self._black_formula(self.F0, K, T, r, sabr_vol, option_type)
    
    def _black_formula(self, F, K, T, r, sigma, option_type):
        """Black formula for pricing forwards."""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(F - K, 0) * np.exp(-r * T)
            else:
                return max(K - F, 0) * np.exp(-r * T)
        
        if sigma <= 1e-8:
            if option_type.lower() == 'call':
                return max(F - K, 0) * np.exp(-r * T)
            else:
                return max(K - F, 0) * np.exp(-r * T)
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = F * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
        
        return max(price * np.exp(-r * T), 0)
    
    def monte_carlo_price(self, K, T, r, option_type='call', n_paths=100000, n_steps=252):
        """
        Price option using Monte Carlo simulation of SABR process.
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        F = np.full(n_paths, self.F0)
        sigma = np.full(n_paths, self.sigma0)
        
        # Simulation
        for step in range(n_steps):
            # Generate correlated random numbers
            Z1 = np.random.standard_normal(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)
            
            # Update volatility (log-normal dynamics)
            sigma = sigma * np.exp(self.alpha * Z2 * sqrt_dt - 0.5 * self.alpha**2 * dt)
            
            # Update forward price
            if self.beta == 0:
                # Normal dynamics
                F = F + sigma * Z1 * sqrt_dt
            elif abs(self.beta - 1.0) < 1e-8:
                # Lognormal dynamics
                F = F * np.exp(sigma * Z1 * sqrt_dt - 0.5 * sigma**2 * dt)
            else:
                # General CEV dynamics (Euler scheme)
                F_beta = np.power(np.maximum(F, 1e-8), self.beta)
                dF = sigma * F_beta * Z1 * sqrt_dt
                F = np.maximum(F + dF, 1e-8)  # Ensure positivity
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(F - K, 0)
        else:
            payoffs = np.maximum(K - F, 0)
        
        # Discount and calculate price
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'paths_used': n_paths
        }


def sabr_price(F0, K, T, r, alpha, beta, rho, option_type='call'):
    """
    Convenience function for SABR option pricing.
    """
    model = SABRModel(F0, alpha, beta, rho)
    return model.price(K, T, r, option_type)


def sabr_implied_vol(F0, K, T, alpha, beta, rho):
    """
    Calculate SABR implied volatility.
    """
    model = SABRModel(F0, alpha, beta, rho)
    return model.implied_volatility(K, T)
