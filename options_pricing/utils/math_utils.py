"""
Mathematical Utilities for Options Pricing

This module provides numerical methods and utilities commonly used in
quantitative finance and options pricing.
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d, griddata
from scipy.optimize import minimize_scalar, newton
import warnings


class MonteCarloSimulator:
    """
    Monte Carlo simulation framework with variance reduction techniques.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the Monte Carlo simulator.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
    
    def generate_correlated_normals(self, n_samples, correlation_matrix):
        """
        Generate correlated normal random variables using Cholesky decomposition.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        correlation_matrix : array-like
            Correlation matrix
            
        Returns:
        --------
        ndarray
            Array of correlated normal variables
        """
        dim = correlation_matrix.shape[0]
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Generate independent normals
        Z = self.rng.standard_normal((n_samples, dim))
        
        # Apply correlation
        return Z @ L.T
    
    def antithetic_sampling(self, random_samples):
        """
        Apply antithetic variance reduction.
        
        Parameters:
        -----------
        random_samples : ndarray
            Original random samples
            
        Returns:
        --------
        ndarray
            Combined original and antithetic samples
        """
        antithetic_samples = -random_samples
        return np.vstack([random_samples, antithetic_samples])
    
    def control_variate_adjustment(self, target_values, control_values, control_expectation):
        """
        Apply control variate variance reduction.
        
        Parameters:
        -----------
        target_values : array-like
            Values to be adjusted
        control_values : array-like
            Control variate values
        control_expectation : float
            Known expectation of control variate
            
        Returns:
        --------
        ndarray
            Adjusted values
        """
        target_values = np.array(target_values)
        control_values = np.array(control_values)
        
        # Calculate optimal coefficient
        covariance = np.cov(target_values, control_values)[0, 1]
        control_variance = np.var(control_values)
        
        if control_variance > 1e-10:
            beta = covariance / control_variance
            adjusted = target_values - beta * (control_values - control_expectation)
            return adjusted
        else:
            return target_values
    
    def importance_sampling_weights(self, samples, target_density, proposal_density):
        """
        Calculate importance sampling weights.
        
        Parameters:
        -----------
        samples : array-like
            Sample values
        target_density : callable
            Target probability density function
        proposal_density : callable
            Proposal probability density function
            
        Returns:
        --------
        ndarray
            Importance weights
        """
        weights = np.array([target_density(x) / proposal_density(x) for x in samples])
        return weights / np.sum(weights)


class NumericalIntegration:
    """
    Numerical integration methods for finance applications.
    """
    
    @staticmethod
    def adaptive_simpson(func, a, b, tolerance=1e-8, max_levels=15):
        """
        Adaptive Simpson's rule for numerical integration.
        
        Parameters:
        -----------
        func : callable
            Function to integrate
        a, b : float
            Integration bounds
        tolerance : float
            Convergence tolerance
        max_levels : int
            Maximum recursion levels
            
        Returns:
        --------
        float
            Integral approximation
        """
        def simpson_rule(f, x0, x2, h):
            x1 = x0 + h
            return h / 3 * (f(x0) + 4 * f(x1) + f(x2))
        
        def adaptive_simpson_recursive(f, a, b, epsilon, S, fa, fb, fc, level):
            if level >= max_levels:
                return S
            
            c = (a + b) / 2
            h = (b - a) / 4
            d = a + h
            e = b - h
            
            fd = f(d)
            fe = f(e)
            
            S1 = simpson_rule(f, a, c, h)
            S2 = simpson_rule(f, c, b, h)
            
            if abs(S1 + S2 - S) <= 15 * epsilon:
                return S1 + S2 + (S1 + S2 - S) / 15
            
            return (adaptive_simpson_recursive(f, a, c, epsilon/2, S1, fa, fc, fd, level+1) +
                   adaptive_simpson_recursive(f, c, b, epsilon/2, S2, fc, fb, fe, level+1))
        
        h = (b - a) / 2
        c = (a + b) / 2
        fa = func(a)
        fb = func(b)
        fc = func(c)
        S = simpson_rule(func, a, b, h)
        
        return adaptive_simpson_recursive(func, a, b, tolerance, S, fa, fb, fc, 0)
    
    @staticmethod
    def gauss_legendre_quadrature(func, a, b, n=20):
        """
        Gauss-Legendre quadrature for numerical integration.
        
        Parameters:
        -----------
        func : callable
            Function to integrate
        a, b : float
            Integration bounds
        n : int
            Number of quadrature points
            
        Returns:
        --------
        float
            Integral approximation
        """
        # Get Gauss-Legendre nodes and weights
        nodes, weights = np.polynomial.legendre.leggauss(n)
        
        # Transform from [-1, 1] to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        
        # Calculate integral
        integral = 0.5 * (b - a) * np.sum(weights * func(x))
        
        return integral
    
    @staticmethod
    def fourier_transform_pricing(characteristic_func, strike_range, n_points=2**12, alpha=1.5):
        """
        FFT-based option pricing using characteristic functions.
        
        Parameters:
        -----------
        characteristic_func : callable
            Characteristic function of log-asset price
        strike_range : tuple
            (min_strike, max_strike)
        n_points : int
            Number of FFT points
        alpha : float
            Damping parameter
            
        Returns:
        --------
        tuple
            (strikes, option_prices)
        """
        # FFT parameters
        eta = 0.25
        lambda_val = 2 * np.pi / (n_points * eta)
        
        # Create grids
        u = np.arange(0, n_points) * eta
        k = -lambda_val * n_points / 2 + lambda_val * np.arange(0, n_points)
        
        # Calculate integrand
        integrand = np.zeros(n_points, dtype=complex)
        
        for i, u_val in enumerate(u):
            if i == 0:
                u_val = 1e-8  # Avoid singularity
            
            psi = characteristic_func(u_val - (alpha + 1) * 1j) / ((alpha + u_val * 1j) * (alpha + 1 + u_val * 1j))
            integrand[i] = psi
        
        # Apply dampening
        integrand *= np.exp(-alpha * k)
        
        # FFT
        fft_result = np.fft.fft(integrand)
        option_values = np.real(fft_result) / np.pi
        
        # Convert log-strikes to strikes
        strikes = np.exp(k)
        
        # Filter for desired strike range
        mask = (strikes >= strike_range[0]) & (strikes <= strike_range[asset:1])
        
        return strikes[mask], option_values[mask]


class VolatilitySurface:
    """
    Volatility surface interpolation and manipulation.
    """
    
    def __init__(self, strikes, maturities, implied_vols):
        """
        Initialize volatility surface.
        
        Parameters:
        -----------
        strikes : array-like
            Strike prices
        maturities : array-like
            Time to maturities
        implied_vols : array-like
            Implied volatilities
        """
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.implied_vols = np.array(implied_vols)
        
        # Create interpolation function
        points = np.column_stack([self.strikes, self.maturities])
        self.interpolator = interp1d(
            points.T, self.implied_vols, 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
    
    def get_volatility(self, strike, maturity):
        """
        Get interpolated volatility for given strike and maturity.
        
        Parameters:
        -----------
        strike : float or array-like
            Strike price(s)
        maturity : float or array-like
            Time to maturity
            
        Returns:
        --------
        float or array
            Interpolated implied volatility
        """
        if np.isscalar(strike) and np.isscalar(maturity):
            points = np.array([[strike], [maturity]])
            return float(self.interpolator(points))
        else:
            strike = np.atleast_1d(strike)
            maturity = np.atleast_1d(maturity)
            points = np.column_stack([strike.ravel(), maturity.ravel()])
            return griddata(
                np.column_stack([self.strikes, self.maturities]),
                self.implied_vols,
                points,
                method='linear',
                fill_value=np.nan
            ).reshape(strike.shape)
    
    def fit_svi_parameterization(self):
        """
        Fit SVI (Stochastic Volatility Inspired) parameterization to surface.
        
        SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        where w = total implied variance, k = log-moneyness
        
        Returns:
        --------
        dict
            SVI parameters for each maturity slice
        """
        unique_maturities = np.unique(self.maturities)
        svi_params = {}
        
        for T in unique_maturities:
            # Get volatilities for this maturity
            mask = self.maturities == T
            strikes_T = self.strikes[mask]
            vols_T = self.implied_vols[mask]
            
            # Convert to log-moneyness and total variance
            # Assuming ATM forward is approximately current spot
            F = np.median(strikes_T)  # Rough approximation
            k = np.log(strikes_T / F)
            w = vols_T**2 * T
            
            # SVI objective function
            def svi_objective(params):
                a, b, rho, m, sigma = params
                w_svi = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
                return np.sum((w - w_svi)**2)
            
            # Initial guess and bounds
            initial_guess = [
                np.mean(w),  # a
                0.1,         # b
                0.0,         # rho
                0.0,         # m
                0.1          # sigma
            ]
            
            bounds = [
                (0, None),      # a >= 0
                (0, None),      # b >= 0
                (-1, 1),        # -1 <= rho <= 1
                (None, None),   # m unbounded
                (0, None)       # sigma >= 0
            ]
            
            try:
                result = minimize(svi_objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    a, b, rho, m, sigma = result.x
                    svi_params[T] = {
                        'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma,
                        'rmse': np.sqrt(result.fun / len(k))
                    }
                else:
                    warnings.warn(f"SVI fitting failed for maturity {T}")
                    
            except Exception as e:
                warnings.warn(f"SVI fitting error for maturity {T}: {str(e)}")
        
        return svi_params


class RootFinding:
    """
    Root finding algorithms for implied volatility and other applications.
    """
    
    @staticmethod
    def newton_raphson(func, derivative, x0, tolerance=1e-8, max_iterations=100):
        """
        Newton-Raphson root finding method.
        
        Parameters:
        -----------
        func : callable
            Function to find root of
        derivative : callable
            Derivative of the function
        x0 : float
            Initial guess
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum iterations
            
        Returns:
        --------
        float
            Root of the function
        """
        x = x0
        
        for i in range(max_iterations):
            fx = func(x)
            
            if abs(fx) < tolerance:
                return x
            
            fpx = derivative(x)
            
            if abs(fpx) < 1e-12:
                raise ValueError("Derivative too small - Newton-Raphson failed")
            
            x_new = x - fx / fpx
            
            if abs(x_new - x) < tolerance:
                return x_new
            
            x = x_new
        
        warnings.warn(f"Newton-Raphson did not converge after {max_iterations} iterations")
        return x
    
    @staticmethod
    def brent_method(func, a, b, tolerance=1e-8, max_iterations=100):
        """
        Brent's method for root finding (bracketing method).
        
        Parameters:
        -----------
        func : callable
            Function to find root of
        a, b : float
            Bracketing interval [a, b]
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum iterations
            
        Returns:
        --------
        float
            Root of the function
        """
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            raise ValueError("Function values at endpoints must have opposite signs")
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c = a
        fc = fa
        mflag = True
        
        for i in range(max_iterations):
            if abs(b - a) < tolerance:
                return b
            
            if fa != fc and fb != fc:
                # Inverse quadratic interpolation
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                    (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)
            
            # Check conditions for using bisection
            condition1 = not ((3 * a + b) / 4 <= s <= b)
            condition2 = mflag and abs(s - b) >= abs(b - c) / 2
            condition3 = not mflag and abs(s - b) >= abs(c - a) / 2
            condition4 = mflag and abs(b - c) < tolerance
            condition5 = not mflag and abs(c - a) < tolerance
            
            if condition1 or condition2 or condition3 or condition4 or condition5:
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False
            
            fs = func(s)
            a, c = b, b
            fa, fc = fb, fb
            
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs
            
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
        
        warnings.warn(f"Brent method did not converge after {max_iterations} iterations")
        return b


# Utility functions for common calculations
def black_scholes_vega(S, K, T, r, sigma):
    """Calculate Black-Scholes vega for implied volatility calculations."""
    from scipy.stats import norm
    
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility_newton(price_func, market_price, vega_func, initial_guess=0.2, 
                             tolerance=1e-6, max_iterations=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    -----------
    price_func : callable
        Function that takes volatility and returns option price
    market_price : float
        Market price of the option
    vega_func : callable
        Function that takes volatility and returns vega
    initial_guess : float
        Initial volatility guess
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations
        
    Returns:
    --------
    float
        Implied volatility
    """
    sigma = initial_guess
    
    for i in range(max_iterations):
        price = price_func(sigma)
        vega = vega_func(sigma)
        
        price_diff = price - market_price
        
        if abs(price_diff) < tolerance:
            return sigma
        
        if vega == 0:
            raise ValueError("Vega is zero - cannot compute implied volatility")
        
        sigma_new = sigma - price_diff / vega
        
        if sigma_new <= 0:
            sigma_new = sigma / 2
        
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    warnings.warn(f"Implied volatility calculation did not converge after {max_iterations} iterations")
    return sigma
