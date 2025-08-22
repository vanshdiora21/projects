"""
Options Pricing Models: Production-Ready Quantitative Finance Framework

A comprehensive Python framework for options pricing with stochastic volatility models.
Built for institutional-grade quantitative finance applications.

Version: 1.0.0
Author: Quantitative Finance Team
License: MIT

Main Components:
- Black-Scholes Model: Classic constant volatility pricing
- Heston Model: Stochastic volatility with FFT and Monte Carlo
- SABR Model: Stochastic Alpha, Beta, Rho for interest rates
- Calibration Framework: Robust parameter estimation
- Risk Management Tools: Greeks and scenario analysis
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"
__email__ = "options-pricing@example.com"
__license__ = "MIT"

# Core model imports
from .models import (
    BlackScholesModel, 
    HestonModel, 
    SABRModel,
    black_scholes_price,
    black_scholes_greeks,
    heston_price,
    sabr_price,
    sabr_implied_vol
)

# Utility imports
from .utils import (
    MonteCarloSimulator,
    NumericalIntegration,
    VolatilitySurface,
    RootFinding
)

# Calibration imports
from .calibration import (
    HestonCalibrator,
    SABRCalibrator,
    generate_synthetic_market_data
)

__all__ = [
    # Models
    'BlackScholesModel', 'HestonModel', 'SABRModel',
    
    # Convenience functions
    'black_scholes_price', 'black_scholes_greeks',
    'heston_price', 'sabr_price', 'sabr_implied_vol',
    
    # Utilities
    'MonteCarloSimulator', 'NumericalIntegration', 
    'VolatilitySurface', 'RootFinding',
    
    # Calibration
    'HestonCalibrator', 'SABRCalibrator', 
    'generate_synthetic_market_data'
]

def get_version():
    """Return the current version."""
    return __version__

def print_info():
    """Print framework information."""
    print(f"Options Pricing Framework v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print()
    print("Available models:")
    print("  • Black-Scholes: Classic constant volatility model")
    print("  • Heston: Stochastic volatility with mean reversion")
    print("  • SABR: Stochastic alpha, beta, rho model")
    print()
    print("Key features:")
    print("  • Multiple pricing methods (analytical, FFT, Monte Carlo)")
    print("  • Model calibration to market data")
    print("  • Complete Greeks calculation")
    print("  • Variance reduction techniques")
    print("  • Command line interface")
    print("  • Comprehensive test suite")
