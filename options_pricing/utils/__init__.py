"""
Utilities Package for Options Pricing

This package contains mathematical utilities and numerical methods
used throughout the options pricing framework.
"""

from .math_utils import (
    MonteCarloSimulator,
    NumericalIntegration,
    VolatilitySurface,
    RootFinding,
    black_scholes_vega,
    implied_volatility_newton
)

__all__ = [
    'MonteCarloSimulator',
    'NumericalIntegration', 
    'VolatilitySurface',
    'RootFinding',
    'black_scholes_vega',
    'implied_volatility_newton'
]
