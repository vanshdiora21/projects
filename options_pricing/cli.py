#!/usr/bin/env python3
"""
Command Line Interface for Options Pricing Models

This CLI provides access to all major functionality of the options pricing framework
including pricing, calibration, and analysis tools.
"""

import click
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.black_scholes import black_scholes_price, black_scholes_greeks
from models.heston import heston_price, HestonModel
from models.sabr import sabr_price, SABRModel
from calibration.calibrator import HestonCalibrator, SABRCalibrator, generate_synthetic_market_data


def load_config():
    """Load configuration from config.json."""
    config_path = project_root / "config" / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        click.echo("Warning: config.json not found, using defaults")
        return {}


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Options Pricing CLI - Advanced quantitative finance toolkit."""
    pass


@cli.command()
@click.option('--model', type=click.Choice(['bs', 'heston', 'sabr']), required=True, 
              help='Pricing model to use')
@click.option('--spot', type=float, required=True, help='Current spot price')
@click.option('--strike', type=float, required=True, help='Strike price')
@click.option('--maturity', type=float, required=True, help='Time to maturity (years)')
@click.option('--rate', type=float, default=0.05, help='Risk-free rate')
@click.option('--option-type', type=click.Choice(['call', 'put']), default='call',
              help='Option type')
@click.option('--volatility', type=float, help='Volatility (for Black-Scholes)')
@click.option('--kappa', type=float, help='Heston: mean reversion rate')
@click.option('--theta', type=float, help='Heston: long-term variance')
@click.option('--sigma-v', type=float, help='Heston: vol of vol')
@click.option('--rho', type=float, help='Heston/SABR: correlation parameter')
@click.option('--v0', type=float, help='Heston: initial variance')
@click.option('--alpha', type=float, help='SABR: vol of vol')
@click.option('--beta', type=float, help='SABR: CEV parameter')
@click.option('--method', type=click.Choice(['analytical', 'fft', 'monte_carlo']), 
              default='analytical', help='Pricing method')
@click.option('--output', type=click.Path(), help='Output file path')
def price(model, spot, strike, maturity, rate, option_type, volatility, kappa, theta, 
          sigma_v, rho, v0, alpha, beta, method, output):
    """Price an option using the specified model."""
    
    config = load_config()
    
    try:
        if model == 'bs':
            if volatility is None:
                volatility = config.get('models', {}).get('black_scholes', {}).get(
                    'default_params', {}).get('sigma', 0.2)
            
            price_result = black_scholes_price(spot, strike, maturity, rate, volatility, option_type)
            greeks = black_scholes_greeks(spot, strike, maturity, rate, volatility, option_type)
            
            result = {
                'model': 'Black-Scholes',
                'price': price_result,
                'greeks': greeks,
                'parameters': {
                    'spot': spot, 'strike': strike, 'maturity': maturity,
                    'rate': rate, 'volatility': volatility, 'option_type': option_type
                }
            }
            
        elif model == 'heston':
            # Use defaults from config if not provided
            defaults = config.get('models', {}).get('heston', {}).get('default_params', {})
            kappa = kappa or defaults.get('kappa', 2.0)
            theta = theta or defaults.get('theta', 0.04)
            sigma_v = sigma_v or defaults.get('sigma_v', 0.3)
            rho = rho or defaults.get('rho', -0.5)
            v0 = v0 or defaults.get('V0', 0.04)
            
            if method == 'monte_carlo':
                heston_model = HestonModel(spot, v0, rate, kappa, theta, sigma_v, rho)
                mc_result = heston_model.price_monte_carlo(strike, maturity, option_type)
                price_result = mc_result['price']
                std_error = mc_result['std_error']
            else:
                price_result = heston_price(spot, v0, strike, maturity, rate, kappa, 
                                          theta, sigma_v, rho, option_type, 'fft')
                std_error = None
            
            result = {
                'model': 'Heston',
                'price': price_result,
                'std_error': std_error,
                'parameters': {
                    'spot': spot, 'strike': strike, 'maturity': maturity, 'rate': rate,
                    'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0,
                    'option_type': option_type, 'method': method
                }
            }
            
        elif model == 'sabr':
            # Use defaults from config if not provided
            defaults = config.get('models', {}).get('sabr', {}).get('default_params', {})
            alpha = alpha or defaults.get('alpha', 0.3)
            beta = beta or defaults.get('beta', 0.5)
            rho = rho or defaults.get('rho', -0.3)
            
            price_result = sabr_price(spot, strike, maturity, rate, alpha, beta, rho, option_type)
            
            result = {
                'model': 'SABR',
                'price': price_result,
                'parameters': {
                    'spot': spot, 'strike': strike, 'maturity': maturity, 'rate': rate,
                    'alpha': alpha, 'beta': beta, 'rho': rho, 'option_type': option_type
                }
            }
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(f"\n{result['model']} Option Price: {result['price']:.6f}")
            if 'std_error' in result and result['std_error']:
                click.echo(f"Monte Carlo Standard Error: {result['std_error']:.6f}")
            if 'greeks' in result:
                click.echo("\nGreeks:")
                for greek, value in result['greeks'].items():
                    if greek != 'price':
                        click.echo(f"  {greek.capitalize()}: {value:.6f}")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', type=click.Choice(['heston', 'sabr']), required=True,
              help='Model to calibrate')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Market data CSV file')
@click.option('--method', type=click.Choice(['least_squares', 'maximum_likelihood']),
              default='least_squares', help='Calibration method')
@click.option('--optimizer', type=click.Choice(['L-BFGS-B', 'differential_evolution']),
              default='L-BFGS-B', help='Optimization algorithm')
@click.option('--output', type=click.Path(), help='Output file for calibrated parameters')
@click.option('--fixed-beta', type=float, help='Fix beta parameter (SABR only)')
def calibrate(model, data, method, optimizer, output, fixed_beta):
    """Calibrate model parameters to market data."""
    
    try:
        # Load market data
        market_data = pd.read_csv(data)
        
        # Validate required columns
        required_columns = ['strike', 'maturity', 'price', 'option_type']
        for col in required_columns:
            if col not in market_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        click.echo(f"Loaded {len(market_data)} market data points")
        
        if model == 'heston':
            calibrator = HestonCalibrator(market_data)
            initial_params = calibrator._get_initial_params()
            bounds = calibrator._get_parameter_bounds()
            
        elif model == 'sabr':
            calibrator = SABRCalibrator(market_data, fixed_beta=fixed_beta)
            initial_params = calibrator._get_initial_params()
            bounds = calibrator._get_parameter_bounds()
        
        click.echo(f"Starting {model.upper()} calibration...")
        
        # Perform calibration
        result = calibrator.calibrate(
            initial_params=initial_params,
            bounds=bounds,
            method=method,
            optimizer=optimizer
        )
        
        if result['success']:
            click.echo("✓ Calibration successful!")
            click.echo(f"Objective value: {result['objective_value']:.6f}")
            click.echo(f"RMSE: {result['rmse']:.6f}")
            click.echo(f"R-squared: {result['r_squared']:.6f}")
            click.echo(f"Execution time: {result['execution_time']:.2f}s")
            
            # Display parameters
            click.echo("\nCalibrated Parameters:")
            if model == 'heston':
                param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'V0']
            else:
                if fixed_beta is not None:
                    param_names = ['alpha', 'rho']
                else:
                    param_names = ['alpha', 'beta', 'rho']
            
            for name, value in zip(param_names, result['optimal_params']):
                click.echo(f"  {name}: {value:.6f}")
            
            # Save results if output specified
            if output:
                output_data = {
                    'model': model,
                    'calibration_results': result,
                    'parameters': dict(zip(param_names, result['optimal_params']))
                }
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                click.echo(f"\nResults saved to {output}")
        
        else:
            click.echo("✗ Calibration failed!")
            click.echo(f"Message: {result.get('message', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', type=click.Choice(['heston', 'sabr', 'black_scholes']),
              default='heston', help='Model for synthetic data')
@click.option('--n-strikes', type=int, default=10, help='Number of strike prices')
@click.option('--n-maturities', type=int, default=5, help='Number of maturities')
@click.option('--noise-level', type=float, default=0.01, help='Noise level')
@click.option('--output', type=click.Path(), required=True, help='Output CSV file')
def generate_data(model, n_strikes, n_maturities, noise_level, output):
    """Generate synthetic market data for testing."""
    
    try:
        click.echo(f"Generating synthetic {model} data...")
        
        data = generate_synthetic_market_data(
            model_type=model,
            n_strikes=n_strikes,
            n_maturities=n_maturities,
            noise_level=noise_level
        )
        
        data.to_csv(output, index=False)
        click.echo(f"✓ Generated {len(data)} data points")
        click.echo(f"Saved to {output}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Market data CSV file')
@click.option('--model', type=click.Choice(['bs']), default='bs',
              help='Model for implied volatility calculation')
def implied_vol(data, model):
    """Calculate implied volatilities from market prices."""
    
    try:
        # Load market data
        market_data = pd.read_csv(data)
        
        # Calculate implied volatilities
        if model == 'bs':
            from calibration.calibrator import ModelCalibrator
            calibrator = ModelCalibrator(None, market_data)
            data_with_iv = calibrator.calculate_implied_volatilities()
            
            # Display summary statistics
            valid_ivs = data_with_iv['implied_vol'].dropna()
            click.echo(f"Calculated implied volatilities for {len(valid_ivs)} options")
            click.echo(f"Mean IV: {valid_ivs.mean():.4f}")
            click.echo(f"Std IV: {valid_ivs.std():.4f}")
            click.echo(f"Min IV: {valid_ivs.min():.4f}")
            click.echo(f"Max IV: {valid_ivs.max():.4f}")
            
            # Save results
            output_file = data.replace('.csv', '_with_iv.csv')
            data_with_iv.to_csv(output_file, index=False)
            click.echo(f"Results saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Display information about the options pricing framework."""
    
    click.echo("Options Pricing Framework")
    click.echo("=" * 50)
    click.echo("A comprehensive options pricing library with:")
    click.echo("• Black-Scholes model with Greeks")
    click.echo("• Heston stochastic volatility model")
    click.echo("• SABR model for interest rate derivatives")
    click.echo("• Monte Carlo simulation with variance reduction")
    click.echo("• FFT-based pricing for efficient computation")
    click.echo("• Model calibration to market data")
    click.echo("• Comprehensive testing suite")
    click.echo("")
    click.echo("Use 'python cli.py --help' for command overview")
    click.echo("Use 'python cli.py COMMAND --help' for specific command help")


if __name__ == '__main__':
    cli()
