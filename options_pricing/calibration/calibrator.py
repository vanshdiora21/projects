"""
Model Calibration Framework

This module provides tools for calibrating option pricing models to market data.
Supports various calibration methods including least squares, maximum likelihood,
and robust estimation techniques.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy.stats import norm
import warnings
from typing import Dict, List, Tuple, Optional, Callable
import time


class ModelCalibrator:
    """
    Generic framework for calibrating option pricing models to market data.
    """
    
    def __init__(self, model_class, market_data: pd.DataFrame):
        """
        Initialize the calibrator.
        
        Parameters:
        -----------
        model_class : class
            Option pricing model class to calibrate
        market_data : pd.DataFrame
            Market data with columns: ['strike', 'maturity', 'price', 'option_type', 'spot', 'rate']
        """
        self.model_class = model_class
        self.market_data = market_data.copy()
        self.calibrated_params = {}
        self.calibration_results = {}
        
        # Validate market data
        required_columns = ['strike', 'maturity', 'price', 'option_type']
        for col in required_columns:
            if col not in self.market_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Set default values if not provided
        if 'spot' not in self.market_data.columns:
            self.market_data['spot'] = 100.0
        if 'rate' not in self.market_data.columns:
            self.market_data['rate'] = 0.05
        
        # Calculate additional metrics
        self._calculate_moneyness()
        self._calculate_time_to_expiry()
    
    def _calculate_moneyness(self):
        """Calculate moneyness metrics."""
        self.market_data['moneyness'] = self.market_data['strike'] / self.market_data['spot']
        self.market_data['log_moneyness'] = np.log(self.market_data['moneyness'])
    
    def _calculate_time_to_expiry(self):
        """Calculate time to expiry in years."""
        if self.market_data['maturity'].dtype == 'datetime64[ns]':
            # If datetime, calculate days to expiry
            today = pd.Timestamp.now()
            self.market_data['time_to_expiry'] = (
                (self.market_data['maturity'] - today).dt.days / 365.25
            )
        else:
            # Assume already in years
            self.market_data['time_to_expiry'] = self.market_data['maturity']
    
    def prepare_calibration_data(self, filters: Dict = None) -> pd.DataFrame:
        """
        Prepare and filter calibration data.
        
        Parameters:
        -----------
        filters : dict, optional
            Filtering criteria, e.g., {'moneyness_min': 0.8, 'moneyness_max': 1.2}
            
        Returns:
        --------
        pd.DataFrame
            Filtered calibration data
        """
        data = self.market_data.copy()
        
        if filters:
            for key, value in filters.items():
                if key == 'moneyness_min':
                    data = data[data['moneyness'] >= value]
                elif key == 'moneyness_max':
                    data = data[data['moneyness'] <= value]
                elif key == 'maturity_min':
                    data = data[data['time_to_expiry'] >= value]
                elif key == 'maturity_max':
                    data = data[data['time_to_expiry'] <= value]
                elif key == 'price_min':
                    data = data[data['price'] >= value]
                elif key == 'option_types':
                    data = data[data['option_type'].isin(value)]
        
        # Remove invalid data
        data = data[data['price'] > 0]
        data = data[data['time_to_expiry'] > 0]
        data = data[data['strike'] > 0]
        
        return data
    
    def calculate_implied_volatilities(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate implied volatilities from market prices using Black-Scholes.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Data to calculate IVs for (uses market_data if None)
            
        Returns:
        --------
        pd.DataFrame
            Data with implied volatilities added
        """
        if data is None:
            data = self.market_data.copy()
        
        from ..models.black_scholes import BlackScholesModel
        
        implied_vols = []
        
        for _, row in data.iterrows():
            try:
                bs_model = BlackScholesModel(
                    S0=row['spot'],
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    r=row['rate'],
                    sigma=0.2,  # Initial guess
                    option_type=row['option_type']
                )
                
                iv = bs_model.implied_volatility(row['price'])
                implied_vols.append(iv)
                
            except Exception as e:
                warnings.warn(f"Failed to calculate IV for row {row.name}: {str(e)}")
                implied_vols.append(np.nan)
        
        data['implied_vol'] = implied_vols
        return data
    
    def setup_objective_function(self, method='least_squares', weights=None, 
                                regularization=None) -> Callable:
        """
        Setup the objective function for calibration.
        
        Parameters:
        -----------
        method : str
            'least_squares', 'weighted_least_squares', 'maximum_likelihood'
        weights : array-like, optional
            Weights for each observation
        regularization : dict, optional
            Regularization parameters
            
        Returns:
        --------
        callable
            Objective function
        """
        calibration_data = self.prepare_calibration_data()
        
        def objective_function(params):
            """
            Objective function for parameter optimization.
            
            Parameters:
            -----------
            params : array-like
                Model parameters to optimize
                
            Returns:
            --------
            float
                Objective function value
            """
            try:
                # Create model instance with current parameters
                model = self._create_model_instance(params, calibration_data.iloc[0])
                
                total_error = 0
                valid_points = 0
                errors = []
                
                for _, row in calibration_data.iterrows():
                    try:
                        # Update model parameters for this data point
                        self._update_model_for_row(model, row)
                        
                        # Calculate model price
                        if hasattr(model, 'price'):
                            model_price = model.price()
                        else:
                            model_price = self._get_model_price(model, row)
                        
                        market_price = row['price']
                        
                        if method == 'least_squares':
                            error = (model_price - market_price)**2
                        elif method == 'weighted_least_squares':
                            weight = weights[valid_points] if weights is not None else 1.0
                            error = weight * (model_price - market_price)**2
                        elif method == 'maximum_likelihood':
                            # Assume log-normal errors
                            if model_price > 0 and market_price > 0:
                                error = (np.log(model_price) - np.log(market_price))**2
                            else:
                                error = (model_price - market_price)**2
                        elif method == 'relative_error':
                            if market_price > 0:
                                error = ((model_price - market_price) / market_price)**2
                            else:
                                error = (model_price - market_price)**2
                        
                        errors.append(error)
                        total_error += error
                        valid_points += 1
                        
                    except Exception as e:
                        # Penalize invalid calculations
                        total_error += 1e6
                
                # Add regularization if specified
                if regularization:
                    reg_penalty = self._calculate_regularization_penalty(params, regularization)
                    total_error += reg_penalty
                
                # Return average error if we have valid points
                if valid_points > 0:
                    return total_error / valid_points
                else:
                    return 1e6
                    
            except Exception as e:
                warnings.warn(f"Error in objective function: {str(e)}")
                return 1e6
        
        return objective_function
    
    def _create_model_instance(self, params, sample_row):
        """Create model instance with given parameters."""
        # This method should be overridden by specific model calibrators
        raise NotImplementedError("Subclasses must implement _create_model_instance")
    
    def _update_model_for_row(self, model, row):
        """Update model parameters for specific market data row."""
        # This method should be overridden by specific model calibrators
        pass
    
    def _get_model_price(self, model, row):
        """Get model price for specific market data row."""
        # This method should be overridden by specific model calibrators
        return model.price()
    
    def _calculate_regularization_penalty(self, params, regularization):
        """Calculate regularization penalty."""
        penalty = 0
        
        if 'l1' in regularization:
            penalty += regularization['l1'] * np.sum(np.abs(params))
        
        if 'l2' in regularization:
            penalty += regularization['l2'] * np.sum(params**2)
        
        if 'bounds_penalty' in regularization:
            # Penalty for parameters outside reasonable bounds
            bounds = regularization['bounds_penalty']
            for i, (param, (lower, upper)) in enumerate(zip(params, bounds)):
                if param < lower:
                    penalty += 1000 * (lower - param)**2
                elif param > upper:
                    penalty += 1000 * (param - upper)**2
        
        return penalty
    
    def calibrate(self, initial_params: List[float], bounds: List[Tuple], 
                 method='least_squares', optimizer='L-BFGS-B', 
                 calibration_options: Dict = None) -> Dict:
        """
        Perform model calibration.
        
        Parameters:
        -----------
        initial_params : list
            Initial parameter values
        bounds : list of tuples
            Parameter bounds [(min, max), ...]
        method : str
            Objective function method
        optimizer : str
            Optimization algorithm
        calibration_options : dict, optional
            Additional calibration options
            
        Returns:
        --------
        dict
            Calibration results
        """
        if calibration_options is None:
            calibration_options = {}
        
        # Setup objective function
        objective_func = self.setup_objective_function(method=method)
        
        # Record start time
        start_time = time.time()
        
        # Perform optimization
        if optimizer == 'differential_evolution':
            result = differential_evolution(
                objective_func, 
                bounds, 
                seed=calibration_options.get('seed', 42),
                maxiter=calibration_options.get('maxiter', 1000),
                popsize=calibration_options.get('popsize', 15)
            )
        elif optimizer == 'least_squares':
            # For nonlinear least squares
            def residual_func(params):
                return np.sqrt(objective_func(params))
            
            result = least_squares(
                residual_func,
                initial_params,
                bounds=([b[0] for b in bounds], [b[asset:1] for b in bounds]),
                max_nfev=calibration_options.get('max_nfev', 2000)
            )
        else:
            # Standard optimization
            result = minimize(
                objective_func,
                initial_params,
                method=optimizer,
                bounds=bounds,
                options={
                    'maxiter': calibration_options.get('maxiter', 1000),
                    'ftol': calibration_options.get('ftol', 1e-9)
                }
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store results
        calibration_results = {
            'success': result.success if hasattr(result, 'success') else True,
            'optimal_params': result.x,
            'objective_value': result.fun,
            'execution_time': execution_time,
            'iterations': getattr(result, 'nit', getattr(result, 'nfev', None)),
            'message': getattr(result, 'message', 'Optimization completed'),
            'method': method,
            'optimizer': optimizer
        }
        
        # Calculate fit statistics
        fit_stats = self._calculate_fit_statistics(result.x)
        calibration_results.update(fit_stats)
        
        # Store calibrated parameters
        self.calibrated_params = result.x
        self.calibration_results = calibration_results
        
        return calibration_results
    
    def _calculate_fit_statistics(self, optimal_params) -> Dict:
        """Calculate goodness-of-fit statistics."""
        calibration_data = self.prepare_calibration_data()
        
        model_prices = []
        market_prices = []
        
        for _, row in calibration_data.iterrows():
            try:
                model = self._create_model_instance(optimal_params, row)
                self._update_model_for_row(model, row)
                model_price = self._get_model_price(model, row)
                
                model_prices.append(model_price)
                market_prices.append(row['price'])
            except:
                continue
        
        if len(model_prices) == 0:
            return {'rmse': np.inf, 'mape': np.inf, 'r_squared': -np.inf}
        
        model_prices = np.array(model_prices)
        market_prices = np.array(market_prices)
        
        # Calculate statistics
        rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((model_prices - market_prices) / market_prices)) * 100
        
        # R-squared
        ss_res = np.sum((market_prices - model_prices)**2)
        ss_tot = np.sum((market_prices - np.mean(market_prices))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        return {
            'rmse': rmse,
            'mape': mape,
            'r_squared': r_squared,
            'mean_error': np.mean(model_prices - market_prices),
            'max_error': np.max(np.abs(model_prices - market_prices))
        }
    
    def cross_validate(self, k_folds=5, test_size=0.2) -> Dict:
        """
        Perform cross-validation of the calibrated model.
        
        Parameters:
        -----------
        k_folds : int
            Number of folds for cross-validation
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        data = self.prepare_calibration_data()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        cv_results = []
        
        for train_idx, test_idx in kf.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Create temporary calibrator with training data
            temp_calibrator = self.__class__(self.model_class, train_data)
            
            # Calibrate on training data
            try:
                result = temp_calibrator.calibrate(
                    self._get_initial_params(),
                    self._get_parameter_bounds()
                )
                
                # Test on validation data
                test_errors = []
                for _, row in test_data.iterrows():
                    try:
                        model = temp_calibrator._create_model_instance(
                            result['optimal_params'], row
                        )
                        temp_calibrator._update_model_for_row(model, row)
                        model_price = temp_calibrator._get_model_price(model, row)
                        
                        error = abs(model_price - row['price'])
                        test_errors.append(error)
                    except:
                        continue
                
                cv_results.append({
                    'train_rmse': result['rmse'],
                    'test_rmse': np.sqrt(np.mean(np.array(test_errors)**2)),
                    'test_mape': np.mean(np.array(test_errors) / test_data['price']) * 100
                })
                
            except Exception as e:
                warnings.warn(f"Cross-validation fold failed: {str(e)}")
                continue
        
        if cv_results:
            return {
                'mean_train_rmse': np.mean([r['train_rmse'] for r in cv_results]),
                'mean_test_rmse': np.mean([r['test_rmse'] for r in cv_results]),
                'std_test_rmse': np.std([r['test_rmse'] for r in cv_results]),
                'mean_test_mape': np.mean([r['test_mape'] for r in cv_results]),
                'fold_results': cv_results
            }
        else:
            return {'error': 'All cross-validation folds failed'}
    
    def _get_initial_params(self):
        """Get initial parameter values - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _get_parameter_bounds(self):
        """Get parameter bounds - to be implemented by subclasses."""
        raise NotImplementedError


class HestonCalibrator(ModelCalibrator):
    """Specialized calibrator for Heston model."""
    
    def __init__(self, market_data: pd.DataFrame):
        from ..models.heston import HestonModel
        super().__init__(HestonModel, market_data)
    
    def _create_model_instance(self, params, sample_row):
        """Create Heston model instance."""
        kappa, theta, sigma_v, rho, v0 = params
        
        return self.model_class(
            S0=sample_row['spot'],
            V0=v0,
            r=sample_row['rate'],
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho
        )
    
    def _update_model_for_row(self, model, row):
        """Update Heston model for specific row."""
        model.S0 = row['spot']
        # Other parameters remain constant during calibration
    
    def _get_model_price(self, model, row):
        """Get Heston model price."""
        return model.price_fft(row['strike'], row['time_to_expiry'], row['option_type'])
    
    def _get_initial_params(self):
        """Default initial parameters for Heston."""
        return [2.0, 0.04, 0.3, -0.5, 0.04]  # kappa, theta, sigma_v, rho, v0
    
    def _get_parameter_bounds(self):
        """Parameter bounds for Heston."""
        return [
            (0.1, 10),     # kappa
            (0.01, 1),     # theta
            (0.1, 2),      # sigma_v
            (-0.99, 0.99), # rho
            (0.01, 1)      # v0
        ]


class SABRCalibrator(ModelCalibrator):
    """Specialized calibrator for SABR model."""
    
    def __init__(self, market_data: pd.DataFrame, fixed_beta=None):
        from ..models.sabr import SABRModel
        super().__init__(SABRModel, market_data)
        self.fixed_beta = fixed_beta
    
    def _create_model_instance(self, params, sample_row):
        """Create SABR model instance."""
        if self.fixed_beta is not None:
            alpha, rho = params
            beta = self.fixed_beta
        else:
            alpha, beta, rho = params
        
        # SABR uses forward price - approximate with spot for simplicity
        return self.model_class(
            F0=sample_row['spot'],
            alpha=alpha,
            beta=beta,
            rho=rho
        )
    
    def _get_model_price(self, model, row):
        """Get SABR model price."""
        return model.price(row['strike'], row['time_to_expiry'], row['rate'], row['option_type'])
    
    def _get_initial_params(self):
        """Default initial parameters for SABR."""
        if self.fixed_beta is not None:
            return [0.3, -0.3]  # alpha, rho
        else:
            return [0.3, 0.5, -0.3]  # alpha, beta, rho
    
    def _get_parameter_bounds(self):
        """Parameter bounds for SABR."""
        if self.fixed_beta is not None:
            return [
                (0.01, 2),     # alpha
                (-0.99, 0.99)  # rho
            ]
        else:
            return [
                (0.01, 2),     # alpha
                (0, 1),        # beta
                (-0.99, 0.99)  # rho
            ]


# Utility functions for calibration
def generate_synthetic_market_data(model_type='heston', n_strikes=10, n_maturities=5, 
                                 noise_level=0.01) -> pd.DataFrame:
    """
    Generate synthetic market data for testing calibration procedures.
    
    Parameters:
    -----------
    model_type : str
        'heston', 'sabr', or 'black_scholes'
    n_strikes : int
        Number of strike prices
    n_maturities : int
        Number of maturities
    noise_level : float
        Noise level to add to prices
        
    Returns:
    --------
    pd.DataFrame
        Synthetic market data
    """
    # Basic parameters
    S0 = 100
    r = 0.05
    
    # Generate strikes and maturities
    strikes = np.linspace(80, 120, n_strikes)
    maturities = np.linspace(0.25, 2.0, n_maturities)
    
    data = []
    
    for T in maturities:
        for K in strikes:
            for option_type in ['call', 'put']:
                row = {
                    'spot': S0,
                    'strike': K,
                    'maturity': T,
                    'time_to_expiry': T,
                    'rate': r,
                    'option_type': option_type
                }
                
                # Generate true price based on model type
                if model_type == 'heston':
                    from ..models.heston import heston_price
                    true_price = heston_price(
                        S0=S0, V0=0.04, K=K, T=T, r=r,
                        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5,
                        option_type=option_type, method='fft'
                    )
                elif model_type == 'sabr':
                    from ..models.sabr import sabr_price
                    true_price = sabr_price(
                        F0=S0, K=K, T=T, r=r,
                        alpha=0.3, beta=0.5, rho=-0.3,
                        option_type=option_type
                    )
                else:  # black_scholes
                    from ..models.black_scholes import black_scholes_price
                    true_price = black_scholes_price(S0, K, T, r, 0.2, option_type)
                
                # Add noise
                noise = np.random.normal(0, noise_level * true_price)
                row['price'] = max(true_price + noise, 0.01)  # Ensure positive price
                
                data.append(row)
    
    return pd.DataFrame(data)
