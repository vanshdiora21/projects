from arch import arch_model
import numpy as np
import pandas as pd

def fit_garch_model(log_returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fits a GARCH(p, q) model to the log returns.
    Returns a fitted model result object.
    """
    model = arch_model(log_returns, vol='Garch', p=p, q=q, dist='normal')
    result = model.fit(disp="off")
    return result

def forecast_volatility(result, horizon: int = 5) -> np.ndarray:
    """
    Forecasts volatility using a fitted GARCH model.
    Returns predicted standard deviation for the specified horizon.
    """
    forecast = result.forecast(horizon=horizon)
    predicted_variance = forecast.variance.values[-1]
    predicted_volatility = np.sqrt(predicted_variance)
    return predicted_volatility
