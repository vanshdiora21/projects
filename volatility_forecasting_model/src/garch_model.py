
import pandas as pd
from arch import arch_model

def fit_garch_model(returns, p=1, q=1):
    """
    Fits a GARCH(p, q) model to the returns.

    Parameters:
    returns (pd.Series): Series of log returns
    p (int): Order of GARCH terms
    q (int): Order of ARCH terms

    Returns:
    Fitted ARCH model result
    """
    model = arch_model(returns * 100, vol='Garch', p=p, q=q, rescale=False)
    result = model.fit(disp='off')
    return result

def forecast_volatility(result, horizon=5):
    """
    Forecasts volatility from a fitted GARCH model.

    Parameters:
    result: Fitted GARCH model result object
    horizon (int): Number of days ahead to forecast

    Returns:
    pd.Series: Forecasted volatility
    """
    forecasts = result.forecast(horizon=horizon)
    vol_forecast = forecasts.variance.iloc[-1] ** 0.5  # convert variance to std dev
    return vol_forecast
