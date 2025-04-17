import matplotlib.pyplot as plt
import pandas as pd

def plot_volatility_forecast(realized_vol: pd.Series, predicted_vol: pd.Series):
    """
    Plots realized vs predicted volatility.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(realized_vol.index, realized_vol, label='Realized Volatility', linewidth=2)
    plt.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility (GARCH)', 
             linestyle='--', marker='o')
    plt.title("📉 GARCH Predicted vs Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_analysis(forecasts_dict: dict):
    """
    Plots volatility forecasts for different (p, q) GARCH model configurations.
    
    Parameters:
        forecasts_dict (dict): Keys are tuples (p, q), values are 1D np.arrays of volatility forecasts.
    """
    plt.figure(figsize=(10, 5))
    for (p, q), forecast in forecasts_dict.items():
        plt.plot(range(1, len(forecast) + 1), forecast, marker='o', label=f'GARCH({p},{q})')

    plt.title("📊 Sensitivity Analysis: GARCH(p,q) Forecasts")
    plt.xlabel("Day")
    plt.ylabel("Volatility Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
