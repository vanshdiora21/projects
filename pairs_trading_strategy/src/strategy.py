from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def get_hedge_ratio(y, x):
    model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
    return model.coef_[0]

def compute_spread(y, x, hedge_ratio):
    return y - hedge_ratio * x

def generate_zscore_signals(spread, entry_z=1.0, exit_z=0.0):
    zscore = (spread - spread.mean()) / spread.std()
    signal = pd.Series(index=spread.index, data=0)
    signal[zscore > entry_z] = -1
    signal[zscore < -entry_z] = 1
    signal[abs(zscore) < exit_z] = 0
    return signal, zscore
