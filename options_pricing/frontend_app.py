import streamlit as st
import numpy as np
from models.heston import heston_price

st.title("Options Pricing with Heston Model")

# Inputs
spot = st.number_input("Spot Price (S0)", value=100.0, min_value=0.01)
strike = st.number_input("Strike Price (K)", value=100.0, min_value=0.01)
maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
rate = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f")

kappa = st.number_input("kappa (mean reversion rate)", value=2.0, min_value=0.0)
theta = st.number_input("theta (long-term variance)", value=0.04, min_value=0.0)
sigma_v = st.number_input("sigma_v (vol of vol)", value=0.3, min_value=0.0)
rho = st.slider("rho (correlation)", min_value=-1.0, max_value=1.0, value=-0.5, step=0.01)
v0 = st.number_input("V0 (initial variance)", value=0.04, min_value=0.0)

option_type = st.selectbox("Option Type", ["call", "put"])
method = st.selectbox("Pricing Method", ["fft", "monte_carlo"])

if st.button("Calculate Price"):
    price = heston_price(
        S0=spot,
        V0=v0,
        K=strike,
        T=maturity,
        r=rate,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        option_type=option_type,
        method=method,
    )
    st.success(f"Option Price: {price:.4f}")