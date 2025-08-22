"""
Heston Stochastic Volatility Model

A robust implementation of the Heston model for European option pricing using a
characteristic function and numerical integration, complemented by a Monte Carlo
pricing approach with variance reduction techniques.

References:
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility 
  with applications to bond and currency options. The review of financial studies, 6(2), 327-343.
"""

import numpy as np
from scipy.integrate import quad
import warnings


class HestonModel:
    """
    Heston stochastic volatility model for European options pricing.
    
    The dynamics are:
    dS_t = r * S_t * dt + sqrt(V_t) * S_t * dW1_t
    dV_t = kappa * (theta - V_t) * dt + sigma_v * sqrt(V_t) * dW2_t
    
    with correlated Brownian motions dW1_t and dW2_t (correlation rho).
    """
    
    def __init__(self, S0, V0, r, kappa, theta, sigma_v, rho):
        self.S0 = S0
        self.V0 = V0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

        if 2 * kappa * theta <= sigma_v ** 2:
            warnings.warn("Feller condition not satisfied: 2*kappa*theta <= sigma_v^2")

    def characteristic_function(self, u, T):
        if abs(u) < 1e-14:
            return 1.0
        if T <= 0:
            return 1.0

        u = complex(u)
        kappa, theta, sigma_v, rho, V0, r = self.kappa, self.theta, self.sigma_v, self.rho, self.V0, self.r

        d = np.sqrt((rho * sigma_v * u * 1j - kappa) ** 2 - sigma_v ** 2 * (-u * 1j - u ** 2))

        # Ensure Re(d) <= 0 for numerical stability
        if np.real(d) > 0:
            d = -d

        g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)

        # Avoid g near 1 or -1 to prevent division by zero
        if abs(g) >= 1:
            g = 1 / g

        # Limit exponentials to prevent overflow
        exp_dT = np.exp(d * T)
        if np.abs(exp_dT) > 1e100:
            # For very large exponents, approximate or cutoff
            exp_dT = np.exp(np.sign(np.real(d)) * 100)

        numerator = 1 - g * exp_dT
        denominator = 1 - g
        if abs(denominator) < 1e-14:
            denominator = 1e-14

        try:
            C = (r * u * 1j * T + (kappa * theta / sigma_v ** 2) *
                 ((kappa - rho * sigma_v * u * 1j - d) * T - 2 * np.log(numerator / denominator)))
            D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v ** 2) * ((1 - exp_dT) / numerator)
            cf = np.exp(C + D * V0)
        except (OverflowError, ZeroDivisionError, FloatingPointError):
            cf = 0.0

        if not np.isfinite(cf):
            cf = 0.0

        return cf

    def _integrand(self, phi, K, T, j):
        cf = self.characteristic_function(phi - (j - 1) * 1j, T)
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi
        if phi == 0:
            return 0.0
        return (numerator / denominator).real

    def _P(self, K, T, j):
        val, err = quad(lambda u: self._integrand(u, K, T, j), 0, 100, limit=500, epsabs=1e-10)
        return 0.5 + val / np.pi

    def price_fft(self, K, T, option_type='call'):
        if T <= 0:
            if option_type.lower() == 'call':
                return max(self.S0 - K, 0)
            else:
                return max(K - self.S0, 0)

        P1 = self._P(K, T, 1)
        P2 = self._P(K, T, 2)
        call = self.S0 * P1 - K * np.exp(-self.r * T) * P2
        if option_type.lower() == 'call':
            return max(call.real, 0)
        else:
            put = call - self.S0 + K * np.exp(-self.r * T)
            return max(put.real, 0)

    def price_monte_carlo(self, K, T, option_type='call', n_paths=100000, n_steps=252,
                          antithetic=True, control_variate=False):
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        np.random.seed(42)
        
        sim_paths = n_paths // 2 if antithetic else n_paths
        
        S = np.full(sim_paths, self.S0, dtype=np.float64)
        V = np.full(sim_paths, max(self.V0, 1e-8), dtype=np.float64)
        
        for _ in range(n_steps):
            Z1 = np.random.standard_normal(sim_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * np.random.standard_normal(sim_paths)
            
            V = np.maximum(V, 0)
            sqrtV = np.sqrt(V)
            
            dV = self.kappa * (self.theta - V) * dt + self.sigma_v * sqrtV * Z2 * sqrt_dt
            V += dV
            V = np.maximum(V, 0)
            
            dLogS = (self.r - 0.5 * V) * dt + sqrtV * Z1 * sqrt_dt
            S *= np.exp(dLogS)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - K, 0)
        else:
            payoffs = np.maximum(K - S, 0)
        
        if antithetic:
            np.random.seed(42)
            S_anti = np.full(sim_paths, self.S0, dtype=np.float64)
            V_anti = np.full(sim_paths, max(self.V0, 1e-8), dtype=np.float64)
            
            for _ in range(n_steps):
                Z1 = -np.random.standard_normal(sim_paths)
                Z2 = -self.rho * (-Z1) - np.sqrt(1 - self.rho**2) * np.random.standard_normal(sim_paths)
                
                V_anti = np.maximum(V_anti, 0)
                sqrtV_anti = np.sqrt(V_anti)
                
                dV_anti = self.kappa * (self.theta - V_anti) * dt + self.sigma_v * sqrtV_anti * Z2 * sqrt_dt
                V_anti += dV_anti
                V_anti = np.maximum(V_anti, 0)
                
                dLogS_anti = (self.r - 0.5 * V_anti) * dt + sqrtV_anti * Z1 * sqrt_dt
                S_anti *= np.exp(dLogS_anti)
            
            if option_type.lower() == 'call':
                payoffs_anti = np.maximum(S_anti - K, 0)
            else:
                payoffs_anti = np.maximum(K - S_anti, 0)
            
            payoffs = np.concatenate([payoffs, payoffs_anti])
        
        price = np.exp(-self.r * T) * np.mean(payoffs)
        std_error = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return {'price': price, 'std_error': std_error, 'paths_used': len(payoffs)}


def heston_price(S0, V0, K, T, r, kappa, theta, sigma_v, rho, option_type='call', method='fft'):
    model = HestonModel(S0, V0, r, kappa, theta, sigma_v, rho)
    if method == 'fft':
        return model.price_fft(K, T, option_type)
    elif method == 'monte_carlo':
        result = model.price_monte_carlo(K, T, option_type)
        return result['price']
    else:
        raise ValueError("method must be 'fft' or 'monte_carlo'")
