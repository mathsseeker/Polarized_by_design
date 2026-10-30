import numpy as np
from scipy.integrate import solve_ivp
from KURAMOTO_EQUATION import kuramoto_ode, kuramoto_ode_pairwise
from oscillators import omega

# Calculate critical coupling strength from frequency distribution
# For normal distribution: Kc = (2/π) * σ
sigma_omega = np.std(omega)
K_c = 2.0 * sigma_omega


def solve_kuramoto(ode_function, t_span, initial_theta, omega, K, method='RK45', dense_output=True):
    """
    Solves the Kuramoto ODE using scipy's solve_ivp.
    
    Parameters:
    -----------
    ode_function : callable
        Either kuramoto_ode (meanfield) or kuramoto_ode_pairwise
    t_span : tuple
        Time span (t_start, t_end)
    initial_theta : array
        Initial phase angles
    omega : array
        Natural frequencies
    K : float
        Coupling strength
    method : str
        Integration method (default: 'RK45')
    dense_output : bool
        If True, returns solution at dense output (default: True)
    
    Returns:
    --------
    sol : solution object
        Solution from solve_ivp
    """
    sol = solve_ivp(
        ode_function,
        t_span,
        initial_theta,
        args=(omega, K),
        method=method,
        dense_output=dense_output
    )
    return sol


def theoretical_order_parameter(K):
    """
    Theoretical order parameter from Kuramoto bifurcation theory.
    
    Uses the pre-calculated critical coupling Kc from the frequency distribution (normal).
    
    Parameters:
    -----------
    K : float
        Coupling strength
    
    Returns:
    --------
    float : Order parameter r
    """
    if K < K_c:
        return 0.0
    else:
        return np.sqrt(1 - K_c / K)


def theoretical_order_parameter_cauchy(K, K_c_cauchy=0.0):
    """
    Theoretical order parameter for CAUCHY distribution (special case).
    
    For Cauchy distribution: K_c = 0 (bifurcation occurs at any K > 0)
    
    Parameters:
    -----------
    K : float
        Coupling strength
    K_c_cauchy : float
        Critical coupling for Cauchy (always 0)
    
    Returns:
    --------
    float : Order parameter r
    """
    if K <= K_c_cauchy:
        return 0.0
    else:
        return np.sqrt(1 - 2.0 / (np.pi * K))  # Cauchy formula
