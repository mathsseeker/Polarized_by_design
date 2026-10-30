import numpy as np
from oscillators import theta, omega
from scipy.integrate import odeint

def kuramoto_ode(t,theta, omega, K):
    """
    Computes the derivative of the phase angles in the Kuramoto model.

    Parameters:
    theta (array-like): An array of phase angles for each oscillator.
    omega (array-like): An array of natural frequencies for each oscillator.
    K (float): The coupling strength between oscillators.
    N (int): The number of oscillators.

    Returns:
    dtheta_dt (array-like): An array of the derivatives of the phase angles.
    """

    theta = np.mod(theta, 2 * np.pi)
    order_param = np.mean(np.exp(1j * theta))
    r = np.abs(order_param)
    phi = np.angle(order_param)
    dtheta_dt = omega + K * r * np.sin(phi - theta)
    return dtheta_dt


from scipy.integrate import solve_ivp

# Parameters
K = 2.5
t_span = (0, 10)
sol = solve_ivp(
    kuramoto_ode,
    t_span,
    theta,    args=(omega, K)
    )
theta_end = sol.y[:, -1]

def kuramoto_ode_pairwise(t, theta, omega, K):
    """Pairwise form - more expensive, O(N²) but accurate"""
    N = len(theta)
    dtheta_dt = omega.copy()
    for i in range(N):
        coupling = (K / N) * np.sum(np.sin(theta - theta[i]))
        dtheta_dt[i] += coupling
    return dtheta_dt
