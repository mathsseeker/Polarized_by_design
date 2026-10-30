"""
Equations for Millennium Bridge Crowd Synchrony Model
Based on Strogatz et al. (2005) and related work on pedestrian-induced bridge oscillations
"""

import numpy as np


# ============= BRIDGE DYNAMICS =============
def bridge_acceleration(X, X_dot, theta_sum, M, B, K, G):
    """
    Compute bridge acceleration from equation (2).
    
    Bridge dynamics: M Ẍ(t) + B Ẋ(t) + K X(t) = Σ G sin θᵢ(t)
    
    Parameters:
    -----------
    X : float
        Current bridge displacement
    X_dot : float
        Current bridge velocity
    theta_sum : float
        Sum of sin(θᵢ) for all pedestrians
    M : float
        Bridge mass
    B : float
        Bridge damping coefficient
    K : float
        Bridge stiffness
    G : float
        Pedestrian forcing amplitude
    
    Returns:
    --------
    float : Bridge acceleration Ẍ(t)
    """
    X_ddot = (G * theta_sum - B * X_dot - K * X) / M
    return X_ddot


def bridge_natural_frequency(K, M):
    """
    Calculate bridge natural frequency.
    
    Ω = √(K/M)
    
    Parameters:
    -----------
    K : float
        Bridge stiffness
    M : float
        Bridge mass
    
    Returns:
    --------
    float : Natural frequency Ω
    """
    return np.sqrt(K / M)


# ============= PEDESTRIAN PHASE DYNAMICS =============
def pedestrian_phase_dynamics(theta, Omega_pedestrian, A_bridge, Psi_bridge, C, alpha):
    """
    Compute pedestrian phase derivatives from equation (3).
    
    θ̇ᵢ(t) = Ωᵢ + C A(t) sin(Ψ(t) - θᵢ(t) + α)
    
    Parameters:
    -----------
    theta : array
        Phase angles of all pedestrians
    Omega_pedestrian : array
        Natural frequencies of pedestrians
    A_bridge : float
        Current bridge amplitude
    Psi_bridge : float
        Current bridge phase
    C : float
        Coupling constant (pedestrian sensitivity to bridge motion)
    alpha : float
        Phase lag parameter
    
    Returns:
    --------
    array : Phase derivatives for all pedestrians
    """
    dtheta_dt = Omega_pedestrian + C * A_bridge * np.sin(Psi_bridge - theta + alpha)
    return dtheta_dt


# ============= BRIDGE AMPLITUDE AND PHASE =============
def bridge_amplitude(X, X_dot, Omega):
    """
    Compute bridge amplitude from equation (5).
    
    A(t) = √(X(t)² + (Ẋ(t)/Ω)²)
    
    Parameters:
    -----------
    X : float
        Bridge displacement
    X_dot : float
        Bridge velocity
    Omega : float
        Bridge natural frequency
    
    Returns:
    --------
    float : Bridge amplitude A(t)
    """
    return np.sqrt(X**2 + (X_dot / Omega)**2)


def bridge_phase(X, X_dot, Omega):
    """
    Compute bridge phase from equation (5).
    
    Ψ(t) = atan2(X(t), Ẋ(t)/Ω)
    
    Parameters:
    -----------
    X : float
        Bridge displacement
    X_dot : float
        Bridge velocity
    Omega : float
        Bridge natural frequency
    
    Returns:
    --------
    float : Bridge phase Ψ(t)
    """
    return np.arctan2(X, X_dot / Omega)


def from_amplitude_phase_to_displacement(A, Psi, Omega):
    """
    Convert bridge amplitude and phase to displacement and velocity.
    
    X(t) = A(t) sin Ψ(t)
    Ẋ(t) = A(t)Ω cos Ψ(t)
    
    Parameters:
    -----------
    A : float
        Bridge amplitude
    Psi : float
        Bridge phase
    Omega : float
        Bridge natural frequency
    
    Returns:
    --------
    tuple : (X, X_dot) - displacement and velocity
    """
    X = A * np.sin(Psi)
    X_dot = A * Omega * np.cos(Psi)
    return X, X_dot


# ============= SYNCHRONIZATION MEASURE =============
def synchronization_measure(theta):
    """
    Compute synchronization order parameter from equation (6).
    
    R(t)e^(iΦ(t)) = (1/N) Σ e^(iθⱼ(t))
    
    IMPORTANT: Computes magnitude of mean, NOT mean of magnitudes.
    R = | (1/N) Σ e^(iθⱼ) | ≠ (1/N) Σ |e^(iθⱼ)| = 1
    
    Parameters:
    -----------
    theta : array
        Phase angles of all pedestrians
    
    Returns:
    --------
    tuple : (R, Phi) - synchronization amplitude and phase
    """
    # Compute mean of complex exponentials
    z = np.mean(np.exp(1j * theta))  # z = (1/N) Σ e^(iθⱼ)
    
    # Take magnitude of the sum (not sum of magnitudes)
    R = np.abs(z)  # R = |z|
    Phi = np.angle(z)
    return R, Phi


# ============= CRITICAL CROWD SIZE =============
def critical_crowd_size(B, Omega, G, C, P_Omega, alpha=np.pi/2):
    """
    Compute critical crowd size for instability from equation (7).
    
    Nc = 2BΩ / (πGC P(Ω))
    
    For the simplest case with α = π/2 and symmetric P(Ω) about bridge frequency,
    the critical crowd size is given by the above formula.
    
    Parameters:
    -----------
    B : float
        Bridge damping coefficient
    Omega : float
        Bridge natural frequency
    G : float
        Pedestrian forcing amplitude per person
    C : float
        Pedestrian coupling sensitivity
    P_Omega : float
        Density of pedestrian natural frequencies at bridge frequency Ω
    alpha : float
        Phase lag parameter (default: π/2)
    
    Returns:
    --------
    float : Critical crowd size Nc
    """
    N_c = (2 * B * Omega) / (np.pi * G * C * P_Omega)
    return N_c


# ============= FULL SYSTEM ODE =============
def bridge_pedestrian_ode(t, state, M, B, K, G, C, alpha, theta_frequencies):
    """
    Complete ODE system for bridge-pedestrian coupling.
    
    State vector: [X, X_dot, theta1, theta2, ..., thetaN]
    
    Parameters:
    -----------
    t : float
        Time (required by solve_ivp)
    state : array
        [X, X_dot, theta1, theta2, ..., thetaN]
    M : float
        Bridge mass
    B : float
        Bridge damping
    K : float
        Bridge stiffness
    G : float
        Pedestrian forcing amplitude
    C : float
        Pedestrian coupling
    alpha : float
        Phase lag
    theta_frequencies : array
        Natural frequencies of pedestrians
    
    Returns:
    --------
    array : Time derivatives of all state variables
    """
    X = state[0]
    X_dot = state[1]
    theta = state[2:]
    
    N = len(theta_frequencies)
    Omega = bridge_natural_frequency(K, M)
    
    # Bridge amplitude and phase
    A_bridge = bridge_amplitude(X, X_dot, Omega)
    Psi_bridge = bridge_phase(X, X_dot, Omega)
    
    # Bridge acceleration
    theta_sum = np.sum(np.sin(theta))
    X_ddot = bridge_acceleration(X, X_dot, theta_sum, M, B, K, G)
    
    # Pedestrian phase derivatives
    dtheta_dt = pedestrian_phase_dynamics(theta, theta_frequencies, A_bridge, Psi_bridge, C, alpha)
    
    # Combine into state derivatives
    dstate_dt = np.concatenate(([X_dot, X_ddot], dtheta_dt))
    
    return dstate_dt
