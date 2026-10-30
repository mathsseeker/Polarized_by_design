"""
Controlled Crowd Ramp Experiment for Millennium Bridge
Gradually increases crowd size and observes bridge response
"""

import numpy as np
from scipy.integrate import solve_ivp
from equations import (
    bridge_pedestrian_ode, bridge_amplitude, synchronization_measure,
    critical_crowd_size, bridge_natural_frequency
)


def initialize_pedestrians(N, omega_mean=0.0, omega_std=0.5):
    """
    Initialize pedestrian phases and frequencies.
    
    Parameters:
    -----------
    N : int
        Number of pedestrians
    omega_mean : float
        Mean of natural frequency distribution
    omega_std : float
        Standard deviation of natural frequency distribution
    
    Returns:
    --------
    tuple : (theta, omega) - phases and frequencies
    """
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(omega_mean, omega_std, N)
    return theta, omega


def initialize_bridge_state(initial_displacement=0.01, initial_velocity=0.0):
    """
    Initialize bridge state.
    
    Parameters:
    -----------
    initial_displacement : float
        Initial bridge displacement X(0)
    initial_velocity : float
        Initial bridge velocity Ẋ(0)
    
    Returns:
    --------
    tuple : (X, X_dot) - displacement and velocity
    """
    return initial_displacement, initial_velocity


def create_initial_state(X, X_dot, theta):
    """
    Combine bridge and pedestrian states into full state vector.
    
    State: [X, X_dot, theta1, theta2, ..., thetaN]
    
    Parameters:
    -----------
    X : float
        Bridge displacement
    X_dot : float
        Bridge velocity
    theta : array
        Pedestrian phases
    
    Returns:
    --------
    array : Full state vector
    """
    return np.concatenate(([X, X_dot], theta))


def extract_state(state):
    """
    Extract bridge displacement, velocity, and pedestrian phases from state vector.
    
    Parameters:
    -----------
    state : array
        Full state vector [X, X_dot, theta1, ..., thetaN]
    
    Returns:
    --------
    tuple : (X, X_dot, theta)
    """
    X = state[0]
    X_dot = state[1]
    theta = state[2:]
    return X, X_dot, theta


def compute_critical_crowd_size(bridge_params, pedestrian_params):
    """
    Compute critical crowd size Nc using equation (7) from equations.py.
    
    Nc = 2BΩ / (πGC P(Ω))
    
    Parameters:
    -----------
    bridge_params : dict
        Bridge parameters: {'M': mass, 'B': damping, 'K': stiffness, 'G': forcing}
    pedestrian_params : dict
        Pedestrian parameters: {'C': coupling, 'alpha': phase_lag, 'omega_mean', 'omega_std'}
    
    Returns:
    --------
    float : Critical crowd size Nc
    """
    # Get bridge natural frequency
    Omega = bridge_natural_frequency(bridge_params['K'], bridge_params['M'])
    
    # Estimate P(Ω) - probability density of pedestrian frequencies at bridge frequency
    # Using normal distribution: P(Ω) = (1/(σ√(2π))) * exp(-(Ω - μ)² / (2σ²))
    omega_mean = pedestrian_params.get('omega_mean', 0.0)
    omega_std = pedestrian_params.get('omega_std', 0.5)
    
    # PDF of normal distribution at Ω
    P_Omega = (1 / (omega_std * np.sqrt(2 * np.pi))) * np.exp(
        -((Omega - omega_mean) ** 2) / (2 * omega_std ** 2)
    )
    
    # Compute critical crowd size
    N_c = critical_crowd_size(
        bridge_params['B'],
        Omega,
        bridge_params['G'],
        pedestrian_params['C'],
        P_Omega,
        pedestrian_params.get('alpha', np.pi/2)
    )
    
    return N_c


def controlled_crowd_ramp(N0=50, N_max=500, delta_N=50, delta_T=20, 
                         t_total=None, bridge_params=None, pedestrian_params=None):
    """
    Run controlled crowd ramp experiment.
    
    Gradually increases crowd from N0 to N_max, adding delta_N pedestrians every delta_T seconds.
    
    Parameters:
    -----------
    N0 : int
        Initial number of pedestrians
    N_max : int
        Maximum number of pedestrians
    delta_N : int
        Number of pedestrians to add at each step
    delta_T : float
        Time interval between crowd additions (seconds)
    t_total : float
        Total simulation time (if None, computed from crowd ramp)
    bridge_params : dict
        Bridge parameters: {'M': mass, 'B': damping, 'K': stiffness, 'G': forcing}
    pedestrian_params : dict
        Pedestrian parameters: {'C': coupling, 'alpha': phase_lag, 'omega_mean': 0, 'omega_std': 0.5}
    
    Returns:
    --------
    dict : Solution data including times, states, bridge response, and pedestrian sync
    """
    
    # Default parameters
    if bridge_params is None:
        bridge_params = {
            'M': 1000,      # Bridge mass (tons)
            'B': 1500,      # Damping (N·s/m)
            'K': 50000,     # Stiffness (N/m)
            'G': 100        # Forcing amplitude per person (N)
        }
    
    if pedestrian_params is None:
        pedestrian_params = {
            'C': 200,           # Coupling sensitivity
            'alpha': np.pi/2,   # Phase lag
            'omega_mean': 0.0,
            'omega_std': 0.5
        }
    
    # Compute total time if not provided
    num_additions = (N_max - N0) // delta_N
    if t_total is None:
        t_total = num_additions * delta_T + delta_T
    
    # Storage for results
    t_all = np.array([])
    X_all = np.array([])
    X_dot_all = np.array([])
    r_all = np.array([])  # Synchronization measure
    A_all = np.array([])  # Bridge amplitude
    N_all = np.array([])  # Crowd size at each time
    events_time = []      # Times when crowd was added
    
    # Initialize bridge
    X, X_dot = initialize_bridge_state()
    
    # Initialize first crowd
    theta, omega = initialize_pedestrians(N0, 
                                         pedestrian_params['omega_mean'],
                                         pedestrian_params['omega_std'])
    
    current_state = create_initial_state(X, X_dot, theta)
    current_N = N0
    t_current = 0.0
    
    # Compute critical crowd size
    N_critical = compute_critical_crowd_size(bridge_params, pedestrian_params)
    Omega = bridge_natural_frequency(bridge_params['K'], bridge_params['M'])
    
    print(f"Starting crowd ramp: N0={N0}, N_max={N_max}, delta_N={delta_N}, delta_T={delta_T}s")
    print(f"Total simulation time: {t_total}s")
    print(f"Bridge natural frequency: Ω = {Omega:.3f} rad/s")
    print(f"Critical crowd size (theory): Nc = {N_critical:.0f} pedestrians")
    print(f"=" * 60)
    
    # Run simulation with crowd additions
    while t_current < t_total and current_N < N_max:
        # Time to next crowd addition or end of simulation
        t_next_event = min(t_current + delta_T, t_total)
        
        print(f"Time: {t_current:6.1f}s - Running with N={current_N} pedestrians until t={t_next_event:.1f}s")
        
        # Solve ODE until next event
        def ode_system(t, state):
            return bridge_pedestrian_ode(
                t, state, 
                bridge_params['M'], bridge_params['B'], bridge_params['K'], 
                bridge_params['G'], pedestrian_params['C'], pedestrian_params['alpha'],
                omega
            )
        
        sol = solve_ivp(
            ode_system,
            (t_current, t_next_event),
            current_state,
            dense_output=True,
            max_step=0.1,
            method='RK45'
        )
        
        # Store results
        t_all = np.concatenate([t_all, sol.t])
        
        for i, t in enumerate(sol.t):
            state = sol.y[:, i]
            X, X_dot, theta_i = extract_state(state)
            X_all = np.concatenate([X_all, [X]])
            X_dot_all = np.concatenate([X_dot_all, [X_dot]])
            
            # Compute synchronization and amplitude
            r, _ = synchronization_measure(theta_i)
            r_all = np.concatenate([r_all, [r]])
            
            Omega = np.sqrt(bridge_params['K'] / bridge_params['M'])
            A = bridge_amplitude(X, X_dot, Omega)
            A_all = np.concatenate([A_all, [A]])
            
            N_all = np.concatenate([N_all, [current_N]])
        
        # Update state for next segment
        current_state = sol.y[:, -1]
        t_current = t_next_event
        
        # Add new pedestrians if not at max
        if t_current < t_total and current_N < N_max:
            N_new = min(current_N + delta_N, N_max)
            theta_new, omega_new = initialize_pedestrians(
                N_new - current_N,
                pedestrian_params['omega_mean'],
                pedestrian_params['omega_std']
            )
            
            # Append new pedestrians to state
            X, X_dot, theta_old = extract_state(current_state)
            theta = np.concatenate([theta_old, theta_new])
            omega = np.concatenate([omega, omega_new])
            current_state = create_initial_state(X, X_dot, theta)
            current_N = N_new
            
            events_time.append(t_current)
            print(f"  → Added {delta_N} pedestrians. Total: {current_N}")
    
    print("=" * 60)
    print(f"Simulation complete! Final crowd size: {current_N}")
    
    # Compute critical time: first time when N(t) >= N_c
    # t_c = inf{t : N(t) >= N_c}
    indices_exceeding_Nc = np.where(N_all >= N_critical)[0]
    if len(indices_exceeding_Nc) > 0:
        t_c = t_all[indices_exceeding_Nc[0]]
    else:
        t_c = np.inf  # Never reached critical crowd size
    
    # Return results as dictionary
    return {
        'time': t_all,
        'X': X_all,              # Bridge displacement
        'X_dot': X_dot_all,      # Bridge velocity
        'r': r_all,              # Synchronization measure
        'A': A_all,              # Bridge amplitude
        'N': N_all,              # Crowd size at each time
        'events_time': np.array(events_time),  # Times when crowds added
        'bridge_params': bridge_params,
        'pedestrian_params': pedestrian_params,
        'N_critical': N_critical,  # Critical crowd size from theory
        'Omega': Omega,  # Bridge natural frequency
        't_c': t_c  # Time when crowd reaches critical size
    }


if __name__ == "__main__":
    # Run example crowd ramp
    results = controlled_crowd_ramp(
        N0=50,
        N_max=300,
        delta_N=50,
        delta_T=30
    )
    
    print(f"\nResults:")
    print(f"  Time span: {results['time'][0]:.1f}s to {results['time'][-1]:.1f}s")
    print(f"  Max bridge displacement: {np.max(np.abs(results['X'])):.3f} m")
    print(f"  Max bridge amplitude: {np.max(results['A']):.3f} m")
    print(f"  Final synchronization: {results['r'][-1]:.3f}")
    print(f"  Critical crowd size: {results['N_critical']:.0f} pedestrians")
    print(f"  Final crowd size: {int(results['N'][-1])} pedestrians")
    print(f"  Critical time (t_c): ", end="")
    
    if np.isfinite(results['t_c']):
        print(f"{results['t_c']:.1f}s")
        print(f"\n✓ Instability threshold reached at t_c = {results['t_c']:.1f}s")
        print(f"  Bridge amplitude at t_c: {results['A'][np.argmin(np.abs(results['time'] - results['t_c']))]:.3f} m")
    else:
        print(f"Not reached (sim ended at t={results['time'][-1]:.1f}s)")
        print(f"\n✗ Crowd too small to trigger instability.")
