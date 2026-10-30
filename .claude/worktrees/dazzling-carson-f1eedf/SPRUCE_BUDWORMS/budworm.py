import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10) -> float:
    "Spruce budworm population growth model."
    dxdt = r * x * (1 -x/k) - x**2 / (1 + x**2)
    return dxdt


def spruce_budworm_no_carrying_capacity(t: float, x: float, r: float = 0.5, k: float = 10) -> float:
    """Spruce budworm population growth model WITHOUT the -x/k term (no carrying capacity limiting growth)."""
    dxdt = r * x - x**2 / (1 + x**2)
    return dxdt


def plot_spruce_budworm_rate(x_t: float, r: float = 0.5, k: float = 10):
    """
    Plot the rate of change dx/dt as a function of population x.
    
    Parameters:
    -----------
    x_t : float
        Current population (not used in plot, but included per requirements)
    r : float
        Growth rate parameter (default: 0.5)
    k : float
        Carrying capacity (default: 10)
    """
    # Create array of x values from 0 to k
    x_values = np.linspace(0, k, 500)
    
    # Calculate dx/dt for each x value
    dxdt_values = np.array([spruce_budworm(0, x, r, k) for x in x_values])
    
    # Find equilibrium points where dx/dt = 0
    equilibria = []
    # Search for roots in multiple intervals to find all equilibria
    search_points = np.linspace(0, k, 20)
    for x0 in search_points:
        try:
            root = fsolve(lambda x: spruce_budworm(0, x, r, k), x0)[0]
            # Check if root is in valid range and actually is a root
            if 0 <= root <= k and abs(spruce_budworm(0, root, r, k)) < 1e-6:
                # Check if we haven't already found this root
                if not any(abs(root - eq) < 1e-3 for eq in equilibria):
                    equilibria.append(root)
        except:
            pass
    
    equilibria = sorted(equilibria)
    
    # Classify stability of equilibria
    stable_eq = []
    unstable_eq = []
    
    for eq in equilibria:
        # Check the derivative at the equilibrium point
        epsilon = 1e-6
        # If dx/dt goes from positive to negative, it's stable (crosses from above)
        # If dx/dt goes from negative to positive, it's unstable (crosses from below)
        left_val = spruce_budworm(0, eq - epsilon, r, k)
        right_val = spruce_budworm(0, eq + epsilon, r, k)
        
        if left_val > 0 and right_val < 0:
            stable_eq.append(eq)
        else:
            unstable_eq.append(eq)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, dxdt_values, 'k-', linewidth=2, label='dx/dt')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Null rate of change')
    plt.grid(True, alpha=0.3)
    
    # Mark the current population with a vertical dashed line
    plt.axvline(x=x_t, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Current population (x={x_t:.2f})')
    
    # Mark equilibrium points
    for eq in stable_eq:
        plt.plot(eq, 0, 'bo', markersize=10, label='Stable equilibrium' if eq == stable_eq[0] else '')
    
    for eq in unstable_eq:
        plt.plot(eq, 0, 'ro', markersize=10, label='Unstable equilibrium' if eq == unstable_eq[0] else '')
    
    plt.xlabel('Population (x)', fontsize=12)
    plt.ylabel('Rate of change (dx/dt)', fontsize=12)
    plt.title(f'Spruce Budworm Phase Portrait (r={r}, k={k})', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, k)
    
    # Add some padding to y-axis
    y_margin = (max(dxdt_values) - min(dxdt_values)) * 0.1
    plt.ylim(min(dxdt_values) - y_margin, max(dxdt_values) + y_margin)
    
    plt.tight_layout()
    plt.show()
    
    # Print equilibrium information
    print(f"Equilibrium points for r={r}, k={k}:")
    print(f"  Stable equilibria: {stable_eq}")
    print(f"  Unstable equilibria: {unstable_eq}")


def evolve_spruce_budworm(t: np.ndarray, x: np.ndarray, r: float = 0.5, k: float = 10, t_eval: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve the spruce budworm population forward in time using numerical integration.
    
    Parameters:
    -----------
    t : np.ndarray
        Time array (existing time points)
    x : np.ndarray
        Population array (existing population values)
    r : float
        Growth rate parameter (default: 0.5)
    k : float
        Carrying capacity (default: 10)
    t_eval : float
        Duration to evolve the system forward (default: 10.0)
    
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Updated time and population arrays
    """
    # Define time span from last time point
    t_span = (t[-1], t[-1] + t_eval)
    
    # Create evaluation points distributed along the time span
    # Use 100 points for smooth resolution
    t_eval_points = np.linspace(t_span[0], t_span[1], 100)
    
    # Solve the ODE
    solution = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[x[-1]],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )
    
    t_new = solution.t
    x_new = solution.y[0]
    
    # Concatenate results (skip first point to avoid duplication)
    t = np.concatenate([t, t_new[1:]])
    x = np.concatenate([x, x_new[1:]])
    
    # Ensure non-negative population
    x = np.clip(x, 0, None)
    
    return t, x


def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    """
    Plot the population dynamics over time.
    
    This function visualizes how the spruce budworm population evolves
    from the initial condition through time.
    
    Parameters:
    -----------
    t : np.ndarray
        Time array containing all time points
    x : np.ndarray
        Population array containing population values at each time point
    
    Returns:
    --------
    tuple
        Figure and axes objects (fig, ax) for further customization if needed
    """
    # STEP 1: Create a new figure and axes object
    # - figsize=(10, 6) sets the figure width to 10 inches and height to 6 inches
    # - This creates a canvas for our plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # STEP 2: Plot the population trajectory
    # - t is plotted on the x-axis (time)
    # - x is plotted on the y-axis (population)
    # - 'g-' means green color ('g') with solid line ('-')
    # - linewidth=2 makes the line thicker for better visibility
    # - label creates an entry for the legend
    ax.plot(t, x, 'g-', linewidth=2, label='Population trajectory')
    
    # STEP 3: Set y-axis to start at 0
    # - Populations cannot be negative, so we force the lower limit to 0
    # - max(x) * 1.1 adds 10% padding above the maximum population for better visualization
    # - This ensures the entire trajectory is visible with some breathing room
    ax.set_ylim(0, max(x) * 1.1)
    
    # STEP 4: Add grid for easier reading of values
    # - grid(True) enables the grid
    # - alpha=0.3 makes the grid semi-transparent so it doesn't overwhelm the data
    # - linestyle='--' uses dashed lines for the grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # STEP 5: Label the axes
    # - xlabel sets the label for the x-axis (horizontal)
    # - ylabel sets the label for the y-axis (vertical)
    # - fontsize=12 makes the labels readable
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Population (x)', fontsize=12)
    
    # STEP 6: Add a descriptive title
    # - The title appears at the top of the plot
    # - fontsize=14 makes it slightly larger than axis labels to stand out
    ax.set_title('Spruce Budworm Population Dynamics', fontsize=14)
    
    # STEP 7: Add a legend
    # - This shows what the green line represents
    # - loc='best' automatically places the legend where it won't obscure data
    # - fontsize=10 keeps it readable but not too large
    ax.legend(loc='best', fontsize=10)
    
    # STEP 8: Adjust layout to prevent label cutoff
    # - tight_layout() automatically adjusts spacing to ensure nothing is cut off
    # - This is especially useful when saving the figure
    plt.tight_layout()
    
    # STEP 9: Display the plot
    # - show() renders the plot in a window or notebook
    plt.show()
    
    # STEP 10: Return the figure and axes objects
    # - This allows the user to further customize the plot if needed
    # - For example, they could add more data, change colors, or save to file
    return fig, ax


def compare_models_rate(x_t: float, r: float = 0.5, k: float = 10):
    """
    Compare the rate of change dx/dt for both models side by side.
    
    Parameters:
    -----------
    x_t : float
        Current population
    r : float
        Growth rate parameter (default: 0.5)
    k : float
        Carrying capacity (default: 10)
    """
    x_values = np.linspace(0, k, 500)
    
    # Calculate dx/dt for both models
    dxdt_original = np.array([spruce_budworm(0, x, r, k) for x in x_values])
    dxdt_no_cap = np.array([spruce_budworm_no_carrying_capacity(0, x, r, k) for x in x_values])
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original model
    ax1.plot(x_values, dxdt_original, 'b-', linewidth=2, label='dx/dt (with -x/k)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=x_t, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Current population (x={x_t:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Population (x)', fontsize=12)
    ax1.set_ylabel('Rate of change (dx/dt)', fontsize=12)
    ax1.set_title(f'Original Model: r*x*(1-x/k) - x²/(1+x²)\n(r={r}, k={k})', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, k)
    
    # Plot model without carrying capacity
    ax2.plot(x_values, dxdt_no_cap, 'r-', linewidth=2, label='dx/dt (without -x/k)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=x_t, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Current population (x={x_t:.2f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Population (x)', fontsize=12)
    ax2.set_ylabel('Rate of change (dx/dt)', fontsize=12)
    ax2.set_title(f'Modified Model: r*x - x²/(1+x²)\n(r={r}, k={k})', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, k)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nComparison for r={r}, k={k}:")
    print(f"Original model has carrying capacity limiting growth")
    print(f"Modified model removes the -x/k term, allowing unlimited growth")


def compare_models_evolution(x0: float, r: float = 0.5, k: float = 10, t_max: float = 50.0):
    """
    Compare the population evolution over time for both models.
    
    Parameters:
    -----------
    x0 : float
        Initial population
    r : float
        Growth rate parameter (default: 0.5)
    k : float
        Carrying capacity (default: 10)
    t_max : float
        Maximum time to simulate (default: 50.0)
    """
    t_span = (0, t_max)
    t_eval_points = np.linspace(0, t_max, 500)
    
    # Solve for original model
    sol_original = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[x0],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )
    
    # Solve for model without carrying capacity
    sol_no_cap = solve_ivp(
        fun=spruce_budworm_no_carrying_capacity,
        t_span=t_span,
        y0=[x0],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original model
    ax1.plot(sol_original.t, sol_original.y[0], 'b-', linewidth=2, label='Original Model')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Population (x)', fontsize=12)
    ax1.set_title(f'Original Model: r*x*(1-x/k) - x²/(1+x²)\n(x0={x0}, r={r}, k={k})', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, None)
    
    # Plot model without carrying capacity
    ax2.plot(sol_no_cap.t, sol_no_cap.y[0], 'r-', linewidth=2, label='Modified Model')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Population (x)', fontsize=12)
    ax2.set_title(f'Modified Model: r*x - x²/(1+x²)\n(x0={x0}, r={r}, k={k})', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, None)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nEvolution comparison (x0={x0}, r={r}, k={k}, t_max={t_max}):")
    print(f"Original model final population: {sol_original.y[0][-1]:.4f}")
    print(f"Modified model final population: {sol_no_cap.y[0][-1]:.4f}")

