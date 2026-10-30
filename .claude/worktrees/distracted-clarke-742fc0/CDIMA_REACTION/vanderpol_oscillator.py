"""
Van der Pol Oscillator with Interactive Animation

The Van der Pol oscillator is a prototypical nonlinear system that exhibits 
self-sustained oscillations. For a wide range of initial conditions, trajectories 
converge to a stable limit cycle.

The system (Strogatz form):
    ẋ = μ(y - f(x))
    ẏ = -x/μ

where μ > 0 controls the strength of the nonlinearity (and the fast-slow 
character for large μ), and f(x) = x³/3 - x.

For large μ, the system exhibits relaxation oscillations with two distinct phases:
- Slow phase: trajectories move slowly near the cubic nullcline y = f(x) = x³/3 - x
- Fast phase: trajectories rapidly jump between branches of the nullcline

Reference: Strogatz (2024, chap. 7.5)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider

from vanderpol_function import vanderpol
from vanderpol_solve import solve_vanderpol


# ============================================================================
# Nullclines
# ============================================================================


def analyze_fixed_point_stability(mu: float) -> dict:
    """
    Analyze stability of the fixed point at origin (0, 0).
    
    For Strogatz form:
        ẋ = μ(y - f(x)) where f(x) = x³/3 - x
        ẏ = -x/μ
    
    The Jacobian at (0, 0) is:
        J = [[μ,    μ  ],
             [-1/μ, 0  ]]
    
    Eigenvalues: λ = (μ ± √(μ² - 4))/2
    
    Returns dict with stability info.
    """
    import numpy as np
    
    # Eigenvalues
    discriminant = mu**2 - 4
    if discriminant >= 0:
        lambda1 = (mu + np.sqrt(discriminant)) / 2
        lambda2 = (mu - np.sqrt(discriminant)) / 2
        eigenvalues = [lambda1, lambda2]
    else:
        real_part = mu / 2
        imag_part = np.sqrt(-discriminant) / 2
        eigenvalues = [complex(real_part, imag_part), complex(real_part, -imag_part)]
    
    # Determine stability
    if mu > 0:
        if discriminant >= 0:
            stability = "Unstable node"
        else:
            stability = "Unstable focus (spiral)"
    elif mu < 0:
        stability = "Stable"
    else:
        stability = "Center"
    
    return {
        'eigenvalues': eigenvalues,
        'stability': stability,
        'mu': mu
    }


def vanderpol_nullclines(mu: float, x_min=-3.0, x_max=3.0, n=1000):
    """
    Return arrays for Van der Pol nullclines (Strogatz form):
      ẋ = μ(y - f(x)) = 0  ->  y = f(x) = x³/3 - x
      ẏ = -x/μ = 0  ->  x = 0
    """
    x = np.linspace(x_min, x_max, n)

    # x-nullcline: y = f(x) = x³/3 - x (cubic curve)
    y_x_null = (x**3) / 3.0 - x

    # y-nullcline: x = 0 (vertical line at x=0)
    # We'll return NaN for y values and mark x=0 separately
    y_y_null = np.full_like(x, np.nan)

    return x, y_x_null, y_y_null


# ============================================================================
# Interactive Animation
# ============================================================================


def create_interactive_plot(mu_init=1.0, x0_init=0.1, y0_init=0.1):
    """
    Create an interactive Van der Pol visualization with animation.
    
    Features:
    - Phase plane (left): shows trajectory, nullclines, and limit cycle
    - Time series (right): shows x(t) evolution
    - Click on the phase plane to set new initial conditions
    - Use the slider to change μ parameter
    """
    # Initial solution
    sol = solve_vanderpol(mu=mu_init, x0=x0_init, y0=y0_init, tf=30, n=3000)
    
    # Create figure with two subplots and space for slider
    fig = plt.figure(figsize=(14, 7))
    
    # Phase plane axis (left)
    ax1 = plt.axes([0.05, 0.2, 0.4, 0.7])
    
    # Time series axis (right)
    ax2 = plt.axes([0.55, 0.2, 0.4, 0.7])
    
    # Slider axis (larger and more prominent)
    ax_slider = plt.axes([0.15, 0.08, 0.7, 0.04])
    slider_mu = Slider(
        ax_slider, 
        'μ (damping)', 
        0.1, 
        10.0, 
        valinit=mu_init, 
        valstep=0.1,
        color='steelblue'
    )
    
    # Add text to display current μ value
    mu_text = fig.text(0.45, 0.135, f'Current μ = {mu_init:.1f}', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add instructions
    fig.text(0.5, 0.015, '💡 CONTROLS: Drag slider to change μ  |  Click on phase plane to set initial conditions', 
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # === Phase Plane (ax1) ===
    # Plot nullclines
    x_nc, y_x_null, y_y_null = vanderpol_nullclines(mu_init, x_min=-3, x_max=3)
    line_x_null, = ax1.plot(x_nc, y_x_null, 'b-', linewidth=2, label='ẋ-nullcline (y = x³/3 - x)', alpha=0.7)
    # y-nullcline is x=0, shown as vertical line
    ax1.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.7, label='ẏ-nullcline (x = 0)')
    line_y_null = None  # Placeholder for update function
    
    # Initialize trajectory line
    (plot_trajectory,) = ax1.plot([], [], 'k-', lw=1.5, alpha=0.7, label='Trajectory')
    
    # Fixed point at origin (0, 0) - analyze stability
    stability_info = analyze_fixed_point_stability(mu_init)
    ax1.plot([0], [0], 'r*', markersize=15, label=f"Fixed point ({stability_info['stability']})", 
             zorder=6, markeredgecolor='black', markeredgewidth=0.5)
    
    # Add stability info as annotation
    stability_text = ax1.text(
        0.05, 0.95, 
        f"Fixed Point (0,0):\n{stability_info['stability']}", 
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, edgecolor='darkred')
    )
    
    # Initial and current points
    point_init, = ax1.plot([x0_init], [y0_init], 'go', markersize=10, label='Initial', zorder=5)
    point_current, = ax1.plot([], [], 'ko', markersize=6, label='Current', zorder=5)
    
    # Set limits and labels for phase plane
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Phase Plane (μ = {mu_init:.1f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # === Time Series (ax2) ===
    (plot_time_series,) = ax2.plot([], [], 'b-', lw=2, label='x(t)')
    
    # Set limits and labels for time series
    ax2.set_xlim(0, 30)
    ax2.set_ylim(-3, 3)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('x(t)', fontsize=12)
    ax2.set_title('Time Series', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Animation function
    def animate(frame: int, xy: tuple[np.ndarray, np.ndarray], t: np.ndarray):
        """Update function called once per frame."""
        x, y = xy
        # Update phase plane trajectory
        plot_trajectory.set_data(x[:frame], y[:frame])
        if frame > 0:
            point_current.set_data([x[frame-1]], [y[frame-1]])
        
        # Update time series
        plot_time_series.set_data(t[:frame], x[:frame])
        
        return plot_trajectory, point_current, plot_time_series
    

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(sol.y, sol.t),
        frames=len(sol.t),
        interval=1,
        blit=False,  # Set to False for better widget compatibility
        repeat=True,
    )
    
    # Store current parameters for updates
    params = {'mu': mu_init, 'x0': x0_init, 'y0': y0_init}
    current_sol = {'data': sol}  # Store solution in dict for nonlocal access
    
    def update_plot(new_mu=None, new_x0=None, new_y0=None):
        """Recompute solution and restart animation."""
        nonlocal ani
        
        # Update parameters
        if new_mu is not None:
            params['mu'] = new_mu
        if new_x0 is not None:
            params['x0'] = new_x0
        if new_y0 is not None:
            params['y0'] = new_y0
        
        # Stop current animation
        ani.event_source.stop()
        
        # Recompute solution
        new_sol = solve_vanderpol(
            mu=params['mu'],
            x0=params['x0'],
            y0=params['y0'],
            tf=30,
            n=3000
        )
        current_sol['data'] = new_sol
        
        # Update nullclines
        x_nc, y_x_null, y_y_null = vanderpol_nullclines(params['mu'], x_min=-3, x_max=3)
        line_x_null.set_data(x_nc, y_x_null)
        # y-nullcline (x=0) is fixed, no update needed
        
        # Update initial point
        point_init.set_data([params['x0']], [params['y0']])
        
        # Clear current point
        point_current.set_data([], [])
        plot_trajectory.set_data([], [])
        plot_time_series.set_data([], [])
        
        # Update stability information
        stability_info = analyze_fixed_point_stability(params['mu'])
        stability_text.set_text(f"Fixed Point (0,0):\n{stability_info['stability']}")
        
        # Update titles
        ax1.set_title(f"Phase Plane (μ = {params['mu']:.1f})", fontsize=13)
        mu_text.set_text(f'Current μ = {params["mu"]:.1f}')
        
        # Recreate animation with new data
        ani = animation.FuncAnimation(
            fig,
            animate,
            fargs=(new_sol.y, new_sol.t),
            frames=len(new_sol.t),
            interval=1,
            blit=False,
            repeat=True,
        )
        
        fig.canvas.draw_idle()
    
    def mouse_click(event: MouseEvent):
        """Handle mouse clicks to set new initial conditions."""
        if event.inaxes == ax1:
            x0_new = event.xdata
            y0_new = event.ydata
            update_plot(new_x0=x0_new, new_y0=y0_new)
    
    def slider_update(val):
        """Handle slider changes."""
        # Update the text display first
        mu_text.set_text(f'Current μ = {val:.1f}')
        # Then update the plot
        update_plot(new_mu=val)
    
    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", mouse_click)
    slider_mu.on_changed(slider_update)
    
    # Keep slider in scope to prevent garbage collection
    fig._slider = slider_mu
    
    return fig, ax1, ax2, ani


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    # Create interactive plot with default parameters
    # Try different values of μ:
    # - μ = 1: smooth limit cycle
    # - μ = 3: relaxation oscillation starts to emerge
    # - μ = 10: clear fast-slow dynamics
    fig, ax1, ax2, ani = create_interactive_plot(mu_init=3.0, x0_init=2.0, y0_init=-2.0)
    plt.show()
