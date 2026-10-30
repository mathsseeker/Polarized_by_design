import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider

from cdima_function import cdima
from cdima_solve import solve_cdima


def cdima_nullclines(a: float, x_min=-5.0, x_max=5.0, n=2000):
    """
    Return arrays for CDIMA nullclines:
      dx/dt = 0  -> y = ((a-x)(1+x^2))/(4x)  (x != 0)
      dy/dt = 0  -> x = 0  OR  y = 1 + x^2
    """
    x = np.linspace(x_min, x_max, n)

    # y-nullcline branch: y = 1 + x^2 (valid for all x)
    y_y_null = 1.0 + x**2

    # x-nullcline branch: avoid x=0 division
    eps = 1e-12
    x_safe = np.where(np.abs(x) < eps, np.nan, x)
    y_x_null = ((a - x) * (1.0 + x**2)) / (4.0 * x_safe)

    return x, y_x_null, y_y_null


def find_equilibria(a: float, b: float) -> list[tuple[float, float]]:
    """
    Find equilibrium points for CDIMA system.
    At equilibrium: dx/dt = 0 and dy/dt = 0
    
    From dy/dt = 0: either x = 0 or y = 1 + x²
    From dx/dt = 0: y = ((a-x)(1+x²))/(4x)
    
    Setting these equal gives equilibrium conditions.
    """
    equilibria = []
    
    # Check x = 0 (if it satisfies dx/dt = 0)
    # At x=0: dx/dt = a - 0 - 0 = a, so x=0 is equilibrium only if a=0
    
    # For y = 1 + x², substitute into dx/dt = 0:
    # a - x - 4x(1+x²)/(1+x²) = 0
    # a - x - 4x = 0
    # a - 5x = 0
    # x = a/5
    
    x_eq = a / 5.0
    y_eq = 1.0 + x_eq**2
    equilibria.append((x_eq, y_eq))
    
    return equilibria


def check_stability(a: float, b: float) -> dict:
    """
    Analyze stability of equilibrium point via Jacobian eigenvalues.
    
    Jacobian at equilibrium (x*, y*):
    J = [[-1 - 4y*/(1+x*²) + 8x*²y*/(1+x*²)²,  -4x*/(1+x*²)        ],
         [b(1 - y*/(1+x*²)) + 2bx*²y*/(1+x*²)², -bx*/(1+x*²)       ]]
    
    At equilibrium: x* = a/5, y* = 1 + x*²
    """
    import numpy as np
    
    # Equilibrium point
    x_star = a / 5.0
    y_star = 1.0 + x_star**2
    
    denom = 1.0 + x_star**2
    
    # Jacobian elements
    J11 = -1.0 - (4.0 * y_star) / denom + (8.0 * x_star**2 * y_star) / (denom**2)
    J12 = -4.0 * x_star / denom
    J21 = b * (1.0 - y_star / denom) + (2.0 * b * x_star**2 * y_star) / (denom**2)
    J22 = -b * x_star / denom
    
    # At equilibrium, y* = 1 + x*², so y*/denom = (1+x*²)/(1+x*²) = 1
    # Therefore: 1 - y*/denom = 0, simplifying J21
    J21 = (2.0 * b * x_star**2 * y_star) / (denom**2)
    
    # Create Jacobian matrix
    J = np.array([[J11, J12],
                  [J21, J22]])
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    
    # Determine stability
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    if all(real_parts < 0):
        stability = "stable"
    elif all(real_parts > 0):
        stability = "unstable"
    else:
        stability = "saddle"
    
    # Check for Hopf bifurcation (complex eigenvalues crossing imaginary axis)
    trace = np.trace(J)
    det = np.linalg.det(J)
    
    return {
        'eigenvalues': eigenvalues,
        'stability': stability,
        'trace': trace,
        'determinant': det,
        'equilibrium': (x_star, y_star)
    }


def hopf_bifurcation_curve(a_values: np.ndarray) -> np.ndarray:
    """
    Compute Hopf bifurcation curve in (a,b) parameter space.
    
    Hopf bifurcation occurs when:
    1. Trace(J) = 0 (real part of eigenvalues = 0)
    2. Det(J) > 0 (complex eigenvalues)
    
    For CDIMA at equilibrium (a/5, 1+(a/5)²), setting Trace(J) = 0
    gives the relationship between a and b at bifurcation.
    """
    b_hopf = []
    
    for a in a_values:
        x_star = a / 5.0
        y_star = 1.0 + x_star**2
        denom = 1.0 + x_star**2
        
        # Jacobian trace elements
        J11 = -1.0 - (4.0 * y_star) / denom + (8.0 * x_star**2 * y_star) / (denom**2)
        # J22 = -b * x_star / denom, but we solve for b
        
        # Trace = J11 + J22 = 0 at Hopf bifurcation
        # J11 + (-b*x_star/denom) = 0
        # b = -J11 * denom / x_star
        
        if x_star > 0.01:  # Avoid division by zero
            b_crit = -J11 * denom / x_star
            b_hopf.append(b_crit)
        else:
            b_hopf.append(np.nan)
    
    return np.array(b_hopf)


def plot_phase_portrait(a: float, b: float, x0: float, y0: float, 
                        xlim=(-5, 5), ylim=(-5, 15), tf=20.0, n=2000):
    """
    Plot nullclines and solution trajectory together.
    """
    # Compute nullclines
    x_nc, y_x_null, y_y_null = cdima_nullclines(a, x_min=xlim[0], x_max=xlim[1], n=n)

    # Solve the IVP
    sol = solve_cdima(a=a, b=b, x0=x0, y0=y0, tf=tf, n=n)
    x_traj = sol.y[0]
    y_traj = sol.y[1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot nullclines
    ax.plot(x_nc, y_x_null, label="x-nullcline (dx/dt = 0)", color="C0", linewidth=2)
    ax.plot(x_nc, y_y_null, label="y-nullcline (y = 1 + x²)", color="C1", linewidth=2)
    ax.axvline(0.0, linestyle="--", color="C1", alpha=0.8, linewidth=2)
    
    # Plot trajectory
    ax.plot(x_traj, y_traj, 'k-', linewidth=1.5, alpha=0.7, label=f"Trajectory from ({x0}, {y0})")
    ax.plot(x0, y0, 'go', markersize=10, label="Initial point", zorder=5)
    ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label="Final point", zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"CDIMA Phase Portrait (a={a}, b={b})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig, ax


def create_interactive_plot(a_init=10.0, b_init=4.0, x0_init=0.5, y0_init=3.0):
    """
    Create an interactive CDIMA visualization with animation.
    
    Features:
    - Phase plane (left): shows trajectory, nullclines, and equilibrium
    - Time series (right): shows x(t) evolution
    - Click on the phase plane to set new initial conditions
    - Use sliders to change a and b parameters
    """
    # Initial solution
    sol = solve_cdima(a=a_init, b=b_init, x0=x0_init, y0=y0_init, tf=20, n=2000)
    
    # Create figure with two subplots and space for sliders
    fig = plt.figure(figsize=(14, 7))
    
    # Phase plane axis (left)
    ax1 = plt.axes([0.05, 0.25, 0.4, 0.65])
    
    # Time series axis (right)
    ax2 = plt.axes([0.55, 0.25, 0.4, 0.65])
    
    # Slider axes
    ax_slider_a = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_slider_b = plt.axes([0.15, 0.06, 0.7, 0.03])
    
    slider_a = Slider(
        ax_slider_a, 
        'a parameter', 
        0.1, 
        20.0, 
        valinit=a_init, 
        valstep=0.5,
        color='steelblue'
    )
    
    slider_b = Slider(
        ax_slider_b, 
        'b parameter', 
        0.1, 
        10.0, 
        valinit=b_init, 
        valstep=0.5,
        color='coral'
    )
    
    # Add text to display current parameter values
    param_text = fig.text(0.5, 0.19, f'Current: a = {a_init:.1f}, b = {b_init:.1f}', 
                          ha='center', fontsize=11, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add instructions
    fig.text(0.5, 0.015, 'CONTROLS: Drag sliders to change a, b  |  Click on phase plane to set initial conditions', 
             ha='center', fontsize=10, style='italic', color='darkblue')
    
    # === Phase Plane (ax1) ===
    # Plot nullclines
    x_nc, y_x_null, y_y_null = cdima_nullclines(a_init, x_min=-1, x_max=5)
    line_x_null, = ax1.plot(x_nc, y_x_null, 'b-', linewidth=2, label='x-nullcline (dx/dt = 0)', alpha=0.7)
    line_y_null, = ax1.plot(x_nc, y_y_null, 'r-', linewidth=2, label='y-nullcline (y = 1 + x²)', alpha=0.7)
    ax1.axvline(0, color='r', linewidth=2, linestyle='--', alpha=0.7)
    
    # Initialize trajectory line
    (plot_trajectory,) = ax1.plot([], [], 'k-', lw=1.5, alpha=0.7, label='Trajectory')
    
    # Equilibrium point
    equilibria = find_equilibria(a_init, b_init)
    eq_x, eq_y = equilibria[0]
    point_eq, = ax1.plot([eq_x], [eq_y], 'r*', markersize=15, 
                         label=f"Equilibrium ({eq_x:.2f}, {eq_y:.2f})", 
                         zorder=6, markeredgecolor='black', markeredgewidth=0.5)
    
    # Initial and current points
    point_init, = ax1.plot([x0_init], [y0_init], 'go', markersize=10, label='Initial', zorder=5)
    point_current, = ax1.plot([], [], 'ko', markersize=6, label='Current', zorder=5)
    
    # Set limits and labels for phase plane
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-5, 30)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Phase Plane (a = {a_init:.1f}, b = {b_init:.1f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # === Time Series (ax2) ===
    (plot_time_series,) = ax2.plot([], [], 'b-', lw=2, label='x(t)')
    
    # Set limits and labels for time series
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-1, 5)
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
        interval=5,
        blit=False,
        repeat=True,
    )
    
    # Store current parameters for updates
    params = {'a': a_init, 'b': b_init, 'x0': x0_init, 'y0': y0_init}
    current_sol = {'data': sol}
    
    def update_plot(new_a=None, new_b=None, new_x0=None, new_y0=None):
        """Recompute solution and restart animation."""
        nonlocal ani
        
        # Update parameters
        if new_a is not None:
            params['a'] = new_a
        if new_b is not None:
            params['b'] = new_b
        if new_x0 is not None:
            params['x0'] = new_x0
        if new_y0 is not None:
            params['y0'] = new_y0
        
        # Stop current animation
        ani.event_source.stop()
        
        # Recompute solution
        new_sol = solve_cdima(
            a=params['a'],
            b=params['b'],
            x0=params['x0'],
            y0=params['y0'],
            tf=20,
            n=2000
        )
        current_sol['data'] = new_sol
        
        # Update nullclines
        x_nc, y_x_null, y_y_null = cdima_nullclines(params['a'], x_min=-1, x_max=5)
        line_x_null.set_data(x_nc, y_x_null)
        line_y_null.set_data(x_nc, y_y_null)
        
        # Update equilibrium point
        equilibria = find_equilibria(params['a'], params['b'])
        eq_x, eq_y = equilibria[0]
        point_eq.set_data([eq_x], [eq_y])
        point_eq.set_label(f"Equilibrium ({eq_x:.2f}, {eq_y:.2f})")
        
        # Update initial point
        point_init.set_data([params['x0']], [params['y0']])
        
        # Clear current point
        point_current.set_data([], [])
        plot_trajectory.set_data([], [])
        plot_time_series.set_data([], [])
        
        # Update titles
        ax1.set_title(f"Phase Plane (a = {params['a']:.1f}, b = {params['b']:.1f})", fontsize=13)
        param_text.set_text(f'Current: a = {params["a"]:.1f}, b = {params["b"]:.1f}')
        
        # Update legend
        ax1.legend(loc='upper left', fontsize=9)
        
        # Recreate animation with new data
        ani = animation.FuncAnimation(
            fig,
            animate,
            fargs=(new_sol.y, new_sol.t),
            frames=len(new_sol.t),
            interval=5,
            blit=False,
            repeat=True,
        )
        
        fig.canvas.draw_idle()
    
    def mouse_click(event: MouseEvent):
        """Handle mouse clicks to set new initial conditions."""
        if event.inaxes == ax1:
            x0_new = event.xdata
            y0_new = event.ydata
            if x0_new is not None and y0_new is not None:
                update_plot(new_x0=x0_new, new_y0=y0_new)
    
    def slider_a_update(val):
        """Handle slider a changes."""
        update_plot(new_a=val)
    
    def slider_b_update(val):
        """Handle slider b changes."""
        update_plot(new_b=val)
    
    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", mouse_click)
    slider_a.on_changed(slider_a_update)
    slider_b.on_changed(slider_b_update)
    
    # Keep sliders in scope to prevent garbage collection
    fig._slider_a = slider_a
    fig._slider_b = slider_b
    
    return fig, ax1, ax2, ani


def compute_stability_map(a_range, b_range):
    """
    Compute stability map for CDIMA system across parameter space.
    Returns a 2D array indicating stability regions.
    """
    stability_map = np.zeros((len(b_range), len(a_range)))
    
    for i, b_val in enumerate(b_range):
        for j, a_val in enumerate(a_range):
            info = check_stability(a_val, b_val)
            real_parts = np.real(info['eigenvalues'])
            
            # Classification:
            # 0 = stable (all real parts < 0)
            # 1 = unstable (any real part > 0)
            if all(real_parts < -0.01):
                stability_map[i, j] = 0  # Stable
            else:
                stability_map[i, j] = 1  # Unstable
    
    return stability_map


def plot_professional_bifurcation():
    """
    Professional interactive CDIMA bifurcation analysis with modern styling.
    Features:
    - Stability heatmap in (a,b) space
    - Click to explore different parameters
    - Animated phase portrait and time series
    - Sliders for fine-tuning
    """
    # Create figure with modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#f5f5f5')
    
    # Define axes with better spacing
    ax_stability = plt.axes([0.05, 0.35, 0.35, 0.55])  # Stability map
    ax_phase = plt.axes([0.48, 0.45, 0.24, 0.45])      # Phase portrait
    ax_time = plt.axes([0.48, 0.1, 0.24, 0.25])        # Time series
    ax_3d_view = plt.axes([0.75, 0.35, 0.23, 0.55])    # Alternative view
    
    # Slider axes
    ax_slider_a = plt.axes([0.48, 0.02, 0.24, 0.02])
    ax_slider_b = plt.axes([0.75, 0.02, 0.23, 0.02])
    
    # Compute stability map
    a_range = np.linspace(0.5, 20, 100)
    b_range = np.linspace(0.5, 15, 100)
    A, B = np.meshgrid(a_range, b_range)
    
    print("Computing stability map...")
    stability_map = compute_stability_map(a_range, b_range)
    
    # Compute Hopf bifurcation curve
    a_curve = np.linspace(0.5, 20, 300)
    b_hopf = hopf_bifurcation_curve(a_curve)
    
    # Plot stability heatmap
    contour = ax_stability.contourf(A, B, stability_map, levels=[0, 0.5, 1], 
                                    colors=['#4CAF50', '#FF5252'], alpha=0.6)
    ax_stability.plot(a_curve, b_hopf, linewidth=3, label='Hopf Bifurcation', 
                     linestyle='--', color='black', alpha=0.8)
    
    # Add contour lines for better visualization
    CS = ax_stability.contour(A, B, stability_map, levels=[0.5], colors='white', 
                             linewidths=2, alpha=0.9)
    
    # Current selection marker
    current_marker, = ax_stability.plot([10], [4], 'o', color='yellow', 
                                       markersize=15, markeredgecolor='black', 
                                       markeredgewidth=2, label='Current', zorder=10)
    
    ax_stability.set_xlim(0.5, 20)
    ax_stability.set_ylim(0.5, 15)
    ax_stability.set_xlabel('Parameter a', fontsize=12, fontweight='bold')
    ax_stability.set_ylabel('Parameter b', fontsize=12, fontweight='bold')
    ax_stability.set_title('CDIMA Stability Map\n(a-b Parameter Space)', 
                          fontsize=13, fontweight='bold', pad=10)
    ax_stability.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_stability.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar-like legend
    from matplotlib.patches import Rectangle
    legend_stable = Rectangle((0.02, 0.85), 0.15, 0.05, transform=ax_stability.transAxes,
                             facecolor='#4CAF50', alpha=0.6, edgecolor='black')
    legend_unstable = Rectangle((0.02, 0.78), 0.15, 0.05, transform=ax_stability.transAxes,
                               facecolor='#FF5252', alpha=0.6, edgecolor='black')
    ax_stability.add_patch(legend_stable)
    ax_stability.add_patch(legend_unstable)
    ax_stability.text(0.18, 0.875, 'Stable', transform=ax_stability.transAxes, 
                     fontsize=9, va='center')
    ax_stability.text(0.18, 0.805, 'Unstable', transform=ax_stability.transAxes, 
                     fontsize=9, va='center')
    
    # Initialize phase portrait
    ax_phase.set_xlim(-1, 5)
    ax_phase.set_ylim(-5, 30)
    ax_phase.set_xlabel('x', fontsize=11, fontweight='bold')
    ax_phase.set_ylabel('y', fontsize=11, fontweight='bold')
    ax_phase.set_title('Phase Portrait', fontsize=12, fontweight='bold')
    ax_phase.grid(True, alpha=0.3, linestyle=':')
    ax_phase.set_facecolor('#ffffff')
    
    # Initialize time series
    ax_time.set_xlim(0, 20)
    ax_time.set_ylim(-1, 5)
    ax_time.set_xlabel('Time (t)', fontsize=11, fontweight='bold')
    ax_time.set_ylabel('x(t)', fontsize=11, fontweight='bold')
    ax_time.set_title('Time Evolution', fontsize=12, fontweight='bold')
    ax_time.grid(True, alpha=0.3, linestyle=':')
    ax_time.set_facecolor('#ffffff')
    
    # Initialize 3D view (trajectory endpoints)
    ax_3d_view.set_xlim(-1, 5)
    ax_3d_view.set_ylim(-1, 5)
    ax_3d_view.set_xlabel('x(t)', fontsize=11, fontweight='bold')
    ax_3d_view.set_ylabel('y(t)', fontsize=11, fontweight='bold')
    ax_3d_view.set_title('State Space (Full)', fontsize=12, fontweight='bold')
    ax_3d_view.grid(True, alpha=0.3, linestyle=':')
    ax_3d_view.set_facecolor('#ffffff')
    
    # Sliders
    slider_a = Slider(ax_slider_a, 'a', 0.5, 20.0, valinit=10.0, 
                     valstep=0.1, color='#2196F3')
    slider_b = Slider(ax_slider_b, 'b', 0.5, 15.0, valinit=4.0, 
                     valstep=0.1, color='#FF9800')
    
    # Storage for plot elements
    phase_artists = {'nullclines': [], 'points': [], 'eq': None}
    view3d_artists = []
    
    # Animated elements
    plot_trajectory, = ax_phase.plot([], [], '-', lw=2.5, alpha=0.8, 
                                     color='#1976D2', label='Trajectory')
    point_current, = ax_phase.plot([], [], 'o', markersize=8, 
                                   color='#D32F2F', zorder=5, label='Current')
    point_init, = ax_phase.plot([], [], 's', markersize=10, 
                               color='#388E3C', zorder=5, label='Start')
    
    plot_time_series, = ax_time.plot([], [], '-', lw=2.5, color='#1976D2')
    plot_3d, = ax_3d_view.plot([], [], '-', lw=1.5, alpha=0.6, color='#7B1FA2')
    
    def animate(frame: int, xy: tuple[np.ndarray, np.ndarray], t: np.ndarray):
        """Update function called once per frame."""
        x, y = xy
        
        # Update phase plane
        plot_trajectory.set_data(x[:frame], y[:frame])
        if frame > 0:
            point_current.set_data([x[frame-1]], [y[frame-1]])
        
        # Update time series
        plot_time_series.set_data(t[:frame], x[:frame])
        
        # Update 3D view
        plot_3d.set_data(x[:frame], y[:frame])
        
        return plot_trajectory, point_current, plot_time_series, plot_3d
    
    # Initial solution
    sol = solve_cdima(a=10.0, b=4.0, x0=0.5, y0=3.0, tf=20, n=500)
    
    # Create initial animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(sol.y, sol.t),
        frames=len(sol.t),
        interval=10,
        blit=False,
        repeat=True,
    )
    
    def update_simulation(a_val, b_val, x0=0.5, y0=3.0):
        """Update all plots for given parameters."""
        # Stop current animation (following template pattern)
        ani.event_source.stop()
        
        # Clear previous artists
        for artist in phase_artists['nullclines'] + phase_artists['points']:
            artist.remove()
        if phase_artists['eq']:
            phase_artists['eq'].remove()
        for artist in view3d_artists:
            artist.remove()
        phase_artists['nullclines'].clear()
        phase_artists['points'].clear()
        view3d_artists.clear()
        
        # Plot nullclines in phase portrait
        x_nc, y_x_null, y_y_null = cdima_nullclines(a_val, x_min=-1, x_max=5)
        line1, = ax_phase.plot(x_nc, y_x_null, '--', linewidth=2, alpha=0.6, 
                              color='#1976D2', label='ẋ = 0')
        line2, = ax_phase.plot(x_nc, y_y_null, '--', linewidth=2, alpha=0.6, 
                              color='#D32F2F', label='ẏ = 0')
        line3 = ax_phase.axvline(0, color='#D32F2F', linewidth=1.5, 
                                linestyle='--', alpha=0.4)
        phase_artists['nullclines'].extend([line1, line2, line3])
        
        # Solve ODE with fewer points for faster animation
        new_sol = solve_cdima(a=a_val, b=b_val, x0=x0, y0=y0, tf=20, n=500)
        
        # Plot equilibrium
        eq = find_equilibria(a_val, b_val)[0]
        stability_info = check_stability(a_val, b_val)
        phase_artists['eq'], = ax_phase.plot([eq[0]], [eq[1]], '*', markersize=18, 
                                            color='#FF9800', markeredgecolor='black', 
                                            markeredgewidth=1.5, zorder=6, 
                                            label=f'Eq ({stability_info["stability"]})')
        
        # Plot in 3D view
        eq_3d, = ax_3d_view.plot([eq[0]], [eq[1]], '*', markersize=15, 
                                color='#FF9800', markeredgecolor='black', 
                                markeredgewidth=1, zorder=6)
        null_x, = ax_3d_view.plot(x_nc, y_x_null, '--', linewidth=1.5, 
                                 alpha=0.4, color='#1976D2')
        null_y, = ax_3d_view.plot(x_nc, y_y_null, '--', linewidth=1.5, 
                                 alpha=0.4, color='#D32F2F')
        view3d_artists.extend([eq_3d, null_x, null_y])
        
        # Update initial point
        point_init.set_data([x0], [y0])
        
        # Clear trajectories
        plot_trajectory.set_data([], [])
        point_current.set_data([], [])
        plot_time_series.set_data([], [])
        plot_3d.set_data([], [])
        
        # Update titles
        ax_phase.set_title(f'Phase Portrait\n(a={a_val:.1f}, b={b_val:.1f}) - {stability_info["stability"]}', 
                          fontsize=11, fontweight='bold')
        ax_phase.legend(loc='upper left', fontsize=8, framealpha=0.95)
        
        # Restart animation with new data (following the template pattern)
        ani.frame_seq = ani.new_frame_seq()
        ani._args = (new_sol.y, new_sol.t)
        ani.event_source.start()
        
        fig.canvas.draw_idle()
    
    # Store current parameters
    current_params = {'a': 10.0, 'b': 4.0}
    
    # Initialize with default values
    update_simulation(10, 4)
    
    def on_click(event):
        """Handle clicks on stability map and phase portrait."""
        if event.inaxes == ax_stability:
            # Click on stability map - change parameters
            a_click = event.xdata
            b_click = event.ydata
            
            if a_click and b_click and 0.5 <= a_click <= 20 and 0.5 <= b_click <= 15:
                current_params['a'] = a_click
                current_params['b'] = b_click
                current_marker.set_data([a_click], [b_click])
                slider_a.set_val(a_click)
                slider_b.set_val(b_click)
                update_simulation(a_click, b_click)
        
        elif event.inaxes == ax_phase:
            # Click on phase portrait - change initial conditions
            x0_click = event.xdata
            y0_click = event.ydata
            
            if x0_click and y0_click:
                update_simulation(current_params['a'], current_params['b'], x0_click, y0_click)
        
        elif event.inaxes == ax_3d_view:
            # Click on 3D view - change initial conditions
            x0_click = event.xdata
            y0_click = event.ydata
            
            if x0_click and y0_click:
                update_simulation(current_params['a'], current_params['b'], x0_click, y0_click)
    
    def on_slider_a(val):
        """Handle slider a changes."""
        current_params['a'] = val
        current_marker.set_data([val], [slider_b.val])
        update_simulation(val, slider_b.val)
    
    def on_slider_b(val):
        """Handle slider b changes."""
        current_params['b'] = val
        current_marker.set_data([slider_a.val], [val])
        update_simulation(slider_a.val, val)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    slider_a.on_changed(on_slider_a)
    slider_b.on_changed(on_slider_b)
    
    # Keep sliders in scope
    fig._slider_a = slider_a
    fig._slider_b = slider_b
    
    # Add instructions
    fig.text(0.5, 0.96, 'CDIMA Interactive Bifurcation Analysis', 
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.93, 'Click: Stability Map (change a,b) | Phase/State Plots (change IC) | Use Sliders', 
             ha='center', fontsize=10, style='italic', color='#424242')
    
    plt.style.use('default')  # Reset style for next plots
    return fig, ax_stability, ax_phase, ani
    """
    Plot bifurcation diagram in (a,b) parameter space with Hopf bifurcation curve.
    Click on the diagram to see the corresponding phase portrait with animation.
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Bifurcation diagram (left)
    ax_bif = plt.axes([0.05, 0.15, 0.4, 0.75])
    
    # Phase plane (right)
    ax_phase = plt.axes([0.55, 0.35, 0.4, 0.55])
    
    # Time series (bottom right)
    ax_time = plt.axes([0.55, 0.1, 0.4, 0.2])
    
    # Compute bifurcation curve
    a_range = np.linspace(0.5, 20, 500)
    b_hopf = hopf_bifurcation_curve(a_range)
    
    # Plot bifurcation curve
    ax_bif.plot(a_range, b_hopf, 'r-', linewidth=3, label='Hopf bifurcation curve')
    ax_bif.fill_between(a_range, 0, b_hopf, alpha=0.3, color='lightblue', label='Stable equilibrium')
    ax_bif.fill_between(a_range, b_hopf, 15, alpha=0.3, color='lightcoral', label='Oscillations/Unstable')
    
    # Mark sample points
    sample_points = [(10, 4), (10, 8), (5, 2), (15, 6)]
    for a_pt, b_pt in sample_points:
        ax_bif.plot(a_pt, b_pt, 'ko', markersize=8)
    
    # Current selection marker
    current_marker, = ax_bif.plot([10], [4], 'g*', markersize=20, 
                                  markeredgecolor='black', markeredgewidth=1.5,
                                  label='Current selection', zorder=10)
    
    ax_bif.set_xlim(0, 20)
    ax_bif.set_ylim(0, 15)
    ax_bif.set_xlabel('Parameter a', fontsize=13)
    ax_bif.set_ylabel('Parameter b', fontsize=13)
    ax_bif.set_title('CDIMA Bifurcation Diagram (a-b Parameter Space)', fontsize=14, fontweight='bold')
    ax_bif.grid(True, alpha=0.3)
    ax_bif.legend(loc='upper left', fontsize=10)
    
    # Add text annotation
    ax_bif.text(0.5, 0.95, 'Click on diagram to explore different parameter values',
                transform=ax_bif.transAxes, ha='center', va='top',
                fontsize=10, style='italic', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Initialize phase plane
    ax_phase.set_xlim(-1, 5)
    ax_phase.set_ylim(-5, 30)
    ax_phase.set_xlabel('x', fontsize=11)
    ax_phase.set_ylabel('y', fontsize=11)
    ax_phase.set_title('Phase Portrait', fontsize=12)
    ax_phase.grid(True, alpha=0.3)
    ax_phase.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_phase.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Initialize time series
    ax_time.set_xlim(0, 20)
    ax_time.set_ylim(-1, 5)
    ax_time.set_xlabel('Time (t)', fontsize=11)
    ax_time.set_ylabel('x(t)', fontsize=11)
    ax_time.set_title('Time Series', fontsize=12)
    ax_time.grid(True, alpha=0.3)
    ax_time.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Storage for plot elements
    phase_artists = {'nullclines': [], 'points': [], 'eq': None}
    
    # Create animated elements
    plot_trajectory, = ax_phase.plot([], [], 'k-', lw=1.5, alpha=0.7, label='Trajectory')
    point_current, = ax_phase.plot([], [], 'ko', markersize=6, zorder=5)
    point_init, = ax_phase.plot([], [], 'go', markersize=10, label='Initial', zorder=5)
    
    plot_time_series, = ax_time.plot([], [], 'b-', lw=2)
    
    # Animation reference
    ani_ref = {'animation': None, 'sol': None}
    
    def animate(frame: int):
        """Update function for animation."""
        if ani_ref['sol'] is None:
            return plot_trajectory, point_current, plot_time_series
        
        sol = ani_ref['sol']
        x, y = sol.y[0], sol.y[1]
        t = sol.t
        
        # Update phase plane trajectory
        plot_trajectory.set_data(x[:frame], y[:frame])
        if frame > 0:
            point_current.set_data([x[frame-1]], [y[frame-1]])
        
        # Update time series
        plot_time_series.set_data(t[:frame], x[:frame])
        
        return plot_trajectory, point_current, plot_time_series
    
    def update_phase_plane(a_val, b_val, x0=0.5, y0=3.0):
        """Update phase plane and time series for given (a, b)."""
        nonlocal ani_ref
        
        # Stop existing animation
        if ani_ref['animation'] is not None:
            ani_ref['animation'].event_source.stop()
        
        # Clear previous nullclines and points
        for artist in phase_artists['nullclines'] + phase_artists['points']:
            artist.remove()
        if phase_artists['eq']:
            phase_artists['eq'].remove()
        phase_artists['nullclines'].clear()
        phase_artists['points'].clear()
        
        # Compute and plot nullclines
        x_nc, y_x_null, y_y_null = cdima_nullclines(a_val, x_min=-1, x_max=5)
        line1, = ax_phase.plot(x_nc, y_x_null, 'b-', linewidth=2, alpha=0.7, label='x-nullcline')
        line2, = ax_phase.plot(x_nc, y_y_null, 'r-', linewidth=2, alpha=0.7, label='y-nullcline')
        line3 = ax_phase.axvline(0, color='r', linewidth=1.5, linestyle='--', alpha=0.5)
        phase_artists['nullclines'].extend([line1, line2, line3])
        
        # Solve ODE
        sol = solve_cdima(a=a_val, b=b_val, x0=x0, y0=y0, tf=20, n=2000)
        ani_ref['sol'] = sol
        
        # Plot equilibrium
        eq = find_equilibria(a_val, b_val)[0]
        stability_info = check_stability(a_val, b_val)
        phase_artists['eq'], = ax_phase.plot([eq[0]], [eq[1]], 'r*', markersize=15, 
                                             markeredgecolor='black', markeredgewidth=0.5, 
                                             zorder=6, label=f'Eq: {stability_info["stability"]}')
        
        # Update initial point
        point_init.set_data([x0], [y0])
        
        # Clear trajectory
        plot_trajectory.set_data([], [])
        point_current.set_data([], [])
        plot_time_series.set_data([], [])
        
        # Update title with stability info
        ax_phase.set_title(f'Phase Portrait (a={a_val:.1f}, b={b_val:.1f})\n' +
                          f'Equilibrium: {stability_info["stability"]}', 
                          fontsize=11)
        ax_phase.legend(loc='upper left', fontsize=9)
        
        # Create new animation
        ani_ref['animation'] = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sol.t),
            interval=5,
            blit=False,
            repeat=True,
        )
        
        fig.canvas.draw_idle()
    
    # Initialize with default values
    update_phase_plane(10, 4)
    
    def on_click(event):
        """Handle clicks on bifurcation diagram."""
        if event.inaxes == ax_bif:
            a_click = event.xdata
            b_click = event.ydata
            
            if a_click and b_click:
                # Update marker
                current_marker.set_data([a_click], [b_click])
                
                # Update phase plane with animation
                update_phase_plane(a_click, b_click)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    return fig, ax_bif, ax_phase, ani_ref


if __name__ == "__main__":
    # Professional bifurcation analysis with interactive features
    fig, ax_stability, ax_phase, ani = plot_professional_bifurcation()
    plt.show()
    
    # Alternative: Interactive animation with sliders (uncomment to use)
    # fig, ax1, ax2, ani = create_interactive_plot(a_init=10.0, b_init=4.0, x0_init=0.5, y0_init=3.0)
    # plt.show()