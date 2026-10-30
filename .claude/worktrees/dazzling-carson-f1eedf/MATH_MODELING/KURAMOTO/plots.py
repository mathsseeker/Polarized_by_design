import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from KURAMOTO_EQUATION import sol, kuramoto_ode
from oscillators import theta, omega, num_oscillators
from numerical_integration import solve_kuramoto, theoretical_order_parameter, theoretical_order_parameter_cauchy, K_c, sigma_omega


def compute_order_parameter(theta):
    """Compute the order parameter (synchronization measure) from phase angles."""
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    return r


def compute_order_parameter_time_series(sol):
    """Compute order parameter r(t) for entire solution."""
    r_t = [compute_order_parameter(sol.y[:, i]) for i in range(len(sol.t))]
    return np.array(r_t)


# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 10))
ax_phase = plt.subplot(2, 2, 1)
ax_bifurcation = plt.subplot(2, 2, 2)
ax_rt = plt.subplot(2, 1, 2)  # Bottom spans full width

# Define the labels and size of this plot
ax_phase.set_title("Kuramoto Model")
ax_phase.set_xlabel("Cos(theta)")
ax_phase.set_ylabel("Sin(theta)")
ax_phase.set_xlim(-1.1, 1.1)
ax_phase.set_ylim(-1.1, 1.1)
ax_phase.set_aspect("equal")
ax_phase.grid(True)

# Draw unit circle (optional, but it helps to visualize the boundary)
circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
ax_phase.add_artist(circle)

# Initialize scatter plot for oscillators
scatter = ax_phase.scatter([], [], s=50, color="blue", alpha=0.25)
# [] means we are starting with an empty array of points,
# which we will update in the animation loop
# s is the size of the dots, and alpha is the transparency

# Initialize line for the phase centroid (order parameter)
(centroid_line,) = ax_phase.plot([], [], color="red", linewidth=2)
(centroid_point,) = ax_phase.plot([], [], "ro", markersize=8)
# Same idea here:
# we start with empty data and update it in the animation loop


# ============= BIFURCATION DIAGRAM =============
# Plot theoretical bifurcation curves
K_values = np.linspace(0, 5, 200)
r_theoretical = [theoretical_order_parameter(K) for K in K_values]
r_cauchy = [theoretical_order_parameter_cauchy(K) for K in K_values]

ax_bifurcation.plot(K_values, r_theoretical, 'b-', linewidth=2, label=f'Theoretical (Normal σ={sigma_omega:.2f})')
ax_bifurcation.plot(K_values, r_cauchy, 'g--', linewidth=2, label='Theoretical (Cauchy)', alpha=0.7)

# Compute empirical bifurcation data with MULTIPLE RUNS for error bars
K_test_values = np.linspace(0, 5, 20)
r_empirical_mean = []
r_empirical_std = []
num_runs = 10

print("Computing bifurcation diagram with error bars...")
for K_test in K_test_values:
    r_runs = []
    for run in range(num_runs):
        # Initialize oscillators freshly for each run
        from oscillators import initialize_oscillators
        theta_init, omega_init = initialize_oscillators(num_oscillators)
        
        # Run simulation
        sol_test = solve_kuramoto(
            kuramoto_ode,
            (0, 100),
            theta_init,
            omega_init,
            K_test
        )
        # Compute order parameter at final time
        r_final = compute_order_parameter(sol_test.y[:, -1])
        r_runs.append(r_final)
    
    r_empirical_mean.append(np.mean(r_runs))
    r_empirical_std.append(np.std(r_runs))
    print(f"K = {K_test:.2f}, r = {np.mean(r_runs):.3f} ± {np.std(r_runs):.3f}")

r_empirical_mean = np.array(r_empirical_mean)
r_empirical_std = np.array(r_empirical_std)

# Plot with error bars
ax_bifurcation.errorbar(K_test_values, r_empirical_mean, yerr=r_empirical_std, 
                        fmt='ro', capsize=5, markersize=8, label='Empirical (10 runs)', alpha=0.7, zorder=5)

# Add vertical line at critical coupling
ax_bifurcation.axvline(K_c, color='green', linestyle='--', linewidth=2, label=f'$K_c$ = {K_c:.3f}')

# Formatting bifurcation diagram
ax_bifurcation.set_xlabel('Coupling strength (K)', fontsize=12)
ax_bifurcation.set_ylabel('Order parameter (r)', fontsize=12)
ax_bifurcation.set_title('Kuramoto Bifurcation Diagram', fontsize=12, fontweight='bold')
ax_bifurcation.grid(True, alpha=0.3)
ax_bifurcation.legend(fontsize=9)
ax_bifurcation.set_xlim(0, 5)
ax_bifurcation.set_ylim(0, 1.0)

print(f"\nBifurcation diagram computed!")
print(f"Critical coupling strength: K_c = {K_c:.3f}")


# ============= PLOT r(t) VERSUS TIME =============
print("\nComputing r(t) for different coupling strengths...")

# Choose K values: one below K_c, one above
K_below = K_c * 0.5  # Well below critical point
K_above = K_c * 2.0  # Well above critical point

# Initialize fresh oscillators
from oscillators import initialize_oscillators
theta_rt, omega_rt = initialize_oscillators(num_oscillators)

# Solve for K below critical
sol_below = solve_kuramoto(kuramoto_ode, (0, 200), theta_rt, omega_rt, K_below)
r_below = compute_order_parameter_time_series(sol_below)

# Solve for K above critical
sol_above = solve_kuramoto(kuramoto_ode, (0, 200), theta_rt, omega_rt, K_above)
r_above = compute_order_parameter_time_series(sol_above)

# Plot r(t)
ax_rt.plot(sol_below.t, r_below, 'b-', linewidth=2, label=f'K = {K_below:.3f} (below $K_c$)')
ax_rt.plot(sol_above.t, r_above, 'r-', linewidth=2, label=f'K = {K_above:.3f} (above $K_c$)')
ax_rt.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Formatting r(t) plot
ax_rt.set_xlabel('Time (s)', fontsize=12)
ax_rt.set_ylabel('Order parameter r(t)', fontsize=12)
ax_rt.set_title('Order Parameter Evolution Over Time', fontsize=12, fontweight='bold')
ax_rt.grid(True, alpha=0.3)
ax_rt.legend(fontsize=10, loc='right')
ax_rt.set_ylim(0, 1.0)

print(f"K below K_c = {K_below:.3f}, final r ≈ {r_below[-1]:.3f}")
print(f"K above K_c = {K_above:.3f}, final r ≈ {r_above[-1]:.3f}")


def update(frame):
    ax_phase.clear()
    ax_phase.set_xlim(-1.1, 1.1)
    ax_phase.set_ylim(-1.1, 1.1)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True)
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax_phase.add_artist(circle)
    
    theta_t = sol.y[:, frame]
    x = np.cos(theta_t)
    y = np.sin(theta_t)
    ax_phase.scatter(x, y, s=50, color="blue", alpha=0.5)
    
    # Calculate order parameter (centroid)
    z = np.mean(np.exp(1j * theta_t))
    centroid_x = np.real(z)
    centroid_y = np.imag(z)
    
    # Plot centroid point
    ax_phase.plot(centroid_x, centroid_y, "ro", markersize=8)
    
    # Optionally draw a line from center to centroid
    ax_phase.plot([0, centroid_x], [0, centroid_y], color="red", linewidth=2)
    
    ax_phase.set_title(f"Phase Space Animation\nt = {sol.t[frame]:.2f}s")

plt.tight_layout()
anim = FuncAnimation(fig, update, frames=len(sol.t), repeat=True)
plt.show()