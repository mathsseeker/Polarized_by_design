"""
Visualization of Millennium Bridge Crowd Ramp Experiment Results

Produces three stacked plots showing:
1. N(t) vs t - Crowd size (staircase function)
2. A(t) vs t - Bridge wobble amplitude
3. R(t) vs t - Degree of synchronization

With vertical dashed line at t_c where crowd reaches critical size N_c
"""

import numpy as np
import matplotlib.pyplot as plt
from people import controlled_crowd_ramp


# Run the crowd ramp experiment
print("Running crowd ramp simulation...\n")
results = controlled_crowd_ramp(
    N0=50,
    N_max=300,
    delta_N=50,
    delta_T=30
)

# Extract results
time = results['time']
N = results['N']
A = results['A']
r = results['r']
t_c = results['t_c']
N_c = results['N_critical']

# Create figure with 3 stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# ========== Plot 1: N(t) vs t (Crowd Size) ==========
axes[0].plot(time, N, 'b-', linewidth=2.5, label='N(t)')
if np.isfinite(t_c):
    axes[0].axvline(t_c, color='red', linestyle='--', linewidth=2.5, 
                    label=f'$N = N_c$ at $t_c = {t_c:.1f}$ s')
    axes[0].plot(t_c, N_c, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)
axes[0].set_ylabel('Crowd Size $N(t)$', fontsize=13, fontweight='bold')
axes[0].set_title('Controlled Crowd Ramp Experiment: Bridge Response to Increasing Crowd', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle=':')
axes[0].legend(fontsize=11, loc='upper left')
axes[0].set_xlim([time[0], time[-1]])

# ========== Plot 2: A(t) vs t (Bridge Amplitude) ==========
axes[1].plot(time, A, 'g-', linewidth=2.5, label='$A(t)$')
if np.isfinite(t_c):
    # Find amplitude at t_c
    idx_tc = np.argmin(np.abs(time - t_c))
    A_at_tc = A[idx_tc]
    axes[1].axvline(t_c, color='red', linestyle='--', linewidth=2.5, 
                    label=f'$N = N_c$ at $t_c = {t_c:.1f}$ s')
    axes[1].plot(t_c, A_at_tc, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)
axes[1].set_ylabel('Bridge Amplitude $A(t)$ [m]', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle=':')
axes[1].legend(fontsize=11, loc='upper left')
axes[1].set_xlim([time[0], time[-1]])

# ========== Plot 3: R(t) vs t (Synchronization) ==========
axes[2].plot(time, r, color='purple', linewidth=2.5, label='$R(t)$')
if np.isfinite(t_c):
    # Find synchronization at t_c
    idx_tc = np.argmin(np.abs(time - t_c))
    r_at_tc = r[idx_tc]
    axes[2].axvline(t_c, color='red', linestyle='--', linewidth=2.5, 
                    label=f'$N = N_c$ at $t_c = {t_c:.1f}$ s')
    axes[2].plot(t_c, r_at_tc, 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)
axes[2].set_xlabel('Time $t$ [s]', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Synchronization $R(t)$', fontsize=13, fontweight='bold')
axes[2].set_ylim([0, 1.05])
axes[2].grid(True, alpha=0.3, linestyle=':')
axes[2].legend(fontsize=11, loc='upper left')
axes[2].set_xlim([time[0], time[-1]])

plt.tight_layout()
plt.savefig('crowd_ramp_results.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'crowd_ramp_results.png'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Critical crowd size (theory):   Nc = {N_c:.0f} pedestrians")
print(f"Critical time (simulation):     tc = {t_c:.1f} s" if np.isfinite(t_c) else f"Critical time: Not reached")
print(f"Final crowd size:               N(T) = {int(N[-1])} pedestrians")
print(f"Bridge response at t=(0, T):")
print(f"  Max amplitude:                Amax = {np.max(A):.4f} m")
print(f"  Final synchronization:        R(T) = {r[-1]:.3f}")
if np.isfinite(t_c):
    idx_tc = np.argmin(np.abs(time - t_c))
    print(f"\nAt critical time tc = {t_c:.1f} s:")
    print(f"  Amplitude:                  A(tc) = {A[idx_tc]:.4f} m")
    print(f"  Synchronization:            R(tc) = {r[idx_tc]:.3f}")
    print(f"  Amplitude increase:         ΔA = {A[idx_tc] - np.min(A):.4f} m")
print("="*60)
