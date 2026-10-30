#!/usr/bin/env python3
"""
Couzin Model - REAL-TIME Animation
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from couzin import CouzinModel

print("Starting Couzin simulation...")

# Parameters
params = {
    'n_agents': 30,  # Reduced from 50
    'r_r': 1.0,
    'r_o': 5.0,
    'r_a': 10.0,
    'speed': 1.0,
    'turning_rate': np.pi / 6,
    'noise': 0.1,
    'dt': 0.2,
    'domain_size': 50.0,
    'dim': 2,
    'blind_angle': np.pi / 4,
    'seed': None
}

# Global state
model = CouzinModel(**params)
polarisation_history = [model.polarisation()]
time_history = [0]
is_running = True
time_step = 0

# Create figure
print("Creating figure...")
fig = plt.figure(figsize=(15, 8))
fig.suptitle("Couzin Model - LIVE SIMULATION", fontsize=14, weight='bold')

# Main plot
ax_main = fig.add_axes([0.05, 0.30, 0.40, 0.60])
ax_main.set_xlim(0, params['domain_size'])
ax_main.set_ylim(0, params['domain_size'])
ax_main.set_aspect('equal')
ax_main.set_facecolor('#0a0a0a')

# Polarization plot
ax_polar = fig.add_axes([0.55, 0.55, 0.40, 0.35])
ax_polar.set_ylabel('Order Parameter φ')
ax_polar.set_ylim(-0.05, 1.05)
ax_polar.grid(alpha=0.3)
line_polar, = ax_polar.plot([], [], 'steelblue', linewidth=1.5)

# Noise sweep plot
ax_noise = fig.add_axes([0.55, 0.10, 0.40, 0.35])
ax_noise.set_ylabel('Final Order Parameter φ')
ax_noise.set_ylim(-0.05, 1.05)
ax_noise.grid(alpha=0.3)

# Compute and plot noise sweep
print("Computing noise sweep...")
noise_vals = np.linspace(0.0, 1.0, 10)  # Reduced for speed
pol_vals = []
for nv in noise_vals:
    p = {**params, 'noise': nv, 'seed': 42}
    m = CouzinModel(**p)
    for _ in range(100):  # Reduced for speed
        m.step()
    pol_vals.append(m.polarisation())

ax_noise.plot(noise_vals, pol_vals, 'o-', color='coral', linewidth=1.5, markersize=5)
ax_noise.set_xlabel('Noise σ (rad)')

# Sliders
y0 = 0.18
h = 0.018
gap = 0.035

ax_n = fig.add_axes([0.08, y0, 0.34, h])
ax_rr = fig.add_axes([0.08, y0 - gap, 0.34, h])
ax_ro = fig.add_axes([0.08, y0 - 2*gap, 0.34, h])
ax_ra = fig.add_axes([0.08, y0 - 3*gap, 0.34, h])
ax_noise_sl = fig.add_axes([0.08, y0 - 4*gap, 0.34, h])

sl_n = Slider(ax_n, 'n_agents', 10, 200, valinit=50, valstep=10, color='mediumpurple')
sl_rr = Slider(ax_rr, 'r_r', 0.1, 10.0, valinit=1.0, valstep=0.1, color='red')
sl_ro = Slider(ax_ro, 'r_o', 0.5, 20.0, valinit=5.0, valstep=0.5, color='gold')
sl_ra = Slider(ax_ra, 'r_a', 1.0, 30.0, valinit=10.0, valstep=0.5, color='limegreen')
sl_noise = Slider(ax_noise_sl, 'noise', 0.0, 1.0, valinit=0.1, valstep=0.05, color='cornflowerblue')

# Buttons
ax_reset = fig.add_axes([0.08, y0 - 5.3*gap, 0.12, 0.03])
ax_pause = fig.add_axes([0.21, y0 - 5.3*gap, 0.10, 0.03])
ax_rerun = fig.add_axes([0.32, y0 - 5.3*gap, 0.10, 0.03])

btn_reset = Button(ax_reset, 'Reset', color='lightcoral')
btn_pause = Button(ax_pause, 'Pause', color='lightyellow')
btn_rerun = Button(ax_rerun, 'Re-run', color='lightgreen')

# Animation update
def update(frame):
    global time_step, is_running, model, polarisation_history, time_history
    
    if not is_running:
        return []
    
    model.step()
    time_step += 1
    polarisation_history.append(model.polarisation())
    time_history.append(time_step)
    
    ax_main.clear()
    ax_main.set_xlim(0, params['domain_size'])
    ax_main.set_ylim(0, params['domain_size'])
    ax_main.set_aspect('equal')
    ax_main.set_facecolor('#0a0a0a')
    ax_main.set_title(f't={time_step} φ={polarisation_history[-1]:.2f}')
    
    positions = np.array([a.position for a in model.agents])
    directions = np.array([a.direction for a in model.agents])
    
    ax_main.quiver(positions[:, 0], positions[:, 1],
                   directions[:, 0], directions[:, 1],
                   scale=25, color='cyan', width=0.004, alpha=0.8)
    ax_main.scatter(positions[:, 0], positions[:, 1], s=20, c='magenta', alpha=0.7)
    
    if len(positions) > 0:
        h, k = positions[0, 0], positions[0, 1]
        for r, c in [(params['r_r'], 'red'), (params['r_o'], 'gold'), (params['r_a'], 'limegreen')]:
            circ = Circle((h, k), r, fill=False, edgecolor=c, linewidth=1, linestyle='--', alpha=0.5)
            ax_main.add_patch(circ)
    
    line_polar.set_data(time_history, polarisation_history)
    ax_polar.set_xlim(0, max(100, time_step))
    
    return []

def reset(event):
    global model, time_step, polarisation_history, time_history, is_running
    is_running = True
    btn_pause.label.set_text('Pause')
    model = CouzinModel(**params)
    time_step = 0
    polarisation_history = [model.polarisation()]
    time_history = [0]
    ax_polar.clear()
    ax_polar.set_ylim(-0.05, 1.05)
    ax_polar.grid(alpha=0.3)
    line_polar.clear()

def pause(event):
    global is_running
    is_running = not is_running
    btn_pause.label.set_text('Play' if not is_running else 'Pause')

def rerun(event):
    global model, is_running, time_step, polarisation_history, time_history
    is_running = True
    btn_pause.label.set_text('Pause')
    params['n_agents'] = int(sl_n.val)
    params['r_r'] = sl_rr.val
    params['r_o'] = max(sl_ro.val, params['r_r'] + 0.1)
    params['r_a'] = max(sl_ra.val, params['r_o'] + 0.1)
    params['noise'] = sl_noise.val
    model = CouzinModel(**params)
    time_step = 0
    polarisation_history = [model.polarisation()]
    time_history = [0]

btn_reset.on_clicked(reset)
btn_pause.on_clicked(pause)
btn_rerun.on_clicked(rerun)

print("Starting animation...")
anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

print("\nWindow should appear now!")
print("Controls:")
print("  - Pause: toggle play/pause")
print("  - Reset: restart")
print("  - Re-run: apply slider changes")
print()

plt.show()
print("Done!")
