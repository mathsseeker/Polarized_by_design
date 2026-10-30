#!/usr/bin/env python3
"""
Single-panel plot:  P(T) = 1 - w̄_×(T) on the y-axis,  κ = a·S on x-axis.

Pure post-processing of ``collapse_test.csv`` (WS, N=80, ⟨k⟩=8, 8 runs/cell).
No simulations.  No modification of existing files.

Panel:  P(T) = 1 - w̄_×(T)   (polarization coefficient, paper eq. 7)

  - log-x axis
  - points coloured by S (log scale, colour-bar)
  - binned mean curve (black) with ±1 std band (grey)
  - vertical dashed line at κ* = w_0²/2 = 0.125 (heuristic)
  - horizontal dotted line at 0.5
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE    = os.path.dirname(os.path.abspath(__file__))
IN_CSV  = os.path.join(HERE, "collapse_test.csv")
OUT_PNG = os.path.join(HERE, "polarization_vs_kappa.png")

W0 = 0.5
KAPPA_STAR = W0**2 / 2.0   # 0.125

if not os.path.exists(IN_CSV):
    sys.exit(f"[error] {IN_CSV} not found.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
rows = []
with open(IN_CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append({
            "a":      float(r["a"]),
            "S":      float(r["S"]),
            "kappa":  float(r["kappa"]),
            "P":      float(r["P_mean"]),
            "P_std":  float(r["P_std"]),
        })
print(f"[load]  {IN_CSV}  ({len(rows)} rows)")

kappa = np.array([r["kappa"] for r in rows])
S_arr = np.array([r["S"]     for r in rows])
P     = np.array([r["P"]     for r in rows])
P_s   = np.array([r["P_std"] for r in rows])


# ---------------------------------------------------------------------------
# Bin a 1-D series by log10(kappa)
# ---------------------------------------------------------------------------
def bin_log(x, y, n_bins=22):
    edges = np.linspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    idx = np.clip(np.digitize(np.log10(x), edges) - 1, 0, n_bins - 1)
    cx, my, sy = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 2:
            continue
        cx.append(10 ** (0.5 * (edges[b] + edges[b + 1])))
        my.append(y[sel].mean())
        sy.append(y[sel].std(ddof=1))
    return np.asarray(cx), np.asarray(my), np.asarray(sy)


bx_P, bm_P, bs_P = bin_log(kappa, P)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5.5))
norm = LogNorm(vmin=S_arr.min(), vmax=S_arr.max())

# --- panel:  P(T)  vs  κ ---
sc1 = ax1.scatter(kappa, P, c=S_arr, cmap="viridis", norm=norm,
                  s=30, edgecolor="k", linewidth=0.3, zorder=3)
ax1.errorbar(kappa, P, yerr=P_s, fmt="none",
             ecolor="grey", alpha=0.35, lw=0.7, capsize=0, zorder=2)
ax1.fill_between(bx_P, bm_P - bs_P, bm_P + bs_P,
                 color="grey", alpha=0.25, zorder=4, label="binned ±1 std")
ax1.plot(bx_P, bm_P, "-", color="black", lw=2.2, zorder=5,
         label="binned mean")
ax1.set_xscale("log")
ax1.set_xlabel(r"$\kappa = a \cdot S$")
ax1.set_ylabel(r"$P(T) = 1 - \bar w_\times(T)$")
ax1.set_ylim(-0.02, 1.05)
ax1.axvline(KAPPA_STAR, ls="--", color="red", lw=1.2,
            label=fr"$\kappa^* = w_0^2/2 = {KAPPA_STAR:.3f}$")
ax1.axhline(0.5, ls=":", color="grey", lw=0.7)
ax1.set_title(r"$P(T)$  —  polarization coefficient (paper eq. 7)")
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, which="both", alpha=0.25, lw=0.4)

cb = fig.colorbar(sc1, ax=ax1, shrink=0.85, pad=0.02)
cb.set_label(r"$S$  (log scale)")

fig.suptitle(
    r"$P(T) = 1 - \bar{w}_\times(T)$  vs  $\kappa = a \cdot S$"
    "\n"
    r"WS graph,  $N=80$,  $\langle k \rangle = 8$,  $p_D = 0.5$,  "
    r"$w_0 = 0.5$,  $T = 200$,  8 runs/cell",
    fontsize=12, y=1.02,
)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[save]  {OUT_PNG}")
