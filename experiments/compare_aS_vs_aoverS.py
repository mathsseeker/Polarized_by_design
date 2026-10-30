#!/usr/bin/env python3
"""
Direct comparison: does κ = a · S collapse the data better than κ' = a / S?

This is a PURE POST-PROCESSING test.  We re-read ``universality_sweep.csv``
(produced by ``universality_test.py``) and replot the same observables —

    P(T) =  1 - w̄_×(T)            (polarization coefficient, eq. 7)
    C    =  |2 f_D(T) - 1|        (consensus index)

— against both candidate control parameters:

    left  column :  κ  = a · S       (our claim)
    right column :  κ' = a / S       (the competing claim)

If a · S is the right grouping, the 9 (topology × ⟨k⟩) curves overlap on
the left and fan out on the right.  We quantify "fan-out" with the median
across-group bin-std on a common log-x grid — same diagnostic we used in
``universality_test.py``.

No simulations are run; no other file is modified.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV  = os.path.join(HERE, "universality_sweep.csv")
OUT_PNG = os.path.join(HERE, "compare_aS_vs_aoverS.png")

if not os.path.exists(IN_CSV):
    sys.exit(f"[error] {IN_CSV} not found — run universality_test.py first.")


# ---------------------------------------------------------------------------
# Load the existing CSV  →  Python lists of dicts
# ---------------------------------------------------------------------------
def _f(x):
    return float(x)

rows = []
with open(IN_CSV) as fh:
    for r in csv.DictReader(fh):
        a, S = _f(r["a"]), _f(r["S"])
        rows.append({
            "graph_type":  r["graph_type"],
            "mean_degree": int(r["mean_degree"]),
            "a":           a,
            "S":           S,
            "kappa":       _f(r["kappa"]),       # = a · S (already in file)
            "kappa_ratio": a / S,                # = a / S (new)
            "P_mean":      _f(r["P_mean"]),
            "C_mean":      _f(r["C_mean"]),
        })
print(f"[load]  {IN_CSV}  ({len(rows)} rows)")

groups = sorted({(r["graph_type"], r["mean_degree"]) for r in rows})
print(f"[load]  {len(groups)} (topology, ⟨k⟩) groups: {groups}")


# ---------------------------------------------------------------------------
# Bin a 1-D series by log10(x)
# ---------------------------------------------------------------------------
def bin_log_x(x_arr, y_arr, n_bins=20, lo=None, hi=None):
    lo = lo if lo is not None else np.log10(x_arr.min())
    hi = hi if hi is not None else np.log10(x_arr.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(np.log10(x_arr), edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 2:
            continue
        centres.append(10 ** (0.5 * (edges[b] + edges[b + 1])))
        means.append(y_arr[sel].mean())
    return np.asarray(centres), np.asarray(means)


# ---------------------------------------------------------------------------
# For each (x_var, metric): compute one binned curve per (graph, ⟨k⟩) group,
# then evaluate all 9 curves on a common log-x grid and take std across them.
# Median-of-bin-std is the universal-collapse score (lower = tighter).
# ---------------------------------------------------------------------------
N_BINS = 20

def group_curves_and_spread(x_key, y_key):
    # Global log range across ALL rows for this x_key — use the same x-grid
    # so every group is evaluated on a common scale.
    x_all = np.array([r[x_key] for r in rows])
    lo, hi = np.log10(x_all.min()), np.log10(x_all.max())
    common_x = np.logspace(lo, hi, N_BINS)

    curves = {}
    for g in groups:
        sub = [r for r in rows if (r["graph_type"], r["mean_degree"]) == g]
        x = np.array([r[x_key] for r in sub])
        y = np.array([r[y_key] for r in sub])
        cx, cm = bin_log_x(x, y, n_bins=N_BINS, lo=lo, hi=hi)
        curves[g] = (cx, cm)

    per_bin_std = []
    for xt in common_x:
        ys_here = []
        for g, (cx, cm) in curves.items():
            if cx.size and cx.min() <= xt <= cx.max():
                ys_here.append(np.interp(np.log10(xt), np.log10(cx), cm))
        if len(ys_here) >= 2:
            per_bin_std.append(float(np.std(ys_here, ddof=1)))
    median_std = float(np.median(per_bin_std)) if per_bin_std else float("nan")
    return curves, median_std


print("[bin]   computing binned curves for all 4 combinations …")
combos = [
    ("P(T)", "P_mean", "kappa",       r"$\kappa  = a \cdot S$"),
    ("P(T)", "P_mean", "kappa_ratio", r"$\kappa' = a / S$"),
    ("C",    "C_mean", "kappa",       r"$\kappa  = a \cdot S$"),
    ("C",    "C_mean", "kappa_ratio", r"$\kappa' = a / S$"),
]
collapses = {}
for metric_lbl, y_key, x_key, x_lbl in combos:
    curves, m_std = group_curves_and_spread(x_key, y_key)
    collapses[(metric_lbl, x_key)] = (curves, m_std, x_lbl)
    print(f"        {metric_lbl:<3} vs {x_lbl}  →  median bin-std (across groups) = {m_std:.3f}")


# ---------------------------------------------------------------------------
# Pooled-data within-bin spread.
#
# This is the diagnostic that actually answers "does this x-axis collapse the
# data onto a single curve?":  pool all 2025 rows, bin by log(x), and look at
# the std of y INSIDE each bin.  Small ⇒ at fixed x the y is well-determined
# ⇒ x is a faithful control parameter.  Large ⇒ different (a, S) pairs that
# happen to share x give different y ⇒ x is not the right grouping.
# ---------------------------------------------------------------------------
def pooled_within_bin_std(x_key, y_key, n_bins=N_BINS):
    x = np.array([r[x_key] for r in rows])
    y = np.array([r[y_key] for r in rows])
    edges = np.linspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    idx = np.clip(np.digitize(np.log10(x), edges) - 1, 0, n_bins - 1)
    stds = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 2:
            stds.append(float(y[sel].std(ddof=1)))
    return float(np.median(stds)) if stds else float("nan")

print("[bin]   computing pooled within-bin std (the collapse diagnostic) …")
pooled = {}
for metric_lbl, y_key, x_key, x_lbl in combos:
    pooled[(metric_lbl, x_key)] = pooled_within_bin_std(x_key, y_key)
    print(f"        {metric_lbl:<3} vs {x_lbl}  →  median within-bin std (pooled) = "
          f"{pooled[(metric_lbl, x_key)]:.3f}")


# ---------------------------------------------------------------------------
# Plot — 2×2 grid, identical y-limits per row
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharey="row")

# Distinguishable visual mapping for 9 groups: marker by topology, size by ⟨k⟩
markers = {"er": "o", "ws": "s", "ba": "^"}
sizes   = {4: 22, 8: 36, 16: 56}
# 9 distinct colours via tab10
cmap = plt.colormaps.get_cmap("tab10")
group_colours = {g: cmap(i % 10) for i, g in enumerate(groups)}

layout = [
    ("P(T)", "kappa",       axes[0, 0], r"$P(T) = 1 - \bar{w}_\times(T)$"),
    ("P(T)", "kappa_ratio", axes[0, 1], r"$P(T) = 1 - \bar{w}_\times(T)$"),
    ("C",    "kappa",       axes[1, 0], r"$C = |2 f_D - 1|$"),
    ("C",    "kappa_ratio", axes[1, 1], r"$C = |2 f_D - 1|$"),
]

def verdict_of(med):
    if med < 0.05:
        return "TIGHT"
    if med < 0.12:
        return "MODERATE"
    return "POOR"

for metric_lbl, x_key, ax, ylabel in layout:
    curves, m_std, x_lbl = collapses[(metric_lbl, x_key)]
    y_key = "P_mean" if metric_lbl == "P(T)" else "C_mean"

    # background scatter (all 2025 points)
    for g in groups:
        sub = [r for r in rows if (r["graph_type"], r["mean_degree"]) == g]
        xs = np.array([r[x_key] for r in sub])
        ys = np.array([r[y_key]  for r in sub])
        ax.scatter(xs, ys,
                   s=sizes[g[1]], marker=markers[g[0]],
                   facecolor="none", edgecolor=group_colours[g],
                   linewidth=0.5, alpha=0.35, zorder=2)

    # foreground binned-mean lines, one per group
    for g, (cx, cm) in curves.items():
        ax.plot(cx, cm, "-", color=group_colours[g], lw=1.6,
                label=f"{g[0].upper()}, ⟨k⟩={g[1]}", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, ls=":", color="grey", lw=0.6, zorder=1)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    ax.set_title(
        f"{metric_lbl}  vs  {x_lbl}\n"
        f"median bin-std across 9 groups = {m_std:.3f}  →  {verdict_of(m_std)}",
        fontsize=10,
    )

# One shared legend at the right
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", fontsize=8,
           bbox_to_anchor=(1.04, 0.5), framealpha=0.9)

fig.suptitle(
    r"Which dimensionless group controls the dynamics:  $a \cdot S$  or  $a / S$?",
    fontsize=13, y=1.00,
)
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[save]  {OUT_PNG}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("  COMPARISON  —  a·S  vs  a/S")
print("=" * 78)

# Diagnostic A — spec'd metric: across-group spread at fixed x.
# Reports topology-universality of the function at each x, NOT whether the
# data collapses onto a single curve.  Both candidates tend to look TIGHT
# whenever the dynamics are themselves topology-independent.
print("  Diagnostic A  —  median bin-std across (topology, ⟨k⟩) groups")
print("                  (probes topology-universality, not collapse-onto-curve)")
print(f"  {'observable':<12}{'x-axis':<14}{'median bin-std':>18}{'verdict':>14}")
table = [
    ("P(T)", "kappa",       r"κ = a·S"),
    ("P(T)", "kappa_ratio", r"κ' = a/S"),
    ("C",    "kappa",       r"κ = a·S"),
    ("C",    "kappa_ratio", r"κ' = a/S"),
]
for metric_lbl, x_key, x_lbl in table:
    m_std = collapses[(metric_lbl, x_key)][1]
    print(f"  {metric_lbl:<12}{x_lbl:<14}{m_std:>18.3f}{verdict_of(m_std):>14}")

print()
# Diagnostic B — the one that actually answers "does this x-axis collapse
# the data onto a single curve?":  pool ALL 2025 rows, bin by log(x), measure
# within-bin std of y.  Different (a, S) pairs that happen to share x but
# disagree on y contribute to this spread, so the *correct* x-axis minimises
# it.
print("  Diagnostic B  —  median within-bin std of pooled 2025 points")
print("                  (probes WHETHER the data lies on a single curve)")
print(f"  {'observable':<12}{'x-axis':<14}{'median within-bin std':>24}{'verdict':>14}")
for metric_lbl, x_key, x_lbl in table:
    m_std = pooled[(metric_lbl, x_key)]
    print(f"  {metric_lbl:<12}{x_lbl:<14}{m_std:>24.3f}{verdict_of(m_std):>14}")

print()
# Diagnostic-B ratios — this is the headline number.
for metric_lbl in ("P(T)", "C"):
    s_prod  = pooled[(metric_lbl, "kappa")]
    s_ratio = pooled[(metric_lbl, "kappa_ratio")]
    if s_prod > 0 and not np.isnan(s_prod) and not np.isnan(s_ratio):
        factor = s_ratio / s_prod
        if factor >= 1:
            mag = factor
            winner, loser = "a·S", "a/S"
        else:
            mag = 1 / factor
            winner, loser = "a/S", "a·S"
        print(f"  Pooled-collapse ratio for {metric_lbl}  "
              f"(within-bin std of a/S over a·S)  =  {factor:.2f}")
        print(f"      → {winner} collapses {metric_lbl} {mag:.1f}× more tightly "
              f"onto a single curve than {loser}.")
print("=" * 78)
