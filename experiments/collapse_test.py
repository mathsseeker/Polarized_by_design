#!/usr/bin/env python3
"""
Data-collapse test for the *Polarized by Design* model
(see Rewriting_Toy_Model.pdf and polarized.py).

Hypothesis
----------
The polarization transition is governed by the single dimensionless group

        κ  =  a · S

where  a  is the Hebbian step size (eq. 5) and  S  the stubbornness (§1.2).
If true, the final polarization

        P(T) = 1 - w̄_×(T)                      (eq. 7)

should collapse onto a single curve when plotted against κ, irrespective
of the particular (a, S) pair that produced it.

What this script does
---------------------
1. Sweeps log-spaced (a, S) on a 15 × 15 grid.
2. For each pair, runs 8 independent simulations of the dynamics imported
   from polarized.py — no modification of that file.
3. Writes a CSV  (a, S, κ, P_mean, P_std, n_runs).
4. Renders a two-panel data-collapse figure:
       (a)  scatter of P_mean vs κ, log-x, coloured by S, with ± std bars.
       (b)  the same scatter binned into log-spaced κ bins, line + band.
   A vertical dashed line at κ* = w0²/2 = 0.125 marks the heuristic
   prediction for the transition midpoint.
5. Prints a short qualitative report.

Fixed parameters (same as the existing heatmap):
    graph = Watts–Strogatz,  N = 80,  ⟨k⟩ = 8,
    p_D   = 0.5,  w_0 = 0.5,  T = 200
"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log10

import numpy as np

# Import the dynamics from polarized.py (no modifications)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph, step_opinions, step_weights, initialise


# ---------------------------------------------------------------------------
# Fixed model parameters
# ---------------------------------------------------------------------------
GRAPH_KIND = "ws"
N          = 80
K_AVG      = 8
PD         = 0.5
W0         = 0.5
T_STEPS    = 200
N_RUNS     = 8

GRAPH_SEED = 42
STATE_SEED = 123
BASE_SEED  = 2024

# Sweep axes (log-spaced)
N_A = 15
N_S = 15
A_MIN, A_MAX = 0.005, 0.5
S_MIN, S_MAX = 1.0,   100.0

OUT_CSV = "collapse_test.csv"
OUT_PNG = "collapse_test.png"


# ---------------------------------------------------------------------------
# Final-state observables
# ---------------------------------------------------------------------------
#   P            = 1 - mean weight on cross-opinion edges          (eq. 7)
#   f_E_cross    = |E_×| / |E|        — surviving cross-edge fraction
#   consensus    = |2·f_D - 1|        — 0 = perfect 50/50, 1 = single opinion
#   w_x_density  = (Σ w on E_×) / |E| — total cross-trust per edge
#
# Together these tell us which mechanism produced a given P:
#   high P + high consensus     →  P = 1 by HOMOGENISATION
#   high P + low  consensus     →  P = 1 by FRAGMENTATION
# ---------------------------------------------------------------------------
def final_observables(G, state):
    n_edges = G.number_of_edges()
    cross_w = [G[u][v]["weight"] for u, v in G.edges() if state[u] != state[v]]
    n_cross = len(cross_w)
    w_bar = float(np.mean(cross_w)) if cross_w else 0.0

    f_D = float(np.mean(state))
    return {
        "P":           1.0 - w_bar,
        "f_E_cross":   n_cross / n_edges if n_edges else 0.0,
        "consensus":   abs(2.0 * f_D - 1.0),
        "w_x_density": (sum(cross_w) / n_edges) if n_edges else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-cell worker:  one (a, S) cell → mean & std of P over n_runs sims.
# Builds the graph once per worker call (graph topology is fixed across
# cells); only opinions + weights are reset between runs.
# ---------------------------------------------------------------------------
METRIC_KEYS = ("P", "f_E_cross", "consensus", "w_x_density")


def _cell_worker(args):
    ia, iS, a_val, S_val = args
    G = make_graph(GRAPH_KIND, N, K_AVG, seed=GRAPH_SEED)

    runs = {key: np.empty(N_RUNS) for key in METRIC_KEYS}
    for r in range(N_RUNS):
        sim_seed = BASE_SEED + 1_000_003 * ia + 1_009 * iS + r
        rng = np.random.default_rng(sim_seed)
        state = initialise(G, W0, pD=PD, seed=STATE_SEED + r)
        for _ in range(T_STEPS):
            step_opinions(G, state, S_val, rng)
            step_weights(G, state, a_val)

        obs = final_observables(G, state)
        for key in METRIC_KEYS:
            runs[key][r] = obs[key]

    means = {f"{k}_mean": float(runs[k].mean()) for k in METRIC_KEYS}
    stds  = {f"{k}_std":  float(runs[k].std(ddof=1)) for k in METRIC_KEYS}
    return ia, iS, {**means, **stds}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    a_vals = np.logspace(log10(A_MIN), log10(A_MAX), N_A)
    S_vals = np.logspace(log10(S_MIN), log10(S_MAX), N_S)

    print(f"[setup] WS graph  N={N}  <k>={K_AVG}  pD={PD}  w0={W0}  T={T_STEPS}")
    print(f"[setup] a:  log-spaced  {A_MIN}–{A_MAX}  ({N_A} values)")
    print(f"[setup] S:  log-spaced  {S_MIN}–{S_MAX}  ({N_S} values)")
    print(f"[setup] runs/cell = {N_RUNS}")

    jobs = [(ia, iS, av, Sv)
            for ia, av in enumerate(a_vals)
            for iS, Sv in enumerate(S_vals)]
    total = len(jobs)

    cores = os.cpu_count() or 2
    n_jobs = max(1, min(4, cores - 2))
    print(f"[run]   parallel  workers={n_jobs}")

    # axis 2: index by metric name → (mean, std)
    results = {k: np.full((N_A, N_S, 2), np.nan) for k in METRIC_KEYS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_cell_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            ia, iS, cell = fut.result()
            for k in METRIC_KEYS:
                results[k][ia, iS, 0] = cell[f"{k}_mean"]
                results[k][ia, iS, 1] = cell[f"{k}_std"]
            if i % max(1, total // 20) == 0 or i == total:
                el = time.time() - t0
                eta = el / i * (total - i)
                print(f"  [{i:>4}/{total}]  {el:5.1f}s elapsed  ETA {eta:5.1f}s")

    elapsed = time.time() - t0
    print(f"[done]  {total} cells in {elapsed:.1f}s "
          f"({elapsed/total*1000:.1f} ms/cell)")

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    rows = []
    for ia, av in enumerate(a_vals):
        for iS, Sv in enumerate(S_vals):
            row = {
                "a":      float(av),
                "S":      float(Sv),
                "kappa":  float(av * Sv),
                "n_runs": int(N_RUNS),
            }
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = float(results[k][ia, iS, 0])
                row[f"{k}_std"]  = float(results[k][ia, iS, 1])
            rows.append(row)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save]  {OUT_CSV}  ({len(rows)} rows)")

    # -------------------------------------------------------------------
    # Plot: four-panel diagnostic collapse
    #
    #   Top row    – raw scatter (coloured by S) for the three metrics
    #                that distinguish P=1-by-consensus from P=1-by-
    #                fragmentation.
    #   Bottom-right panel – the three binned curves overlaid, so we can
    #                see whether they cross or move together as κ grows.
    # -------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    kappa = np.array([r["kappa"] for r in rows])
    S_arr = np.array([r["S"]     for r in rows])
    metric_arr = {
        k: np.array([r[f"{k}_mean"] for r in rows]) for k in METRIC_KEYS
    }
    metric_std = {
        k: np.array([r[f"{k}_std"]  for r in rows]) for k in METRIC_KEYS
    }

    kappa_star = W0**2 / 2.0  # = 0.125 with w0=0.5

    # --- compute binned curves for every metric ---
    n_bins = 20
    log_edges = np.linspace(np.log10(kappa.min()), np.log10(kappa.max()),
                            n_bins + 1)
    bin_idx = np.digitize(np.log10(kappa), log_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    def bin_metric(values):
        c, m, s = [], [], []
        for b in range(n_bins):
            sel = bin_idx == b
            if sel.sum() < 2:
                continue
            c.append(10 ** (0.5 * (log_edges[b] + log_edges[b + 1])))
            m.append(values[sel].mean())
            s.append(values[sel].std(ddof=1))
        return np.asarray(c), np.asarray(m), np.asarray(s)

    binned = {k: bin_metric(metric_arr[k]) for k in METRIC_KEYS}

    # --- figure layout ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_specs = [
        ("P",            "$P(T) = 1 - \\bar w_\\times(T)$",
                          "(a)  Polarization $P$",                    axes[0, 0]),
        ("consensus",    "$|2 f_D - 1|$  (0 = 50/50, 1 = consensus)",
                          "(b)  Opinion-asymmetry / consensus index",  axes[0, 1]),
        ("w_x_density",  "$\\Sigma w_{\\times} / |E|$  (cross-trust per edge)",
                          "(c)  Cross-edge weight density",            axes[1, 0]),
    ]

    for key, ylabel, title, ax in panel_specs:
        ys  = metric_arr[key]
        yer = metric_std[key]
        sc = ax.scatter(
            kappa, ys, c=S_arr, cmap="viridis",
            norm=LogNorm(vmin=S_arr.min(), vmax=S_arr.max()),
            s=28, edgecolor="k", linewidth=0.3, zorder=3,
        )
        ax.errorbar(kappa, ys, yerr=yer, fmt="none",
                    ecolor="grey", alpha=0.4, lw=0.7, capsize=0, zorder=2)
        # overlay the binned curve in white-on-black for visibility
        cb_x, cb_m, _ = binned[key]
        ax.plot(cb_x, cb_m, color="black", lw=2.0, zorder=4)
        ax.plot(cb_x, cb_m, color="white", lw=0.8, zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa = a \cdot S$")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.axvline(kappa_star, ls="--", color="red", lw=1,
                   label=fr"$\kappa^* = w_0^2/2 = {kappa_star:.3f}$")
        ax.axhline(0.5, ls=":", color="grey", lw=0.7)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(r"$S$  (log)")

    # --- panel (d): all three binned curves overlaid ---
    axd = axes[1, 1]
    overlay_specs = [
        ("P",           "$P(T)$",                         "C0"),
        ("consensus",   "$|2f_D - 1|$ (consensus)",       "C2"),
        ("w_x_density", "$\\Sigma w_\\times/|E|$",        "C3"),
    ]
    for key, lbl, col in overlay_specs:
        cb_x, cb_m, cb_s = binned[key]
        axd.plot(cb_x, cb_m, "-", color=col, lw=2, label=lbl)
        axd.fill_between(cb_x, cb_m - cb_s, cb_m + cb_s,
                         color=col, alpha=0.2)
    axd.set_xscale("log")
    axd.set_xlabel(r"$\kappa = a \cdot S$")
    axd.set_ylabel("binned metric value")
    axd.set_ylim(-0.02, 1.05)
    axd.axvline(kappa_star, ls="--", color="red", lw=1,
                label=fr"$\kappa^* = {kappa_star:.3f}$")
    axd.axhline(0.5, ls=":", color="grey", lw=0.7)
    axd.set_title("(d)  All three metrics overlaid (binned)")
    axd.legend(loc="center left", fontsize=8)

    fig.suptitle(
        r"Data collapse test: does $\kappa = a \cdot S$ control polarization?"
        "\n"
        r"$P=1$ via consensus  vs  $P=1$ via fragmentation",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save]  {OUT_PNG}")

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    def midpoint_kappa(centres, means, target=0.5):
        """Linear interp of κ at which the binned curve crosses `target`."""
        for i in range(1, len(means)):
            if (means[i - 1] - target) * (means[i] - target) < 0:
                x0, x1 = np.log10(centres[i - 1]), np.log10(centres[i])
                y0, y1 = means[i - 1], means[i]
                xc = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
                return 10 ** xc
        return None

    print()
    print("=" * 72)
    print("  COLLAPSE TEST  —  summary")
    print("=" * 72)

    # κ-bin spread per metric  ⇒ collapse tightness
    print(f"  {'metric':<14}{'median bin-std':>16}{'worst bin-std':>16}{'verdict':>22}")
    for key in METRIC_KEYS:
        _, _, bs = binned[key]
        bs = bs[~np.isnan(bs)]
        if not bs.size:
            continue
        med = float(np.median(bs))
        wrs = float(bs.max())
        if med < 0.05:
            v = "TIGHT"
        elif med < 0.12:
            v = "MODERATE"
        else:
            v = "LOOSE"
        print(f"  {key:<14}{med:>16.3f}{wrs:>16.3f}{v:>22}")

    # Transition midpoint(s)
    print()
    print(f"  heuristic κ* = w0² / 2 = {kappa_star:.3f}")
    for key in METRIC_KEYS:
        cb_x, cb_m, _ = binned[key]
        if cb_m.size < 2:
            continue
        mid = midpoint_kappa(cb_x, cb_m, 0.5)
        if mid is None:
            extreme = "above" if cb_m.min() > 0.5 else "below"
            print(f"  κ at {key}_mean ≈ 0.5:   not crossed (curve stays {extreme})")
        else:
            print(f"  κ at {key}_mean ≈ 0.5:   {mid:.3f}")

    # Mechanism diagnostic at the high-κ end:
    # if consensus is high near the transition, P=1 is reached by
    # homogenisation; if consensus is low but f_E_cross is also low,
    # P=1 is reached by fragmentation.
    high_kappa_mask = kappa > kappa.max() / 10  # top decade
    if high_kappa_mask.any():
        cons_high = metric_arr["consensus"][high_kappa_mask].mean()
        fE_high   = metric_arr["f_E_cross"][high_kappa_mask].mean()
        P_high    = metric_arr["P"][high_kappa_mask].mean()
        print()
        print("  Top κ-decade averages:")
        print(f"     P             = {P_high:.3f}")
        print(f"     |E_x|/|E|     = {fE_high:.3f}")
        print(f"     consensus idx = {cons_high:.3f}")
        if cons_high > 0.7:
            print("  → high-κ regime is dominated by CONSENSUS "
                  "(one opinion takes over).")
        elif fE_high < 0.3:
            print("  → high-κ regime is dominated by FRAGMENTATION "
                  "(two camps, weak ties).")
        else:
            print("  → high-κ regime shows a mix of both mechanisms.")

    # Boundary / failure cells
    n_nan = int(np.isnan(results["P"][..., 0]).sum())
    a_at_bound = (a_vals > 1).sum() + (a_vals <= 0).sum()
    S_at_bound = (S_vals < 1).sum()
    print()
    print(f"  cells with NaN P_mean:                 {n_nan}")
    print(f"  cells with a outside (0, 1]:           {int(a_at_bound)}")
    print(f"  cells with S < 1 (out of paper range): {int(S_at_bound)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
