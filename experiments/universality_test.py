#!/usr/bin/env python3
"""
Universality test for κ = a · S in the *Polarized by Design* model
(see Rewriting_Toy_Model.pdf and polarized.py).

Hypothesis
----------
If κ = a · S is a UNIVERSAL control parameter — a property of the
dynamics rather than the substrate — then the collapse of the final
observables onto κ should survive across:

    graph topology   ∈ { ER, WS, BA }
    mean degree      ⟨k⟩ ∈ { 4, 8, 16 }

with all other parameters fixed.

We report two diagnostics:

    P(T) = 1 - w̄_×(T)               (polarization coefficient, eq. 7)
    C    = |2 f_D(T) - 1|            (consensus index)

What this script does
---------------------
1. Sweeps 9 × 15 × 15 (graph, ⟨k⟩, a, S) cells, 8 runs/cell, on the
   same fixed (N, p_D, w_0, T) as collapse_test.py.
2. Writes universality_sweep.csv.
3. Plots:
     - universality_collapse.png  (Figure 1 — the headline)
     - degree_scaling.png         (Figure 2 — κ_½ vs ⟨k⟩)
4. Prints a console report.

No modification of polarized.py.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log10

import numpy as np

# Import dynamics from polarized.py and the shared observables helper.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph, step_opinions, step_weights, initialise
from collapse_test import final_observables


# ---------------------------------------------------------------------------
# Fixed model parameters  (match collapse_test.py)
# ---------------------------------------------------------------------------
N          = 80
PD         = 0.5
W0         = 0.5
T_STEPS    = 200
N_RUNS     = 8

GRAPH_SEED = 42
STATE_SEED = 123
BASE_SEED  = 2024

# Sweep axes
N_A = 15
N_S = 15
A_MIN, A_MAX = 0.005, 0.5
S_MIN, S_MAX = 1.0,   100.0

GRAPH_TYPES  = ("er", "ws", "ba")
MEAN_DEGREES = (4, 8, 16)

OUT_CSV     = "universality_sweep.csv"
OUT_PNG_1   = "universality_collapse.png"
OUT_PNG_2   = "degree_scaling.png"


# ---------------------------------------------------------------------------
# Per-cell worker: one (graph, k, a, S) cell  →  mean & std of P(T) and C
# ---------------------------------------------------------------------------
def _cell_worker(args):
    graph_kind, k_avg, ia, iS, a_val, S_val = args
    G = make_graph(graph_kind, N, k_avg, seed=GRAPH_SEED)

    P_runs = np.empty(N_RUNS)
    C_runs = np.empty(N_RUNS)
    for r in range(N_RUNS):
        # Distinct seeds per (cell, run); graph hash mixed in so different
        # topologies don't share RNG streams at the same (ia, iS, r).
        sim_seed = (BASE_SEED
                    + 1_000_003 * ia
                    + 1_009 * iS
                    + 17 * hash((graph_kind, k_avg)) % 999983
                    + r)
        rng = np.random.default_rng(sim_seed)
        state = initialise(G, W0, pD=PD, seed=STATE_SEED + r)
        for _ in range(T_STEPS):
            step_opinions(G, state, S_val, rng)
            step_weights(G, state, a_val)

        obs = final_observables(G, state)
        P_runs[r] = obs["P"]
        C_runs[r] = obs["consensus"]

    return (graph_kind, k_avg, ia, iS,
            float(P_runs.mean()), float(P_runs.std(ddof=1)),
            float(C_runs.mean()), float(C_runs.std(ddof=1)))


# ---------------------------------------------------------------------------
# Bin a metric by log(κ) — used both for plotting and for κ_½ extraction.
# ---------------------------------------------------------------------------
def bin_by_log_kappa(kappa, values, n_bins=20):
    log_edges = np.linspace(np.log10(kappa.min()), np.log10(kappa.max()),
                            n_bins + 1)
    idx = np.digitize(np.log10(kappa), log_edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    c, m, s = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 2:
            continue
        c.append(10 ** (0.5 * (log_edges[b] + log_edges[b + 1])))
        m.append(values[sel].mean())
        s.append(values[sel].std(ddof=1))
    return np.asarray(c), np.asarray(m), np.asarray(s)


def kappa_half(centres, means, fraction=0.5):
    """κ at which the binned curve drops below `fraction` × its leftmost value.

    Returns None if the curve never crosses that threshold within the sweep.
    """
    if centres.size < 2:
        return None
    baseline = means[0]
    target = fraction * baseline
    for i in range(1, len(means)):
        if (means[i - 1] - target) * (means[i] - target) <= 0:
            x0, x1 = np.log10(centres[i - 1]), np.log10(centres[i])
            y0, y1 = means[i - 1], means[i]
            if y1 == y0:
                return float(10 ** (0.5 * (x0 + x1)))
            xc = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(10 ** xc)
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    a_vals = np.logspace(log10(A_MIN), log10(A_MAX), N_A)
    S_vals = np.logspace(log10(S_MIN), log10(S_MAX), N_S)

    print(f"[setup] graphs={GRAPH_TYPES}  ⟨k⟩={MEAN_DEGREES}")
    print(f"[setup] N={N}  pD={PD}  w0={W0}  T={T_STEPS}  runs/cell={N_RUNS}")
    print(f"[setup] a: log-spaced {A_MIN}–{A_MAX}  ({N_A} values)")
    print(f"[setup] S: log-spaced {S_MIN}–{S_MAX}  ({N_S} values)")

    jobs = []
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            for ia, av in enumerate(a_vals):
                for iS, Sv in enumerate(S_vals):
                    jobs.append((g, k, ia, iS, av, Sv))
    total = len(jobs)
    print(f"[setup] {total} cells in total "
          f"({len(GRAPH_TYPES) * len(MEAN_DEGREES)} sweeps × {N_A*N_S} cells)")

    cores = os.cpu_count() or 2
    n_jobs = max(1, min(4, cores - 2))
    print(f"[run]   parallel workers={n_jobs}")

    # results[(graph, k)][metric] = ndarray (N_A, N_S, 2)   (mean, std)
    results = {(g, k): {"P": np.full((N_A, N_S, 2), np.nan),
                        "C": np.full((N_A, N_S, 2), np.nan)}
               for g in GRAPH_TYPES for k in MEAN_DEGREES}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_cell_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            (g, k, ia, iS, pm, ps, cm, cs) = fut.result()
            results[(g, k)]["P"][ia, iS, 0] = pm
            results[(g, k)]["P"][ia, iS, 1] = ps
            results[(g, k)]["C"][ia, iS, 0] = cm
            results[(g, k)]["C"][ia, iS, 1] = cs
            if i % max(1, total // 30) == 0 or i == total:
                el = time.time() - t0
                eta = el / i * (total - i)
                print(f"  [{i:>5}/{total}]  {el:6.1f}s elapsed  ETA {eta:6.1f}s")

    elapsed = time.time() - t0
    print(f"[done]  {total} cells in {elapsed:.1f}s "
          f"({elapsed/total*1000:.1f} ms/cell)")

    # -------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------
    rows = []
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            R = results[(g, k)]
            for ia, av in enumerate(a_vals):
                for iS, Sv in enumerate(S_vals):
                    rows.append({
                        "graph_type":  g,
                        "mean_degree": int(k),
                        "a":           float(av),
                        "S":           float(Sv),
                        "kappa":       float(av * Sv),
                        "P_mean":      float(R["P"][ia, iS, 0]),
                        "P_std":       float(R["P"][ia, iS, 1]),
                        "C_mean":      float(R["C"][ia, iS, 0]),
                        "C_std":       float(R["C"][ia, iS, 1]),
                        "n_runs":      int(N_RUNS),
                    })
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save]  {OUT_CSV}  ({len(rows)} rows)")

    # -------------------------------------------------------------------
    # Compute binned curves per (graph, k, metric)
    # -------------------------------------------------------------------
    binned = {}
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            R = results[(g, k)]
            kappa_arr = np.array([av * Sv for av in a_vals for Sv in S_vals])
            P_flat = R["P"][..., 0].ravel()
            C_flat = R["C"][..., 0].ravel()
            binned[(g, k, "P")] = bin_by_log_kappa(kappa_arr, P_flat)
            binned[(g, k, "C")] = bin_by_log_kappa(kappa_arr, C_flat)

    # -------------------------------------------------------------------
    # Figure 1: universality_collapse.png
    # -------------------------------------------------------------------
    import matplotlib.pyplot as plt

    colour = {"er": "C0", "ws": "C1", "ba": "C2"}
    label  = {"er": "ER", "ws": "WS", "ba": "BA"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    metric_specs = [
        ("P", r"$P(T) = 1 - \bar{w}_\times(T)$",  "polarization coefficient"),
        ("C", r"$C = |2 f_D(T) - 1|$",             "consensus diagnostic"),
    ]
    for row_i, (metric_key, ylabel, sub_title) in enumerate(metric_specs):
        for col_i, k in enumerate(MEAN_DEGREES):
            ax = axes[row_i, col_i]
            for g in GRAPH_TYPES:
                cb_x, cb_m, cb_s = binned[(g, k, metric_key)]
                if cb_x.size == 0:
                    continue
                ax.plot(cb_x, cb_m, "-", color=colour[g], lw=2,
                        label=label[g])
                ax.fill_between(cb_x, cb_m - cb_s, cb_m + cb_s,
                                color=colour[g], alpha=0.20)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.05)
            ax.axvline(W0**2 / k, ls="--", color="red", lw=1,
                       label=fr"$w_0^2/\langle k\rangle = {W0**2/k:.3f}$")
            ax.axhline(0.5, ls=":", color="grey", lw=0.7)
            if row_i == 0:
                ax.set_title(fr"$\langle k\rangle = {k}$  —  {sub_title}",
                             fontsize=10)
            else:
                ax.set_title(sub_title, fontsize=10)
            if row_i == 1:
                ax.set_xlabel(r"$\kappa = a \cdot S$")
            if col_i == 0:
                ax.set_ylabel(ylabel)
            ax.legend(loc="lower left", fontsize=8)
            ax.grid(True, which="both", alpha=0.25, lw=0.4)

    fig.suptitle(
        r"Universality test:  does $\kappa = a \cdot S$ collapse across "
        r"topology and mean degree?",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG_1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save]  {OUT_PNG_1}")

    # -------------------------------------------------------------------
    # Figure 2: degree_scaling.png  +  exponent fits
    # -------------------------------------------------------------------
    kappa_half_table = {}     # (graph, k)  → κ_½
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            cb_x, cb_m, _ = binned[(g, k, "P")]
            kappa_half_table[(g, k)] = kappa_half(cb_x, cb_m, fraction=0.5)

    fitted_alpha = {}
    fitted_beta  = {}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    k_dense = np.array(MEAN_DEGREES, dtype=float)
    for g in GRAPH_TYPES:
        ys = np.array([kappa_half_table[(g, k)] for k in MEAN_DEGREES],
                      dtype=float)
        mask = np.isfinite(ys)
        if mask.sum() >= 2:
            log_k = np.log10(k_dense[mask])
            log_y = np.log10(ys[mask])
            alpha, beta = np.polyfit(log_k, log_y, 1)
            fitted_alpha[g] = float(alpha)
            fitted_beta[g]  = float(10**beta)
            label_str = (f"{label[g]}  fit: "
                         fr"$\kappa_{{1/2}} \propto \langle k\rangle^{{{alpha:+.2f}}}$")
        else:
            fitted_alpha[g] = float("nan")
            fitted_beta[g]  = float("nan")
            label_str = f"{label[g]}  (insufficient crossings)"

        ax.plot(k_dense[mask], ys[mask], "o-", color=colour[g], lw=2,
                ms=8, label=label_str)
        # mark cells where κ_½ didn't cross
        if (~mask).any():
            ax.plot(k_dense[~mask], np.full((~mask).sum(), np.nan),
                    "x", color=colour[g], ms=12)

    # Heuristic reference: κ* = w0^2 / ⟨k⟩
    k_fine = np.logspace(np.log10(min(MEAN_DEGREES) * 0.8),
                         np.log10(max(MEAN_DEGREES) * 1.25), 50)
    ax.plot(k_fine, W0**2 / k_fine, "k--", lw=1.5,
            label=fr"heuristic  $\kappa^* = w_0^2/\langle k\rangle$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\langle k \rangle$")
    ax.set_ylabel(r"$\kappa_{1/2}$  (κ at which $P(T)$ has halved)")
    ax.set_title(r"Degree scaling of the polarization threshold $\kappa_{1/2}$")
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG_2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save]  {OUT_PNG_2}")

    # -------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------
    print()
    print("=" * 76)
    print("  UNIVERSALITY TEST — summary")
    print("=" * 76)

    # (i) Within-subplot collapse quality per ⟨k⟩
    #     Compare graph-type binned curves at the same ⟨k⟩, metric=P(T).
    print()
    print("  Within-subplot collapse quality  (cross-topology spread of P(T))")
    print(f"  {'⟨k⟩':<6}{'median bin-std':>18}{'worst bin-std':>18}{'verdict':>22}")
    overall_verdict_tight = True
    overall_verdict_loose = False
    for k in MEAN_DEGREES:
        # On each κ-bin, compute std across the three graph types' means.
        per_bin_stds = []
        # use the WS binning grid as the reference (all three should be similar)
        ref_x, _, _ = binned[("ws", k, "P")]
        for x_target in ref_x:
            means_here = []
            for g in GRAPH_TYPES:
                cb_x, cb_m, _ = binned[(g, k, "P")]
                # interp at x_target on log scale
                if cb_x.size and cb_x.min() <= x_target <= cb_x.max():
                    val = np.interp(np.log10(x_target),
                                    np.log10(cb_x), cb_m)
                    means_here.append(val)
            if len(means_here) >= 2:
                per_bin_stds.append(np.std(means_here, ddof=1))
        if per_bin_stds:
            med = float(np.median(per_bin_stds))
            wrs = float(np.max(per_bin_stds))
            if med < 0.05:
                v = "TIGHT"
            elif med < 0.12:
                v = "MODERATE"
                overall_verdict_tight = False
            else:
                v = "POOR"
                overall_verdict_tight = False
                overall_verdict_loose = True
            print(f"  {k:<6}{med:>18.3f}{wrs:>18.3f}{v:>22}")

    # (ii) κ_½ table + fitted exponents
    print()
    print("  Empirical κ_½ (κ at which the binned P(T) has halved):")
    print(f"  {'graph':<8}" + "".join(f"{f'⟨k⟩={k}':>12}" for k in MEAN_DEGREES)
          + f"{'fitted α':>14}")
    for g in GRAPH_TYPES:
        row = f"  {label[g]:<8}"
        for k in MEAN_DEGREES:
            v = kappa_half_table[(g, k)]
            row += "         —  " if v is None else f"{v:>12.3f}"
        a = fitted_alpha.get(g, float("nan"))
        row += "       n/a   " if np.isnan(a) else f"{a:>+14.3f}"
        print(row)
    print(f"   (heuristic predicts κ_½ ∝ ⟨k⟩^-1, i.e. α = -1.000)")

    # (iii) One-sentence verdict
    print()
    fit_alphas = [a for a in fitted_alpha.values() if not np.isnan(a)]
    if not fit_alphas:
        print("  VERDICT:  κ_½ never crossed in the sweep — extend the κ range.")
    else:
        mean_alpha = float(np.mean(fit_alphas))
        max_dev    = float(np.max(np.abs(np.array(fit_alphas) - mean_alpha)))
        if overall_verdict_loose:
            print("  VERDICT:  κ does NOT collapse across topology — the "
                  "graph-type curves separate by more than 0.12 in std.")
        elif overall_verdict_tight and abs(mean_alpha + 1.0) < 0.20:
            print(f"  VERDICT:  κ is UNIVERSAL up to a 1/⟨k⟩ correction "
                  f"(mean fitted α = {mean_alpha:+.2f}, heuristic α = -1).")
        elif overall_verdict_tight:
            print(f"  VERDICT:  κ collapses across topology, but the degree "
                  f"correction differs from 1/⟨k⟩  "
                  f"(mean fitted α = {mean_alpha:+.2f}, "
                  f"max graph-to-graph deviation = {max_dev:.2f}).")
        else:
            print(f"  VERDICT:  κ collapse is APPROXIMATE  "
                  f"(mean fitted α = {mean_alpha:+.2f}, "
                  f"max deviation = {max_dev:.2f}).")

    # (iv) Failures
    n_nan_P = sum(int(np.isnan(results[(g, k)]["P"][..., 0]).sum())
                  for g in GRAPH_TYPES for k in MEAN_DEGREES)
    n_nan_C = sum(int(np.isnan(results[(g, k)]["C"][..., 0]).sum())
                  for g in GRAPH_TYPES for k in MEAN_DEGREES)
    print()
    print(f"  cells with NaN in P(T): {n_nan_P}")
    print(f"  cells with NaN in C   : {n_nan_C}")
    print("=" * 76)


if __name__ == "__main__":
    main()
