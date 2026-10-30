#!/usr/bin/env python3
"""
Heatmap experiment for the *Polarized by Design* model (see
Rewriting_Toy_Model.pdf).  Sweeps the two free dynamical parameters

    a   ∈ [0, 0.5]   – Hebbian weight step size           (eq. 5)
    S   ∈ [1, 100]   – stubbornness (per the paper)       (§1.2)

on a *fixed* graph with a *fixed* initial opinion assignment, and records
the surviving cross-opinion edge fraction at the final step,

    f×(T) = |E×(T)| / |E|                                 (observable 1)

    |E×(T)| = edges connecting nodes with different opinions at time T
    |E|     = total edges (fixed)

    f× = 0   → consensus (one opinion absorbed the other)
    f× ≈ 0.5 → mixed / initial state (with pD = 0.5)
    f× > 0   → polarisation (both opinions survive, cross-edges remain)

Unlike P(T) = 1 − w̄×, this metric distinguishes consensus (f× ≈ 0)
from true polarisation (f× > 0) unambiguously.

The dynamics are imported from ``polarized.py`` — no re-implementation —
so this heatmap is, by construction, faithful to that interactive model.

Usage:
    python heatmap_polarized.py --quick            # 8×8, 3 runs, T=100
    python heatmap_polarized.py                    # 25×25, 20 runs  (WS + BA)
    python heatmap_polarized.py --graph ba --grid 30 --runs 10

Outputs:
    heatmap_polarized_ws.png  /  heatmap_polarized_ws_results.npz
    heatmap_polarized_ba.png  /  heatmap_polarized_ba_results.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Import dynamics from polarized.py (same directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph, step_opinions, step_weights, initialise
from polarized_vec import run_cell_vectorized


# ---------------------------------------------------------------------------
# Polarization metric  (eq. 6–7 of Rewriting_Toy_Model.pdf)
# ---------------------------------------------------------------------------

def polarization_coefficient(G, state) -> float:
    """P(t) = 1 - mean weight on cross-opinion edges."""
    cross_w = [G[u][v]["weight"] for u, v in G.edges() if state[u] != state[v]]
    w_bar = float(np.mean(cross_w)) if cross_w else 0.0
    return 1.0 - w_bar


# ---------------------------------------------------------------------------
# Per-cell worker
# ---------------------------------------------------------------------------
#
# A single worker call:
#   1. builds the graph ONCE (same graph_seed across all cells & runs),
#   2. runs ``n_runs`` independent simulations of the (a, S) cell,
#      resetting only state + weights (via ``initialise``) between runs.
#
# This avoids rebuilding the topology N_RUNS times per cell.  Each run uses
# a distinct ``sim_seed`` derived from (cell, run) so they are reproducible
# and independent.
# ---------------------------------------------------------------------------

def _cell_worker(args):
    (idx_a, idx_S, a_val, S_val,
     graph_kind, n, k, pD, w0, T, n_runs, base_seed,
     graph_seed, state_seed) = args

    G = make_graph(graph_kind, n, k, seed=graph_seed)  # build topology once

    # Cell-specific seed for the dynamical RNG; state_seed kept constant
    # across cells (same per-run initial states as the original loop did).
    cell_seed = base_seed + 1_000_003 * idx_a + 1_009 * idx_S

    P_runs = run_cell_vectorized(
        G, n_runs=n_runs, T=T, S=S_val, a=a_val, w0=w0, pD=pD,
        base_seed=cell_seed, state_seed=state_seed,
    )

    return idx_a, idx_S, float(P_runs.mean())


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def _progress(k, total, t0):
    elapsed = time.time() - t0
    eta = elapsed / k * (total - k)
    print(f"  [{k:>5}/{total}]  {elapsed:6.1f}s elapsed  ETA {eta:6.1f}s")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _log_edges(vals):
    """Compute bin edges for pcolormesh from log-spaced centres."""
    ratio = vals[1] / vals[0]
    left  = np.array([vals[0] / ratio])
    mids  = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


def plot_heatmap(grid_out, a_vals, S_vals, graph_kind, n, k, pD, w0, T, n_runs, out_png):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # pcolormesh on log-log axes — grid_out[ia, iS], so .T gives (iS, ia)
    # which maps onto (y=S, x=a) correctly.
    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), grid_out.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(
        f"$P(T) = 1 - \\bar{{w}}_\\times(T)$\n"
        f"{graph_kind.upper()},  $N$={n},  $\\langle k \\rangle$={k},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T},  runs/cell={n_runs}"
    )
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_experiment(
    graph_kind: str = "ws",
    n: int = 80,
    k: int = 8,
    pD: float = 0.5,        # paper §1.7: equal probability 1/2
    w0: float = 0.5,        # paper §1.7: w_ij(0) = w0
    T: int = 150,
    n_runs: int = 20,
    grid: int = 25,
    a_max: float = 0.5,
    S_max: float = 100.0,
    graph_seed: int = 42,
    state_seed: int = 123,
    base_seed: int = 2024,
    n_jobs=None,
    serial: bool = False,
    out_npz: str = "heatmap_polarized_results.npz",
    out_png: str = "heatmap_polarized.png",
):
    print(
        f"[setup] {graph_kind.upper()} graph  N={n}  <k>={k}  "
        f"pD={pD:.2f}  w0={w0:.2f}  graph_seed={graph_seed}"
    )

    a_vals = np.logspace(np.log10(0.005), np.log10(a_max), grid)
    S_vals = np.logspace(np.log10(1.0),   np.log10(S_max),  grid)

    print(
        f"[setup] grid {grid}×{grid},  runs/cell={n_runs},  T={T}\n"
        f"        a ∈ [{a_vals[0]:.3f}, {a_vals[-1]:.3f}]   "
        f"S ∈ [{S_vals[0]:.1f}, {S_vals[-1]:.1f}]"
    )

    job_args = [
        (ia, iS, av, Sv,
         graph_kind, n, k, pD, w0, T, n_runs, base_seed,
         graph_seed, state_seed)
        for ia, av in enumerate(a_vals)
        for iS, Sv in enumerate(S_vals)
    ]

    grid_out = np.full((grid, grid), np.nan, dtype=np.float64)
    total = len(job_args)
    t0 = time.time()

    if serial:
        for i, args in enumerate(job_args, 1):
            ia, iS, val = _cell_worker(args)
            grid_out[ia, iS] = val
            if i % max(1, total // 20) == 0 or i == total:
                _progress(i, total, t0)
    else:
        if n_jobs is None:
            # Conservative default for passively-cooled laptops (e.g. MBA):
            # cap at 4 workers and always leave ≥2 cores free for the OS.
            cores = os.cpu_count() or 2
            n_jobs = max(1, min(4, cores - 2))
        print(f"[run]   parallel  workers={n_jobs}")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_cell_worker, a) for a in job_args]
            for i, fut in enumerate(as_completed(futs), 1):
                ia, iS, val = fut.result()
                grid_out[ia, iS] = val
                if i % max(1, total // 40) == 0 or i == total:
                    _progress(i, total, t0)

    elapsed = time.time() - t0
    print(
        f"[done]  {total} cells in {elapsed:.1f}s  "
        f"({elapsed / total * 1000:.1f} ms/cell)"
    )

    np.savez(
        out_npz,
        grid=grid_out,
        a_vals=a_vals,
        S_vals=S_vals,
        graph_kind=graph_kind,
        n=n, k=k, pD=pD, w0=w0,
        T=T, n_runs=n_runs,
        graph_seed=graph_seed,
        state_seed=state_seed,
        base_seed=base_seed,
    )
    print(f"[save]  {out_npz}")

    plot_heatmap(grid_out, a_vals, S_vals, graph_kind, n, k, pD, w0, T, n_runs, out_png)
    return grid_out, a_vals, S_vals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Heatmap of f×(T) = |E×(T)|/|E| over the (a, S) plane "
                    "for the polarized.py model."
    )
    p.add_argument("--graph", choices=["ws", "er", "ba", "all"], default="all",
                   dest="graph_kind",
                   help="Graph topology. 'all' runs WS then BA (default).")
    p.add_argument("--n",    type=int,   default=80,    help="Number of nodes")
    p.add_argument("--k",    type=int,   default=8,     help="Average degree")
    p.add_argument("--pD",   type=float, default=0.5,
                   help="Initial fraction of D nodes (paper: 0.5)")
    p.add_argument("--w0",   type=float, default=0.5,   help="Initial edge weight")
    p.add_argument("--T",    type=int,   default=150,   help="Time steps per simulation")
    p.add_argument("--runs", type=int,   default=20,    dest="n_runs",
                   help="Independent runs per cell")
    p.add_argument("--grid", type=int,   default=25,    help="Grid resolution (grid × grid)")
    p.add_argument("--a-max", type=float, default=0.5,
                   help="Upper end of a-axis (default: 0.5)")
    p.add_argument("--S-max", type=float, default=100.0,
                   help="Upper end of S-axis (default: 100, per paper §1.2)")
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--state-seed", type=int, default=123)
    p.add_argument("--base-seed",  type=int, default=2024)
    p.add_argument("--jobs",   type=int,  default=None,
                   help="Parallel workers (default: min(4, cpu_count - 2))")
    p.add_argument("--serial", action="store_true",
                   help="Disable multiprocessing")
    p.add_argument("--quick", action="store_true",
                   help="Tiny 8×8 grid, 3 runs, T=100 — fast sanity check.")
    args = p.parse_args()

    if args.quick:
        args.grid    = 8
        args.n_runs  = 3
        args.T       = 100

    # Determine which topologies to run
    topologies = ["ws", "ba"] if args.graph_kind == "all" else [args.graph_kind]

    for topo in topologies:
        print(f"\n{'='*60}")
        print(f"  Running heatmap for topology: {topo.upper()}")
        print(f"{'='*60}")
        suffix = f"_quick_{topo}" if args.quick else f"_{topo}"
        run_experiment(
            graph_kind=topo,
            n=args.n, k=args.k, pD=args.pD, w0=args.w0,
            T=args.T, n_runs=args.n_runs, grid=args.grid,
            a_max=args.a_max, S_max=args.S_max,
            graph_seed=args.graph_seed,
            state_seed=args.state_seed,
            base_seed=args.base_seed,
            n_jobs=args.jobs, serial=args.serial,
            out_npz=f"heatmap_polarized_results{suffix}.npz",
            out_png=f"heatmap_polarized{suffix}.png",
        )


if __name__ == "__main__":
    main()
