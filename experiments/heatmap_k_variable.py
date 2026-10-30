#!/usr/bin/env python3
"""
Heatmap over the (a, S) plane for several fixed average degrees ⟨k⟩.

Sweeps the two free dynamical parameters:
    a  ∈ [0.005, 0.5]   – Hebbian weight step    (log-spaced)
    S  ∈ [1,    100]    – stubbornness             (log-spaced)

with ⟨k⟩ fixed at several values (4, 8, 16) to show how the
polarisation boundary shifts with network connectivity.

Observable:  P(T) = 1 − w̄_×(T)   (polarization coefficient, eq. 7)

Usage:
    python heatmap_k_variable.py             # WS + BA, k = 4, 8, 16
    python heatmap_k_variable.py --graph ws  # WS only
    python heatmap_k_variable.py --quick     # fast sanity check (8×8, 3 runs)

Outputs (one PNG + NPZ per topology × ⟨k⟩ combination):
    heatmap_k_variable_ws_k004.png / .npz
    heatmap_k_variable_ws_k008.png / .npz
    heatmap_k_variable_ws_k016.png / .npz
    (same for ba, er)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log10

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph, step_opinions, step_weights, initialise
from polarized_vec import run_cell_vectorized


# ---------------------------------------------------------------------------
# Observable
# ---------------------------------------------------------------------------

def polarization_P(G, state) -> float:
    """P(T) = 1 - w̄_×(T)  — polarization coefficient (eq. 7).

    w̄_× is the mean edge weight over cross-opinion edges at time T.
    If there are no cross-opinion edges, w̄_× = 0 → P = 1.
    """
    cross_w = [G[u][v]["weight"] for u, v in G.edges() if state[u] != state[v]]
    if not cross_w:
        return 1.0
    return 1.0 - float(np.mean(cross_w))


# ---------------------------------------------------------------------------
# Per-cell worker
# ---------------------------------------------------------------------------

def _cell_worker(job):
    (idx_a, idx_S, a_val, S_val,
     graph_kind, n, k_avg, pD, w0, T, n_runs, base_seed,
     graph_seed, state_seed) = job

    G = make_graph(graph_kind, n, k_avg, seed=graph_seed)

    # Cell-specific seed for the dynamical RNG.
    cell_seed = base_seed + 1_000_003 * idx_a + 1_009 * idx_S

    P_runs = run_cell_vectorized(
        G, n_runs=n_runs, T=T, S=S_val, a=a_val, w0=w0, pD=pD,
        base_seed=cell_seed, state_seed=state_seed,
    )

    # Mean for the heatmap, sample variance (ddof=1) for the susceptibility map.
    # Variance is np.nan when n_runs < 2 (cannot estimate spread from 1 sample).
    if n_runs >= 2:
        var = float(P_runs.var(ddof=1))
    else:
        var = float("nan")
    return idx_a, idx_S, float(P_runs.mean()), var


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _progress(done, total, t0):
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done)
    print(f"  [{done:>5}/{total}]  {elapsed:6.1f}s elapsed  ETA {eta:6.1f}s")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _log_edges(vals):
    """Compute bin edges for pcolormesh from log-spaced centres."""
    ratio = vals[1] / vals[0]
    left  = np.array([vals[0] / ratio])
    mids  = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


def plot_heatmap(grid_out, a_vals, S_vals, graph_kind, n, k_avg,
                 pD, w0, T, n_runs, out_png):
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
        f"$P(T) = 1 - \\bar{{w}}_\\times(T)$     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{graph_kind.upper()},  $N$={n},  "
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
    k_avg: int = 8,
    pD: float = 0.5,
    w0: float = 0.5,
    T: int = 1000,
    n_runs: int = 20,
    grid: int = 25,
    a_min: float = 0.005,
    a_max: float = 0.5,
    S_min: float = 1.0,
    S_max: float = 100.0,
    graph_seed: int = 42,
    state_seed: int = 123,
    base_seed: int = 2024,
    n_jobs=None,
    serial: bool = False,
    out_npz: str = "heatmap_k_variable_results.npz",
    out_png: str = "heatmap_k_variable.png",
):
    print(
        f"[setup] {graph_kind.upper()} graph  N={n}  ⟨k⟩={k_avg}  "
        f"pD={pD:.2f}  w0={w0:.2f}  graph_seed={graph_seed}"
    )

    a_vals = np.logspace(log10(a_min), log10(a_max), grid)
    S_vals = np.logspace(log10(S_min), log10(S_max), grid)

    print(
        f"[setup] grid {grid}×{grid},  runs/cell={n_runs},  T={T}\n"
        f"        a ∈ [{a_vals[0]:.4f}, {a_vals[-1]:.4f}]   "
        f"S ∈ [{S_vals[0]:.1f}, {S_vals[-1]:.1f}]  (log-spaced)"
    )

    job_args = [
        (ia, iS, av, Sv,
         graph_kind, n, k_avg, pD, w0, T, n_runs, base_seed,
         graph_seed, state_seed)
        for ia, av in enumerate(a_vals)
        for iS, Sv in enumerate(S_vals)
    ]

    grid_out = np.full((grid, grid), np.nan, dtype=np.float64)  # mean   P,  [ia, iS]
    grid_var = np.full((grid, grid), np.nan, dtype=np.float64)  # var    P,  [ia, iS]
    total = len(job_args)
    t0 = time.time()

    if serial:
        for i, job in enumerate(job_args, 1):
            ia, iS, val, var = _cell_worker(job)
            grid_out[ia, iS] = val
            grid_var[ia, iS] = var
            if i % max(1, total // 20) == 0 or i == total:
                _progress(i, total, t0)
    else:
        if n_jobs is None:
            cores = os.cpu_count() or 2
            n_jobs = max(1, min(4, cores - 2))
        print(f"[run]   parallel  workers={n_jobs}")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_cell_worker, job) for job in job_args]
            for i, fut in enumerate(as_completed(futs), 1):
                ia, iS, val, var = fut.result()
                grid_out[ia, iS] = val
                grid_var[ia, iS] = var
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
        grid_var=grid_var,
        a_vals=a_vals,
        S_vals=S_vals,
        graph_kind=graph_kind,
        n=n, k_avg=k_avg, pD=pD, w0=w0,
        T=T, n_runs=n_runs,
        graph_seed=graph_seed,
        state_seed=state_seed,
        base_seed=base_seed,
    )
    print(f"[save]  {out_npz}")

    plot_heatmap(grid_out, a_vals, S_vals, graph_kind, n, k_avg,
                 pD, w0, T, n_runs, out_png)
    return grid_out, a_vals, S_vals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Heatmap of P(T) over the (a, S) plane for fixed ⟨k⟩ values."
    )
    p.add_argument("--graph", choices=["ws", "er", "ba", "all"], default="all",
                   dest="graph_kind",
                   help="Graph topology. 'all' runs WS then BA (default).")
    p.add_argument("--n",    type=int,   default=80,   help="Number of nodes")
    p.add_argument("--pD",   type=float, default=0.5,  help="Initial fraction of D nodes")
    p.add_argument("--w0",   type=float, default=0.5,  help="Initial edge weight")
    p.add_argument("--T",    type=int,   default=1000, help="Time steps per simulation")
    p.add_argument("--runs", type=int,   default=20,   dest="n_runs",
                   help="Independent runs per cell")
    p.add_argument("--grid", type=int,   default=25,   help="Grid resolution (grid × grid)")
    p.add_argument("--a-min",  type=float, default=0.005,  help="Min a (log axis)")
    p.add_argument("--a-max",  type=float, default=0.5,    help="Max a (log axis)")
    p.add_argument("--S-min",  type=float, default=1.0,    help="Min S (log axis)")
    p.add_argument("--S-max",  type=float, default=100.0,  help="Max S (log axis)")
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16],
                   help="⟨k⟩ values to run (default: 4 8 16)")
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--state-seed", type=int, default=123)
    p.add_argument("--base-seed",  type=int, default=2024)
    p.add_argument("--jobs",   type=int,  default=None,
                   help="Parallel workers (default: min(4, cpu_count - 2))")
    p.add_argument("--serial", action="store_true", help="Disable multiprocessing")
    p.add_argument("--quick",  action="store_true",
                   help="8×8 grid, 3 runs, T=100 — fast sanity check.")
    args = p.parse_args()

    if args.quick:
        args.grid   = 8
        args.n_runs = 3
        args.T      = 100

    topologies = ["ws", "ba"] if args.graph_kind == "all" else [args.graph_kind]

    for topo in topologies:
        for k_val in args.k_values:
            k_tag  = f"k{k_val:03d}"
            suffix = f"_quick_{topo}_{k_tag}" if args.quick else f"_{topo}_{k_tag}"
            print(f"\n{'='*60}")
            print(f"  topology={topo.upper()}   ⟨k⟩={k_val}")
            print(f"{'='*60}")
            run_experiment(
                graph_kind=topo,
                n=args.n, k_avg=k_val, pD=args.pD, w0=args.w0,
                T=args.T, n_runs=args.n_runs, grid=args.grid,
                a_min=args.a_min, a_max=args.a_max,
                S_min=args.S_min, S_max=args.S_max,
                graph_seed=args.graph_seed,
                state_seed=args.state_seed,
                base_seed=args.base_seed,
                n_jobs=args.jobs, serial=args.serial,
                out_npz=f"heatmap_k_variable_results{suffix}.npz",
                out_png=f"heatmap_k_variable{suffix}.png",
            )


if __name__ == "__main__":
    main()
