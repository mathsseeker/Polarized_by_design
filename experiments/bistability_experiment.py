#!/usr/bin/env python3
"""
bistability_experiment.py
─────────────────────────
Extends heatmap_polarized.py with two new observables that decompose
the model's uncertainty into orthogonal components:

    σ_runs(a, S) — std of P(T) ACROSS independent runs at the same (a, S)
                   → identifies BISTABLE cells (same params, opposite fates)
    σ_time(a, S) — std of P(t) WITHIN a run, over the last `tail_frac` of T
                   → identifies CHAOTIC cells (dynamics never settles)

It then auto-picks the cell with maximum σ_runs (restricted to non-trivial
mean P) and produces:

    • Broom plot   — 50 P(t) trajectories at that cell
    • Histogram    — distribution of P_final at that cell
    • Verdict      — quantitative bimodality check

Drop this file next to polarized.py and heatmap_polarized.py.

Outputs (default prefix = "bistab"):
    bistab_heatmaps.png   — triple panel <P>, σ_runs, σ_time (+ ridge contour)
    bistab_broom.png      — 50 independent P(t) traces at the bistable cell
    bistab_hist.png       — histogram of P_final at the same cell
    bistab_results.npz    — all grids + raw P_finals_all for reproducibility
    bistab_summary.txt    — text verdict (bimodal? holds the narrative?)

Usage:
    python bistability_experiment.py --quick           # 12x12, 15 runs (~30 s)
    python bistability_experiment.py                   # 30x30, 40 runs (default)
    python bistability_experiment.py --topo ba         # change topology
    python bistability_experiment.py --cell 0.025,8    # force broom cell
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Import dynamics from src/ (sibling of experiments/).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph, step_opinions, step_weights, initialise
from polarized_fast import (
    initialise_fast,
    step_opinions_fast,
    step_weights_fast,
    polarization_coefficient_fast,
)


# ────────────────────────────────────────────────────────────────────────
# Observable
# ────────────────────────────────────────────────────────────────────────

def polarization_coefficient(G, state) -> float:
    """P(t) = 1 - mean weight on cross-opinion edges (eq. 6-7 of paper)."""
    cross_w = [G[u][v]["weight"] for u, v in G.edges() if state[u] != state[v]]
    w_bar = float(np.mean(cross_w)) if cross_w else 0.0
    return 1.0 - w_bar


# ────────────────────────────────────────────────────────────────────────
# Per-cell worker
# ────────────────────────────────────────────────────────────────────────
# A single call:
#   1. builds the graph once,
#   2. runs n_runs independent simulations with different initial opinions,
#   3. records P_final and the within-run std of P over the last tail_frac of T,
#   4. returns mean_P, sigma_runs, sigma_time AND the raw P_final array.

def _cell_worker(args):
    (idx_a, idx_S, a_val, S_val,
     graph_kind, n, k, pD, w0, T, n_runs, tail_frac, base_seed,
     graph_seed, state_seed) = args

    G = make_graph(graph_kind, n, k, seed=graph_seed)
    tail_len = max(2, int(tail_frac * T))

    P_final = np.empty(n_runs)
    sigma_time_per_run = np.empty(n_runs)

    for r in range(n_runs):
        # Independent but reproducible seed per (cell, run).
        sim_seed = base_seed + 1_000_003 * idx_a + 1_009 * idx_S + r
        rng = np.random.default_rng(sim_seed)

        # Resample initial opinions per run (offset state_seed by r).
        state, A, W = initialise_fast(G, w0, pD=pD, seed=state_seed + r)

        # Burn-in: advance to t = T - tail_len without recording.
        for _ in range(T - tail_len):
            state = step_opinions_fast(A, W, state, S_val, rng)
            W = step_weights_fast(A, W, state, a_val)

        # Tail: record P at every step.
        P_tail = np.empty(tail_len)
        for t in range(tail_len):
            state = step_opinions_fast(A, W, state, S_val, rng)
            W = step_weights_fast(A, W, state, a_val)
            P_tail[t] = polarization_coefficient_fast(A, W, state)

        P_final[r] = P_tail[-1]
        sigma_time_per_run[r] = float(P_tail.std())

    return (idx_a, idx_S,
            float(P_final.mean()),
            float(P_final.std()),
            float(sigma_time_per_run.mean()),
            P_final.copy())


# ────────────────────────────────────────────────────────────────────────
# Sweep driver
# ────────────────────────────────────────────────────────────────────────

def run_sweep(graph_kind, n, k, pD, w0, T, n_runs, grid, a_max,
              S_min, S_max, tail_frac, graph_seed, state_seed,
              base_seed, n_jobs, serial):
    print(f"[setup] {graph_kind.upper()}  N={n}  <k>={k}  pD={pD}  w0={w0}")
    print(f"[setup] grid {grid}x{grid}  runs/cell={n_runs}  T={T}  tail={tail_frac}")

    a_vals = np.logspace(np.log10(0.005), np.log10(a_max), grid)
    S_vals = np.logspace(np.log10(S_min), np.log10(S_max), grid)
    print(f"[setup] a ∈ [{a_vals[0]:.4f}, {a_vals[-1]:.3f}]  "
          f"S ∈ [{S_vals[0]:.1f}, {S_vals[-1]:.1f}]  (both log-spaced)")

    job_args = [
        (ia, iS, av, Sv,
         graph_kind, n, k, pD, w0, T, n_runs, tail_frac, base_seed,
         graph_seed, state_seed)
        for ia, av in enumerate(a_vals)
        for iS, Sv in enumerate(S_vals)
    ]

    mean_grid    = np.full((grid, grid), np.nan)
    sigma_r_grid = np.full((grid, grid), np.nan)
    sigma_t_grid = np.full((grid, grid), np.nan)
    P_finals_all = np.full((grid, grid, n_runs), np.nan)

    total = len(job_args)
    t0 = time.time()

    def _consume(ia, iS, mP, sR, sT, P_fin):
        mean_grid[ia, iS]    = mP
        sigma_r_grid[ia, iS] = sR
        sigma_t_grid[ia, iS] = sT
        P_finals_all[ia, iS] = P_fin

    if serial:
        for i, args_ in enumerate(job_args, 1):
            _consume(*_cell_worker(args_))
            if i % max(1, total // 20) == 0 or i == total:
                _progress(i, total, t0)
    else:
        if n_jobs is None:
            n_jobs = max(1, min(4, (os.cpu_count() or 2) - 2))
        print(f"[run]   parallel workers={n_jobs}")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_cell_worker, a) for a in job_args]
            for i, fut in enumerate(as_completed(futs), 1):
                _consume(*fut.result())
                if i % max(1, total // 40) == 0 or i == total:
                    _progress(i, total, t0)

    elapsed = time.time() - t0
    print(f"[done]  {total} cells in {elapsed:.1f}s  "
          f"({elapsed / total * 1000:.1f} ms/cell)")
    return a_vals, S_vals, mean_grid, sigma_r_grid, sigma_t_grid, P_finals_all


def _progress(k, total, t0):
    e = time.time() - t0
    eta = e / k * (total - k)
    print(f"  [{k:>5}/{total}]  {e:6.1f}s elapsed  ETA {eta:6.1f}s")


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def plot_three_panel(a_vals, S_vals, mean_grid, sigma_r_grid, sigma_t_grid,
                     graph_kind, n, k, pD, w0, T, n_runs, out_png):
    import matplotlib.pyplot as plt

    extent = [a_vals[0], a_vals[-1],
              float(np.log10(S_vals[0])), float(np.log10(S_vals[-1]))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [
        (r"$\langle P(T) \rangle$ — mean polarization",     mean_grid,    "viridis", 0.0, 1.0),
        (r"$\sigma_{\mathrm{runs}}$ — between-run dispersion", sigma_r_grid, "magma",   None, None),
        (r"$\sigma_{\mathrm{time}}$ — within-run dispersion",  sigma_t_grid, "cividis", None, None),
    ]
    for ax, (title, grid_data, cmap, vmn, vmx) in zip(axes, panels):
        im = ax.imshow(grid_data.T, origin="lower", aspect="auto",
                       extent=extent, cmap=cmap, vmin=vmn, vmax=vmx)
        ax.set_xlabel(r"$a$ — Hebbian step")
        ax.set_ylabel(r"$\log_{10} S$ — stubbornness")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    # Overlay the σ_runs ridge on top of the mean panel.
    # This is the headline visual: the bistable band sits inside the smooth mean.
    if np.nanmax(sigma_r_grid) > 0:
        thresh = 0.7 * float(np.nanmax(sigma_r_grid))
        A, Sgrid = np.meshgrid(a_vals, np.log10(S_vals))
        axes[0].contour(A, Sgrid, sigma_r_grid.T,
                        levels=[thresh], colors="white",
                        linewidths=1.5, linestyles="--")

    fig.suptitle(
        f"Bistability decomposition  |  {graph_kind.upper()}  N={n}  <k>={k}  "
        f"pD={pD}  w0={w0}  T={T}  runs/cell={n_runs}\n"
        f"White dashed contour on left panel = σ_runs ridge (>70% of max) — the bistable band",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save] {out_png}")


def plot_broom(traces, a, S, out_png):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    final_vals = np.array([tr[-1] for tr in traces])
    # Color each trace by its final outcome — strengthens the "split fate" visual.
    for tr in traces:
        color = "#d62728" if tr[-1] > 0.7 else ("#1f77b4" if tr[-1] < 0.4 else "#888888")
        ax.plot(tr, alpha=0.45, lw=0.8, color=color)
    ax.axhline(np.median(final_vals), ls=":", color="black",
               label=f"median P(T) = {np.median(final_vals):.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$P(t)$")
    ax.set_title(
        f"Broom plot at the bistable cell  (a={a:.4f}, S={S:.2f})\n"
        f"{len(traces)} independent runs, identical parameters — "
        f"red = polarised, blue = mixed, grey = intermediate"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save] {out_png}")


def plot_hist(P_finals, a, S, out_png):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(P_finals, bins=20, range=(0, 1),
            edgecolor="black", color="C3", alpha=0.75)
    ax.axvline(0.4, ls="--", color="gray", lw=0.8)
    ax.axvline(0.7, ls="--", color="gray", lw=0.8)
    ax.set_xlabel(r"$P(T)$")
    ax.set_ylabel("count")
    ax.set_title(
        f"Distribution of P(T)  at  (a={a:.4f}, S={S:.2f})\n"
        f"Two peaks  →  narrative confirmed.   One peak  →  narrative fails."
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save] {out_png}")


# ────────────────────────────────────────────────────────────────────────
# Diagnostic
# ────────────────────────────────────────────────────────────────────────

def bimodality_report(P_finals, a, S):
    lo  = float(np.mean(P_finals < 0.4))
    hi  = float(np.mean(P_finals > 0.7))
    mid = float(np.mean((P_finals >= 0.4) & (P_finals <= 0.7)))
    # Operational definition: each mode has > 25% mass, valley has < 30%.
    bimodal = (lo > 0.25) and (hi > 0.25) and (mid < 0.30)
    return {
        "a": a, "S": S, "n": len(P_finals),
        "frac_mixed":        lo,
        "frac_intermediate": mid,
        "frac_polarized":    hi,
        "mean": float(np.mean(P_finals)),
        "std":  float(np.std(P_finals)),
        "bimodal_verdict": bimodal,
    }


def write_summary(report, sigma_r_max, out_txt):
    lines = [
        "BISTABILITY EXPERIMENT — SUMMARY",
        "=" * 50,
        f"Max σ_runs over grid:  {sigma_r_max:.3f}",
        f"Chosen cell:           a={report['a']:.4f},  S={report['S']:.2f},"
        f"  n_runs={report['n']}",
        "",
        f"  mean P(T):                   {report['mean']:.3f}",
        f"  std  P(T):                   {report['std']:.3f}",
        "",
        "  Fraction of runs by bucket:",
        f"    Mixed       (P < 0.4):     {report['frac_mixed']*100:5.1f}%",
        f"    Intermediate (0.4-0.7):    {report['frac_intermediate']*100:5.1f}%",
        f"    Polarized   (P > 0.7):     {report['frac_polarized']*100:5.1f}%",
        "",
    ]
    if report["bimodal_verdict"]:
        lines += [
            "VERDICT: distribution looks BIMODAL.",
            "  → Narrative HOLDS: same parameters, opposite fates.",
            "  → Safe to claim 'critical band where history matters'.",
        ]
    else:
        lines += [
            "VERDICT: distribution does NOT look clearly bimodal.",
            "  → Narrative may not hold at this cell. Possible reasons:",
            "      • n_runs too low to resolve the modes — try --runs 80",
            "      • Picked the wrong cell — inspect the σ_runs heatmap",
            "        and try --cell a,S with values right on the ridge",
            "      • The model is genuinely unimodal here (chaotic, not bistable)",
            "        — check σ_time at the same cell.",
        ]
    txt = "\n".join(lines)
    print()
    print(txt)
    with open(out_txt, "w") as f:
        f.write(txt + "\n")
    print(f"\n[save] {out_txt}")


# ────────────────────────────────────────────────────────────────────────
# Broom plot generator (separate, dedicated runs for full P(t) traces)
# ────────────────────────────────────────────────────────────────────────

def run_broom(graph_kind, n, k, pD, w0, T, a, S, n_demo,
              graph_seed, state_seed, base_seed):
    G = make_graph(graph_kind, n, k, seed=graph_seed)
    traces = []
    for r in range(n_demo):
        # Seeds chosen disjoint from those used in the sweep.
        rng = np.random.default_rng(base_seed + 7919 * (r + 1))
        state, A, W = initialise_fast(G, w0, pD=pD, seed=state_seed + 31 * (r + 1))
        P_trace = np.empty(T)
        for t in range(T):
            state = step_opinions_fast(A, W, state, S, rng)
            W = step_weights_fast(A, W, state, a)
            P_trace[t] = polarization_coefficient_fast(A, W, state)
        traces.append(P_trace)
    return traces


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--topo", choices=["ws", "er", "ba"], default="ws")
    p.add_argument("--n",    type=int,   default=80)
    p.add_argument("--k",    type=int,   default=8)
    p.add_argument("--pD",   type=float, default=0.5)
    p.add_argument("--w0",   type=float, default=0.5)
    p.add_argument("--T",    type=int,   default=200)
    p.add_argument("--runs", type=int,   default=40, dest="n_runs")
    p.add_argument("--grid", type=int,   default=30)
    p.add_argument("--a-max", type=float, default=0.08,
                   help="upper bound for a-axis. Default 0.08 zooms into the interesting band.")
    p.add_argument("--S-min", type=float, default=1.0)
    p.add_argument("--S-max", type=float, default=30.0,
                   help="upper bound for S-axis. Default 30 — above that everything is polarized.")
    p.add_argument("--tail-frac", type=float, default=0.2,
                   help="fraction of T at the end used to measure σ_time")
    p.add_argument("--n-demo", type=int, default=50,
                   help="number of runs in the broom plot")
    p.add_argument("--cell", default=None,
                   help="force broom-plot cell as 'a,S' (overrides auto-selection)")
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--state-seed", type=int, default=123)
    p.add_argument("--base-seed",  type=int, default=2024)
    p.add_argument("--jobs",   type=int,  default=None,
                   help="parallel workers (default: min(4, cpu_count - 2))")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--quick",  action="store_true",
                   help="12x12 grid, 15 runs, T=150 — fast sanity check (~30 s)")
    p.add_argument("--out-prefix", default="bistab",
                   help="prefix for output files")
    args = p.parse_args()

    if args.quick:
        args.grid   = 12
        args.n_runs = 15
        args.T      = 150

    # ── Sweep ────────────────────────────────────────────────────────
    (a_vals, S_vals,
     mean_grid, sigma_r_grid, sigma_t_grid,
     P_finals_all) = run_sweep(
        graph_kind=args.topo, n=args.n, k=args.k, pD=args.pD, w0=args.w0,
        T=args.T, n_runs=args.n_runs, grid=args.grid,
        a_max=args.a_max, S_min=args.S_min, S_max=args.S_max,
        tail_frac=args.tail_frac,
        graph_seed=args.graph_seed, state_seed=args.state_seed,
        base_seed=args.base_seed, n_jobs=args.jobs, serial=args.serial,
    )

    # ── Save raw data ────────────────────────────────────────────────
    npz_path = f"{args.out_prefix}_results.npz"
    np.savez(npz_path,
             a_vals=a_vals, S_vals=S_vals,
             mean_grid=mean_grid,
             sigma_r_grid=sigma_r_grid,
             sigma_t_grid=sigma_t_grid,
             P_finals_all=P_finals_all,
             topo=args.topo, n=args.n, k=args.k, pD=args.pD, w0=args.w0,
             T=args.T, n_runs=args.n_runs)
    print(f"[save] {npz_path}")

    # ── Heatmaps ─────────────────────────────────────────────────────
    plot_three_panel(a_vals, S_vals, mean_grid, sigma_r_grid, sigma_t_grid,
                     args.topo, args.n, args.k, args.pD, args.w0,
                     args.T, args.n_runs,
                     f"{args.out_prefix}_heatmaps.png")

    # ── Pick the bistable cell ───────────────────────────────────────
    if args.cell is not None:
        a_chosen, S_chosen = (float(x) for x in args.cell.split(","))
        print(f"[pick] user-chosen cell:           a={a_chosen}, S={S_chosen}")
    else:
        # Restrict to cells with non-trivial mean P, so a high σ corresponds
        # to actual bistability and not to saturation noise at the corners.
        eligible = (mean_grid > 0.15) & (mean_grid < 0.95)
        masked = np.where(eligible, sigma_r_grid, -np.inf)
        if not np.isfinite(masked).any():
            # Fallback: pick global argmax of σ_runs.
            masked = sigma_r_grid
        ia, iS = np.unravel_index(np.argmax(masked), masked.shape)
        a_chosen, S_chosen = float(a_vals[ia]), float(S_vals[iS])
        print(f"[pick] auto-chosen bistable cell:  "
              f"a={a_chosen:.4f},  S={S_chosen:.2f},  "
              f"σ_runs={sigma_r_grid[ia, iS]:.3f}")

    # ── Broom + histogram + diagnostic ───────────────────────────────
    traces = run_broom(args.topo, args.n, args.k, args.pD, args.w0, args.T,
                       a_chosen, S_chosen, args.n_demo,
                       args.graph_seed, args.state_seed, args.base_seed)
    plot_broom(traces, a_chosen, S_chosen, f"{args.out_prefix}_broom.png")

    P_finals_demo = np.array([tr[-1] for tr in traces])
    plot_hist(P_finals_demo, a_chosen, S_chosen, f"{args.out_prefix}_hist.png")

    report = bimodality_report(P_finals_demo, a_chosen, S_chosen)
    write_summary(report, float(np.nanmax(sigma_r_grid)),
                  f"{args.out_prefix}_summary.txt")


if __name__ == "__main__":
    main()
