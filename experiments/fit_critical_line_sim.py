"""
fit_critical_line_sim.py
========================

Simulation-based boundary fit for the polarisation phase line.

Motivation
----------
`fit_critical_line_B.py` extracts the boundary as the P = 0.5 contour of the
original T = 1000 sweep.  That definition is somewhat arbitrary: cells in the
green/teal band (P in roughly [0.5, 0.95]) may simply be slow to polarise and
would reach P ~= 1 if simulated long enough.

This script defines the boundary *operationally*: a cell belongs to the
polarised side if extended simulations (T_long >> T_grid) drive its
polarisation up to the fully-polarised plateau (median P >= P_target across
n_runs realisations).  The boundary  S*(a)  is then the lowest S, per column
a, at which this criterion is satisfied.

For each (a, S) cell flagged as "transition" — 0.3 <= P(T_grid) < 1.0 — we
re-simulate with T_long steps and n_runs independent (graph + state) draws,
using `polarized_fast.step_*` (mathematically identical to `polarized.py`,
verified in `polarized_fast.test_equivalence`).  Cells already fully polarised
at T_grid (P >= 1.0 within float tolerance) are taken as confirmed polarised
without re-simulation.

After the boundary points are collected we fit a power law
        log S* = log c - alpha * log a
and plot the heatmap with:
    - open circles at the simulation-based S*(a),
    - a solid red fit line extrapolated across the full a range,
    - a dashed alpha = 1 reference line through the median point.

Usage
-----
    conda activate ode
    python fit_critical_line_sim.py
    python fit_critical_line_sim.py --T-long 20000 --n-runs 10 --P-target 0.98
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph
from polarized_fast import (
    initialise_fast,
    step_opinions_fast,
    step_weights_fast,
    polarization_coefficient_fast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_edges(vals: np.ndarray) -> np.ndarray:
    ratio = vals[1] / vals[0]
    left = np.array([vals[0] / ratio])
    mids = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


# ---------------------------------------------------------------------------
# Per-cell extended simulation using polarized_fast
# ---------------------------------------------------------------------------

def simulate_cell_extended(
    graph_kind: str, n: int, k_avg: int,
    a: float, S: float, w0: float, pD: float,
    T_long: int, n_runs: int,
    graph_seed: int, state_seed: int, base_seed: int,
) -> np.ndarray:
    """Run n_runs independent extended simulations of T_long steps.

    The same fixed graph is used for all runs (matches the original sweep
    convention: one graph per cell, varied initial states / dynamical noise).
    Returns the per-run final polarisation P(T_long), shape (n_runs,).
    """
    G = make_graph(graph_kind, n, k_avg, seed=graph_seed)
    P_runs = np.empty(n_runs, dtype=np.float64)

    for r in range(n_runs):
        state, A, W = initialise_fast(G, w0=w0, pD=pD, seed=state_seed + r)
        rng = np.random.default_rng(base_seed + r)
        for _ in range(T_long):
            state = step_opinions_fast(A, W, state, S, rng)
            W = step_weights_fast(A, W, state, a)
        P_runs[r] = polarization_coefficient_fast(A, W, state)

    return P_runs


# ---------------------------------------------------------------------------
# Boundary extraction from extended simulations
# ---------------------------------------------------------------------------

def extract_boundary_sim(
    P_grid: np.ndarray,
    a_vals: np.ndarray,
    S_vals: np.ndarray,
    *,
    graph_kind: str, n: int, k_avg: int,
    w0: float, pD: float,
    T_long: int, n_runs: int, P_target: float,
    transition_lo: float, transition_hi: float, polarised_thr: float,
    graph_seed: int, state_seed: int, base_seed: int,
    verbose: bool = True,
):
    """For each column a, run extended sims on cells in the transition band
    [transition_lo, transition_hi) and on confirmed-polarised cells
    (P_grid >= polarised_thr).  S*(a) = lowest S whose median extended P
    reaches P_target.

    Returns (a_pts, S_pts, ext_results) where ext_results is a dict mapping
    (ia, iS) -> per-run final P (length n_runs), useful for diagnostics.
    """
    Na, NS = P_grid.shape
    assert NS == len(S_vals) and Na == len(a_vals)

    a_pts: list[float] = []
    S_pts: list[float] = []
    ext_results: dict[tuple[int, int], np.ndarray] = {}

    t0 = time.time()
    n_sims = 0

    for ia in range(Na):
        col = P_grid[ia]

        # Identify the cells we need to (re-)evaluate
        polarised_mask = col >= polarised_thr
        transition_mask = (col >= transition_lo) & (col < transition_hi)

        if not polarised_mask.any() and not transition_mask.any():
            continue  # entire column below the transition zone — no boundary

        # Walk S from low to high.  As soon as we hit a polarised cell
        # (either already-polarised at T_grid, or one whose extended
        # simulation reaches P_target), that S is S*(a) by the
        # monotonicity assumption near the boundary — stop the column.
        S_star = None

        for iS in range(NS):
            if polarised_mask[iS]:
                S_star = float(S_vals[iS])
                break
            if not transition_mask[iS]:
                # cell is below the transition band (P < transition_lo):
                # do not re-simulate, skip up the column
                continue

            P_runs = simulate_cell_extended(
                graph_kind=graph_kind, n=n, k_avg=k_avg,
                a=a_vals[ia], S=S_vals[iS], w0=w0, pD=pD,
                T_long=T_long, n_runs=n_runs,
                graph_seed=graph_seed,
                state_seed=state_seed + 1000 * iS + 7 * ia,
                base_seed=base_seed + 1000 * iS + 13 * ia,
            )
            ext_results[(ia, iS)] = P_runs
            n_sims += 1
            is_pol = float(np.median(P_runs)) >= P_target
            if verbose:
                elapsed = time.time() - t0
                print(
                    f"  [sim {n_sims:>3}]  ia={ia:>2}/{Na} iS={iS:>2}/{NS}  "
                    f"a={a_vals[ia]:.4f}  S={S_vals[iS]:.2f}  "
                    f"P(T_grid)={col[iS]:.3f}  "
                    f"median P(T_long)={np.median(P_runs):.3f}  "
                    f"{'POL' if is_pol else 'NOT'}  ({elapsed:.0f}s)"
                )
            if is_pol:
                S_star = float(S_vals[iS])
                break  # early stop — found the lowest polarising S in this column

        if S_star is None:
            continue  # no polarised cell found anywhere in this column
        a_pts.append(float(a_vals[ia]))
        S_pts.append(S_star)

    return np.asarray(a_pts), np.asarray(S_pts), ext_results


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_log_log(a_pts: np.ndarray, S_pts: np.ndarray):
    x = np.log(a_pts)
    y = np.log(S_pts)
    res = linregress(x, y)
    alpha = -res.slope
    sigma_alpha = float(res.stderr)
    log_c = float(res.intercept)
    sigma_log_c = float(res.intercept_stderr)
    c = float(np.exp(log_c))
    sigma_c = c * sigma_log_c
    r2 = float(res.rvalue ** 2)
    return alpha, sigma_alpha, c, sigma_c, r2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _title(data, observable_tex: str) -> str:
    kind = str(data["graph_kind"])
    n = int(data["n"])
    k_avg = int(data["k_avg"])
    pD = float(data["pD"])
    w0 = float(data["w0"])
    T = int(data["T"])
    n_runs = int(data["n_runs"])
    return (
        f"{observable_tex}     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{kind.upper()},  $N$={n},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T},  runs/cell={n_runs}"
    )


def plot_main(
    out_png: Path,
    P_grid: np.ndarray, a_vals: np.ndarray, S_vals: np.ndarray,
    a_pts: np.ndarray, S_pts: np.ndarray,
    alpha: float, sigma_alpha: float, c: float, sigma_c: float, r2: float,
    data, T_long: int, n_runs_ext: int, P_target: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), P_grid.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(_title(data, r"$P(T) = 1 - \bar{w}_\times(T)$"))
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    # Simulation-based boundary points
    ax.plot(
        a_pts, S_pts, "o",
        mfc="white", mec="black", ms=5, lw=0,
        label=(
            rf"$S^*(a)$  (sim, $T_{{long}}={T_long}$, "
            rf"$n_{{runs}}={n_runs_ext}$, $\mathrm{{median}}\,P\geq{P_target}$)"
        ),
    )

    # Fit line — extrapolated across the FULL a range (and even slightly past)
    a_line = np.geomspace(a_vals[0] * 0.7, a_vals[-1] * 1.4, 400)
    S_fit = c * a_line ** (-alpha)
    ax.plot(
        a_line, S_fit, "r-", lw=2,
        label=(
            rf"fit:  $\alpha={alpha:.2f}\pm{sigma_alpha:.2f}$,  "
            rf"$c={c:.2g}\pm{sigma_c:.1g}$  ($R^2={r2:.2f}$)"
        ),
    )

    # alpha = 1 reference through the median point of the boundary cloud
    a_med = float(np.exp(np.median(np.log(a_pts))))
    S_med = float(np.exp(np.median(np.log(S_pts))))
    c_ref = a_med * S_med
    S_ref = c_ref / a_line
    ax.plot(
        a_line, S_ref, "r--", lw=1.2, alpha=0.8,
        label=rf"reference $aS={c_ref:.2g}$  ($\alpha=1$, through median)",
    )

    # Pin axes to the heatmap range (the fit line will visibly extend to
    # the corners, demonstrating the extrapolation).
    ax.set_xlim(_log_edges(a_vals)[[0, -1]])
    ax.set_ylim(_log_edges(S_vals)[[0, -1]])

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=False)
    P_grid = data["grid"]
    a_vals = data["a_vals"]
    S_vals = data["S_vals"]

    graph_kind = str(data["graph_kind"])
    n = int(data["n"])
    k_avg = int(data["k_avg"])
    w0 = float(data["w0"])
    pD = float(data["pD"])
    graph_seed = int(data["graph_seed"])
    state_seed = int(data["state_seed"])
    base_seed = int(data["base_seed"])

    print("=" * 72)
    print(f"Loaded: {npz_path.name}")
    print(f"  graph={graph_kind}  N={n}  <k>={k_avg}  pD={pD:.2f}  w0={w0:.2f}")
    print(f"  grid: {len(a_vals)} x {len(S_vals)}  "
          f"a in [{a_vals[0]:.4f}, {a_vals[-1]:.4f}], "
          f"S in [{S_vals[0]:.2f}, {S_vals[-1]:.2f}]")
    print("=" * 72)

    # Count transition cells so user knows the workload
    transition = ((P_grid >= args.transition_lo) & (P_grid < args.transition_hi))
    n_transition = int(transition.sum())
    print(
        f"Transition cells "
        f"[{args.transition_lo}, {args.transition_hi}): {n_transition} "
        f"-> {n_transition} extended sims of "
        f"T_long={args.T_long}, n_runs={args.n_runs}\n"
    )

    a_pts, S_pts, ext_results = extract_boundary_sim(
        P_grid, a_vals, S_vals,
        graph_kind=graph_kind, n=n, k_avg=k_avg,
        w0=w0, pD=pD,
        T_long=args.T_long, n_runs=args.n_runs, P_target=args.P_target,
        transition_lo=args.transition_lo, transition_hi=args.transition_hi,
        polarised_thr=args.polarised_thr,
        graph_seed=graph_seed, state_seed=state_seed, base_seed=base_seed,
        verbose=True,
    )

    if len(a_pts) < 3:
        raise RuntimeError(
            f"Too few boundary points ({len(a_pts)}) - cannot fit a line."
        )

    # Optionally restrict the fit to S > S_fit_min (the deep-low-S region
    # has a separate physics regime; default 3.0 matches fit_critical_line_B).
    fit_mask = S_pts >= args.S_fit_min
    a_fit = a_pts[fit_mask]
    S_fit_pts = S_pts[fit_mask]
    print(f"\n[fit] using {fit_mask.sum()}/{len(a_pts)} boundary points (S > {args.S_fit_min})")

    alpha, sa, c, sc, r2 = fit_log_log(a_fit, S_fit_pts)
    z = (alpha - 1.0) / sa if sa > 0 else float("nan")

    print()
    print(f"[fit-sim]  median-P>={args.P_target} after T_long={args.T_long}")
    print(f"           columns used     : {fit_mask.sum()} / {len(a_vals)}")
    print(f"           alpha            : {alpha:.4f}  +/-  {sa:.4f}")
    print(f"           c                : {c:.4g}  +/-  {sc:.2g}")
    print(f"           R^2              : {r2:.4f}")
    print(f"           z(alpha=1)       : {z:+.2f} sigma "
          f"({'compatible' if abs(z) < 2 else 'rejected'} with aS=c)")

    out_stem = npz_path.stem + "_fit_sim"
    out_png = npz_path.with_name(out_stem + ".png")
    out_npz = npz_path.with_name(out_stem + ".npz")

    plot_main(
        out_png,
        P_grid, a_vals, S_vals, a_pts, S_pts,
        alpha, sa, c, sc, r2, data,
        args.T_long, args.n_runs, args.P_target,
    )

    # Save the per-cell extended-sim results so we can audit later
    np.savez(
        out_npz,
        a_pts=a_pts, S_pts=S_pts,
        alpha=alpha, sigma_alpha=sa, c=c, sigma_c=sc, r2=r2,
        T_long=args.T_long, n_runs_ext=args.n_runs, P_target=args.P_target,
        transition_lo=args.transition_lo, transition_hi=args.transition_hi,
        polarised_thr=args.polarised_thr, S_fit_min=args.S_fit_min,
        ext_cells_ia=np.array([k[0] for k in ext_results.keys()], dtype=np.int64),
        ext_cells_iS=np.array([k[1] for k in ext_results.keys()], dtype=np.int64),
        ext_P_runs=np.array(list(ext_results.values()), dtype=np.float64),
    )
    print(f"[save]  {out_npz}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path(__file__).with_name("heatmap_k_variable_results_er_k004.npz"),
    )
    p.add_argument("--T-long", type=int, default=10_000, dest="T_long",
                   help="Time horizon for extended simulations")
    p.add_argument("--n-runs", type=int, default=10, dest="n_runs",
                   help="Independent realisations per cell")
    p.add_argument("--P-target", type=float, default=0.98, dest="P_target",
                   help="Median final P to classify a cell as 'eventually polarises'")
    p.add_argument("--transition-lo", type=float, default=0.30, dest="transition_lo",
                   help="Lower P(T_grid) bound of cells to re-evaluate")
    p.add_argument("--transition-hi", type=float, default=1.00, dest="transition_hi",
                   help="Upper P(T_grid) bound of cells to re-evaluate")
    p.add_argument("--polarised-thr", type=float, default=1.00, dest="polarised_thr",
                   help="P(T_grid) threshold above which a cell is taken as polarised "
                        "without re-simulating")
    p.add_argument("--S-fit-min", type=float, default=3.0, dest="S_fit_min",
                   help="Restrict the power-law fit to boundary points with S above "
                        "this floor (matches fit_critical_line_B's S>3 convention).")
    args = p.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    run(args)


if __name__ == "__main__":
    main()
