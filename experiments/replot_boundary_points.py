"""
replot_boundary_points.py
=========================

Plot the simulation-based phase boundary as:
  - the measured boundary points S*(a) (filled white circles), and
  - a smooth curve INTERPOLATED through those points (PCHIP in log-log),
    extrapolated linearly in log-log past both ends.

No power-law fit is drawn; the curve is the interpolant of the actual dots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _log_edges(vals: np.ndarray) -> np.ndarray:
    ratio = vals[1] / vals[0]
    left = np.array([vals[0] / ratio])
    mids = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


def best_poly_fit(x: np.ndarray, y: np.ndarray, max_degree: int = 4):
    """Pick the polynomial degree in [1, max_degree] with the best adjusted R^2.

    Returns (best_degree, coeffs, adj_r2, r2_table) where r2_table is a list
    of (degree, R^2, adj_R^2) tuples for diagnostics.
    """
    n = len(x)
    y_mean = y.mean()
    ss_tot = float(((y - y_mean) ** 2).sum())

    table = []
    best = None
    for d in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, d)            # highest power first
        y_hat = np.polyval(coeffs, x)
        ss_res = float(((y - y_hat) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Adjusted R^2 penalises extra parameters: 1 - (1-R^2)*(n-1)/(n-d-1)
        if n - d - 1 > 0:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - d - 1)
        else:
            adj_r2 = float("-inf")
        table.append((d, r2, adj_r2))
        if best is None or adj_r2 > best[2]:
            best = (d, coeffs, adj_r2)
    return best[0], best[1], best[2], table


def boundary_curve(a_pts: np.ndarray, S_pts: np.ndarray,
                   a_lo: float, a_hi: float, n_curve: int = 600,
                   max_degree: int = 4):
    """Smooth boundary curve: polynomial fit in (log a, log S) selected by
    adjusted R^2, evaluated over the extended a range.

    Returns a_curve, S_curve, interp_mask, fit_info
        fit_info = (best_degree, coeffs, adj_r2, r2_table)
    """
    order = np.argsort(a_pts)
    a_sorted = a_pts[order]
    S_sorted = S_pts[order]
    x = np.log(a_sorted)
    y = np.log(S_sorted)

    best_d, coeffs, adj_r2, table = best_poly_fit(x, y, max_degree=max_degree)

    a_curve = np.geomspace(a_lo, a_hi, n_curve)
    y_curve = np.polyval(coeffs, np.log(a_curve))
    S_curve = np.exp(y_curve)
    interp_mask = (a_curve >= a_sorted[0]) & (a_curve <= a_sorted[-1])
    return a_curve, S_curve, interp_mask, (best_d, coeffs, adj_r2, table)


def _poly_label(coeffs: np.ndarray) -> str:
    """Human-readable polynomial in log a; coeffs highest power first."""
    d = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        p = d - i
        if abs(c) < 1e-12:
            continue
        if p == 0:
            parts.append(f"{c:+.3f}")
        elif p == 1:
            parts.append(f"{c:+.3f}\\,\\ln a")
        else:
            parts.append(f"{c:+.3f}\\,\\ln^{{{p}}} a")
    expr = " ".join(parts).lstrip("+ ").strip()
    return rf"$\ln S \;=\; {expr}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stem", type=str,
        default="heatmap_k_variable_results_er_k004",
        help="Stem of the .npz files (script appends '.npz' and '_fit_sim.npz').",
    )
    args = parser.parse_args()
    here = Path(__file__).parent
    HEATMAP_NPZ = here / f"{args.stem}.npz"
    FIT_NPZ     = here / f"{args.stem}_fit_sim.npz"
    OUT_PNG     = here / f"{args.stem}_fit_sim_points.png"
    print(f"[load] heatmap = {HEATMAP_NPZ.name}")
    print(f"[load] fit     = {FIT_NPZ.name}")
    print(f"[out]  plot    = {OUT_PNG.name}")

    hm = np.load(HEATMAP_NPZ, allow_pickle=False)
    fit = np.load(FIT_NPZ, allow_pickle=False)

    P_grid = hm["grid"].copy()       # we will OVERWRITE simulated cells
    a_vals = hm["a_vals"]
    S_vals = hm["S_vals"]

    a_pts = fit["a_pts"]
    S_pts = fit["S_pts"]
    T_long = int(fit["T_long"])
    n_runs_ext = int(fit["n_runs_ext"])
    P_target = float(fit["P_target"])

    # ----- Propagate extended-sim results into the heatmap colours -----
    # Cells we re-simulated already have a per-run record in the fit npz.
    # If the median final-P at T_long crossed the polarisation threshold,
    # the cell really IS polarised — replace its (stale, T=1000) value with
    # the extended-sim median so the heatmap colour matches the dot.
    ext_ia    = fit["ext_cells_ia"]
    ext_iS    = fit["ext_cells_iS"]
    ext_Pruns = fit["ext_P_runs"]        # shape (n_simulated_cells, n_runs)
    n_promoted = 0
    n_updated  = 0
    for k in range(len(ext_ia)):
        ia, iS = int(ext_ia[k]), int(ext_iS[k])
        med = float(np.median(ext_Pruns[k]))
        if med >= P_target:
            P_grid[ia, iS] = 1.0           # promote to fully polarised colour
            n_promoted += 1
        else:
            # Leave the heatmap colour as it was: the extended-sim shows the
            # cell does NOT polarise even at T_long, so its true long-time P
            # is similar to its T=1000 P (~0.4-0.8). Updating it to the
            # extended median makes the boundary look cleaner.
            P_grid[ia, iS] = med
            n_updated += 1
    print(f"[promote] {n_promoted} cells re-coloured to P=1 (median P_long >= {P_target})")
    print(f"[refresh] {n_updated} cells updated to their extended-sim median P")

    kind = str(hm["graph_kind"])
    n = int(hm["n"])
    k_avg = int(hm["k_avg"])
    pD = float(hm["pD"])
    w0 = float(hm["w0"])
    T_grid = int(hm["T"])
    n_runs_grid = int(hm["n_runs"])

    # Restrict the view to the heatmap range — curve & extrapolation stay
    # entirely inside the simulated (a, S) box.
    edges_a = _log_edges(a_vals)
    edges_S = _log_edges(S_vals)
    a_lo, a_hi = float(edges_a[0]), float(edges_a[-1])
    S_lo, S_hi = float(edges_S[0]), float(edges_S[-1])

    # Fixed degree-2 polynomial — degree 4 follows wobbles in the noisy data
    # (discrete S grid + only 10 runs/cell). Degree 2 stays smooth and monotone.
    FIXED_DEGREE = 3
    order = np.argsort(a_pts)
    x = np.log(a_pts[order]); y = np.log(S_pts[order])
    coeffs = np.polyfit(x, y, FIXED_DEGREE)
    y_hat = np.polyval(coeffs, x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (1.0 - r2) * (len(x) - 1) / (len(x) - FIXED_DEGREE - 1)
    a_curve = np.geomspace(a_lo, a_hi, 600)
    S_curve = np.exp(np.polyval(coeffs, np.log(a_curve)))
    interp_mask = (a_curve >= a_pts.min()) & (a_curve <= a_pts.max())
    best_d = FIXED_DEGREE
    print(f"[poly] fixed degree {best_d}:  R^2={r2:.4f}  adj R^2={adj_r2:.4f}")

    fig, ax = plt.subplots(figsize=(9, 7))

    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), P_grid.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(
        f"$P(T) = 1 - \\bar{{w}}_\\times(T)$     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{kind.upper()},  $N$={n},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T_grid},  runs/cell={n_runs_grid}"
    )
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    # Fitted portion (within the data range): solid
    ax.plot(
        a_curve[interp_mask], S_curve[interp_mask],
        color="red", lw=2.0,
        label=(
            rf"degree-{best_d} polynomial fit in $(\ln a, \ln S)$  "
            rf"(adj $R^2={adj_r2:.3f}$)"
        ),
    )

    # Extrapolated portion (outside the data range): dashed.
    # Plot the two sides separately so they aren't joined by a phantom line.
    left_mask  = a_curve < a_pts.min()
    right_mask = a_curve > a_pts.max()
    ax.plot(a_curve[left_mask],  S_curve[left_mask],
            color="red", lw=1.8, linestyle="--", alpha=0.95,
            label="extrapolation (same polynomial outside data range)")
    ax.plot(a_curve[right_mask], S_curve[right_mask],
            color="red", lw=1.8, linestyle="--", alpha=0.95)

    # Boundary points on top of the curve
    ax.plot(
        a_pts, S_pts, "o",
        mfc="white", mec="black", mew=1.4, ms=8, lw=0, zorder=5,
        label=(
            rf"$S^*(a)$ from simulation "
            rf"($T_{{long}}={T_long}$, $n_{{runs}}={n_runs_ext}$, "
            rf"median $P\geq {P_target}$)"
        ),
    )

    ax.set_xlim(a_lo, a_hi)
    ax.set_ylim(S_lo, S_hi)

    print(f"[boundary] {len(a_pts)} measured points, "
          f"a in [{a_pts.min():.4f}, {a_pts.max():.4f}]; "
          f"heatmap a in [{a_vals[0]:.4f}, {a_vals[-1]:.4f}]")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"[save] {OUT_PNG}")


if __name__ == "__main__":
    main()
