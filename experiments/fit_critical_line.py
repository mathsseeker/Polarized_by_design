"""
fit_critical_line.py
====================

Extracts the P(T)=0.5 boundary from a heatmap of (a, S) vs polarisation,
and fits the power law

        log S = log c - alpha * log a       <=>      a * S^(1/alpha) = c

If alpha ~ 1, the boundary collapses to the hyperbola  a * S = c  (the
hypothesis suggested in the chat).  Otherwise the fit gives the true
exponent with 1-sigma error bars and a z-test against alpha=1.

Usage
-----
    conda activate ode
    python fit_critical_line.py                       # defaults to er_k004
    python fit_critical_line.py --npz heatmap_k_variable_results_ba_k008.npz
    python fit_critical_line.py --npz ... --level 0.5 --trim 1

The script is read-only with respect to the input data; it only writes
a PNG next to the input file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Helpers (mirror heatmap_k_variable.py for visual consistency)
# ---------------------------------------------------------------------------

def _log_edges(vals: np.ndarray) -> np.ndarray:
    """Bin edges for pcolormesh given log-spaced centres."""
    ratio = vals[1] / vals[0]
    left  = np.array([vals[0] / ratio])
    mids  = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


# ---------------------------------------------------------------------------
# Core: extract the iso-P contour and fit a power law in log-log
# ---------------------------------------------------------------------------

def extract_contour(
    a_vals: np.ndarray,
    S_vals: np.ndarray,
    P: np.ndarray,
    level: float = 0.5,
) -> np.ndarray:
    """
    Return points (a_k, S_k) on the iso-P=level contour, using marching
    squares (the algorithm matplotlib's contour() uses internally).

    P is indexed as P[ia, iS] -> we pass P.T so that contour gets
    Z[iy=iS, ix=ia], matching the (X=a_vals, Y=S_vals) axes.
    """
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(a_vals, S_vals, P.T, levels=[level])
    # matplotlib >= 3.8 deprecates .allsegs; .allsegs still works, but be safe
    try:
        segs = cs.allsegs[0]
    except AttributeError:
        segs = [p.vertices for p in cs.collections[0].get_paths()]
    plt.close(fig_tmp)

    if not segs:
        return np.empty((0, 2))

    # Keep the longest segment (robust to small islands at the borders)
    main = max(segs, key=len)
    return main  # shape (M, 2): columns are (a, S)


def fit_power_law(
    pts: np.ndarray,
    a_vals: np.ndarray,
    S_vals: np.ndarray,
    trim_cells: int = 1,
):
    """
    Fit  log S = log c - alpha * log a  on the contour points,
    after trimming points that sit on the saturated borders of the grid.

    Returns (alpha, sigma_alpha, c, sigma_c, r2, n_used, mask).
    """
    a, S = pts[:, 0], pts[:, 1]

    # Trim cells touching the borders -- those are unreliable (saturation)
    a_lo = a_vals[trim_cells]
    a_hi = a_vals[-1 - trim_cells]
    S_lo = S_vals[trim_cells]
    S_hi = S_vals[-1 - trim_cells]
    mask = (a >= a_lo) & (a <= a_hi) & (S >= S_lo) & (S <= S_hi)

    if mask.sum() < 5:
        # fall back to using everything if the trim was too aggressive
        mask = np.ones_like(a, dtype=bool)

    x = np.log(a[mask])
    y = np.log(S[mask])

    res = linregress(x, y)
    alpha       = -res.slope
    sigma_alpha = res.stderr            # same magnitude, sign irrelevant
    log_c       = res.intercept
    sigma_log_c = res.intercept_stderr
    c           = float(np.exp(log_c))
    sigma_c     = c * sigma_log_c       # 1st-order error propagation
    r2          = res.rvalue ** 2

    return alpha, sigma_alpha, c, sigma_c, r2, int(mask.sum()), mask


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_with_fit(npz_path: Path, level: float, trim_cells: int) -> Path:
    data = np.load(npz_path, allow_pickle=False)

    P       = data["grid"]              # shape (Na, NS), indexed [ia, iS]
    a_vals  = data["a_vals"]
    S_vals  = data["S_vals"]
    kind    = str(data["graph_kind"])
    n       = int(data["n"])
    k_avg   = int(data["k_avg"])
    pD      = float(data["pD"])
    w0      = float(data["w0"])
    T       = int(data["T"])
    n_runs  = int(data["n_runs"])

    # ----- contour + fit -----
    pts = extract_contour(a_vals, S_vals, P, level=level)
    if len(pts) < 5:
        raise RuntimeError(f"contour P={level} too short ({len(pts)} pts)")

    alpha, sa, c, sc, r2, n_used, mask = fit_power_law(
        pts, a_vals, S_vals, trim_cells=trim_cells
    )

    # z-score against the hypothesis  alpha = 1  (i.e. aS = c)
    z_alpha1 = (alpha - 1.0) / sa if sa > 0 else float("nan")

    print(f"\n[fit]  P={level} contour on  {npz_path.name}")
    print(f"        points on contour : {len(pts)}  (used {n_used} after trim)")
    print(f"        alpha             : {alpha:.4f}  +/-  {sa:.4f}")
    print(f"        c                 : {c:.4g}  +/-  {sc:.2g}")
    print(f"        R^2               : {r2:.4f}")
    print(f"        z(alpha=1)        : {z_alpha1:+.2f} sigma "
          f"({'compatible' if abs(z_alpha1) < 2 else 'rejected'} with aS=c)")

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), P.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(
        f"$P(T) = 1 - \\bar{{w}}_\\times(T)$     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{kind.upper()},  $N$={n},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T},  runs/cell={n_runs}"
    )
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    # contour points: used (white) vs trimmed (grey)
    if (~mask).any():
        ax.plot(pts[~mask, 0], pts[~mask, 1], "o",
                mfc="none", mec="lightgrey", ms=4, lw=0,
                label=f"P={level} (trimmed)")
    ax.plot(pts[mask, 0], pts[mask, 1], "o",
            mfc="white", mec="black", ms=4, lw=0,
            label=f"P={level} contour")

    # fitted line   S = c * a^(-alpha)
    a_line = np.geomspace(a_vals[0], a_vals[-1], 200)
    S_line = c * a_line ** (-alpha)
    in_box = (S_line >= S_vals[0]) & (S_line <= S_vals[-1])
    ax.plot(a_line[in_box], S_line[in_box], "r-", lw=2,
            label=rf"fit:  $S = c\,a^{{-\alpha}}$,  "
                  rf"$\alpha={alpha:.2f}\pm{sa:.2f}$,  "
                  rf"$c={c:.2g}$")

    # reference line for the pure hypothesis  aS = c  (alpha = 1)
    S_ref = c * a_line ** (-1.0)
    in_box_ref = (S_ref >= S_vals[0]) & (S_ref <= S_vals[-1])
    ax.plot(a_line[in_box_ref], S_ref[in_box_ref], "r--", lw=1, alpha=0.7,
            label=rf"reference:  $aS=c$  ($\alpha=1$)")

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_png = npz_path.with_name(npz_path.stem + f"_fit_P{level:.2f}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path(__file__).with_name(
            "heatmap_k_variable_results_er_k004.npz"
        ),
        help="Heatmap results file (.npz produced by heatmap_k_variable.py)",
    )
    p.add_argument("--level", type=float, default=0.5,
                   help="iso-P level to extract (default: 0.5)")
    p.add_argument("--trim",  type=int,   default=1,
                   help="number of border cells to drop before fitting")
    args = p.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)

    plot_with_fit(args.npz, level=args.level, trim_cells=args.trim)


if __name__ == "__main__":
    main()
