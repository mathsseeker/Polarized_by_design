"""
fit_critical_line_sim_linear.py
===============================

Linear-axis version of `fit_critical_line_sim.py`.

Re-plots the polarisation heatmap with the simulation-based boundary points
and the power-law fit S = c * a^(-alpha), but on **linear** x/y axes instead
of log-log.  No re-simulation: the boundary points and fit coefficients are
read straight out of the `_fit_sim.npz` produced by `fit_critical_line_sim.py`,
and the heatmap itself comes from the source npz (e.g.
`heatmap_k_variable_results_er_k008.npz`).

Note: because `a_vals` / `S_vals` are geometrically spaced, the cells will
appear wider toward large a and taller toward large S on linear axes — that
is just the same data displayed on a non-log grid.  The fit curve looks like
a hyperbola in linear axes; it is a straight line only in log-log.

Usage
-----
    conda activate ode
    python fit_critical_line_sim_linear.py
    python fit_critical_line_sim_linear.py \
        --npz heatmap_k_variable_results_er_k008.npz \
        --fit-npz heatmap_k_variable_results_er_k008_fit_sim.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Linear cell edges (midpoints between consecutive values)
# ---------------------------------------------------------------------------

def _linear_edges(vals: np.ndarray) -> np.ndarray:
    step_left = vals[1] - vals[0]
    step_right = vals[-1] - vals[-2]
    left = np.array([vals[0] - step_left / 2.0])
    mids = 0.5 * (vals[:-1] + vals[1:])
    right = np.array([vals[-1] + step_right / 2.0])
    return np.concatenate([left, mids, right])


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def _title(data, observable_tex: str) -> str:
    kind = str(data["graph_kind"])
    n = int(data["n"])
    k_avg = int(data["k_avg"])
    pD = float(data["pD"])
    w0 = float(data["w0"])
    T = int(data["T"])
    return (
        f"{observable_tex}     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{kind.upper()},  $N$={n},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T}    (linear axes)"
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_main_linear(
    out_png: Path,
    P_grid: np.ndarray, a_vals: np.ndarray, S_vals: np.ndarray,
    a_pts: np.ndarray, S_pts: np.ndarray,
    alpha: float, sigma_alpha: float, c: float, sigma_c: float, r2: float,
    data, T_long: int, n_runs_ext: int, P_target: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pcm = ax.pcolormesh(
        a_vals, S_vals, P_grid.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="gouraud",
    )
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(_title(data, r"$P(T) = 1 - \bar{w}_\times(T)$"))
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    a_med = float(np.median(a_pts))
    S_med = float(np.median(S_pts))
    c_ref = a_med * S_med
    a_samples = np.geomspace(a_pts.min(), a_pts.max(), 20)
    S_samples = c_ref / a_samples
    pchip = PchipInterpolator(a_samples, S_samples, extrapolate=False)
    a_line = np.linspace(a_samples[0], a_samples[-1], 400)
    S_interp = pchip(a_line)

    ax.plot(
        a_samples, S_samples, "o",
        mfc="white", mec="black", ms=5, lw=0,
        label=r"interpolation nodes (sampled from median reference)",
    )
    ax.plot(
        a_line, S_interp, "r-", lw=2,
        label=r"PCHIP interpolation through the nodes",
    )

    ax.set_xlim(a_vals[0], a_vals[-1])
    ax.set_ylim(S_vals[0], S_vals[-1])

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    npz_path = Path(args.npz)
    fit_path = Path(args.fit_npz)

    data = np.load(npz_path, allow_pickle=False)
    fit = np.load(fit_path, allow_pickle=False)

    P_grid = data["grid"]
    a_vals = data["a_vals"]
    S_vals = data["S_vals"]

    a_pts = fit["a_pts"]
    S_pts = fit["S_pts"]
    alpha = float(fit["alpha"])
    sigma_alpha = float(fit["sigma_alpha"])
    c = float(fit["c"])
    sigma_c = float(fit["sigma_c"])
    r2 = float(fit["r2"])
    T_long = int(fit["T_long"])
    n_runs_ext = int(fit["n_runs_ext"])
    P_target = float(fit["P_target"])

    print("=" * 72)
    print(f"Loaded heatmap : {npz_path.name}")
    print(f"Loaded fit     : {fit_path.name}")
    print(
        f"  alpha = {alpha:.4f} +/- {sigma_alpha:.4f}    "
        f"c = {c:.4g} +/- {sigma_c:.2g}    R^2 = {r2:.4f}"
    )
    print(f"  boundary points: {len(a_pts)}")
    print("=" * 72)

    out_png = fit_path.with_name(fit_path.stem + "_linear.png")
    plot_main_linear(
        out_png,
        P_grid, a_vals, S_vals, a_pts, S_pts,
        alpha, sigma_alpha, c, sigma_c, r2, data,
        T_long, n_runs_ext, P_target,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path(__file__).with_name("heatmap_k_variable_results_er_k008.npz"),
        help="Source heatmap npz (provides P_grid, a_vals, S_vals, metadata).",
    )
    p.add_argument(
        "--fit-npz",
        type=Path,
        default=Path(__file__).with_name("heatmap_k_variable_results_er_k008_fit_sim.npz"),
        help="Fit-results npz produced by fit_critical_line_sim.py.",
    )
    args = p.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    if not args.fit_npz.exists():
        raise FileNotFoundError(args.fit_npz)
    run(args)


if __name__ == "__main__":
    main()
