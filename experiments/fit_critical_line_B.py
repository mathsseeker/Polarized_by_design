"""
fit_critical_line_B.py
======================

Method B for the polarisation-boundary fit.

For each column a_k (each value of the Hebbian step), find the value
S*(a_k) at which the simulated polarisation P(a_k, S) crosses 0.5,
by *linear interpolation in log-S* between the two adjacent grid cells.

Restrictions imposed by the user:
    -  Search only above S > 3   (below that there is a "churn"
       sub-regime that also produces P ~ 0.5 but is not the
       phase boundary we care about).
    -  Columns without a crossing in S > 3 are skipped (no point
       is invented).
    -  Multiple crossings in a column (rare after the S > 3 cut):
       keep the largest-S crossing — the *upper* boundary.

After extracting the points  {(a_k, S*(a_k))}  the script fits

        log S = log c - alpha * log a

via  scipy.stats.linregress  and reports
        alpha +/- sigma_alpha,    c +/- sigma_c,    R^2,    z(alpha=1)

Outputs (next to the input .npz):
    *_fit_B.png        main heatmap  + boundary points + fit line +
                       alpha=1 reference  through the median point
    *_fit_B_var.png    variance (susceptibility) heatmap  +  S*(a)
                       overlay, to confirm the boundary location

Usage:
    conda activate ode
    python fit_critical_line_B.py
    python fit_critical_line_B.py --npz <file.npz> --S-min 3 --level 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_edges(vals: np.ndarray) -> np.ndarray:
    ratio = vals[1] / vals[0]
    left  = np.array([vals[0] / ratio])
    mids  = np.sqrt(vals[:-1] * vals[1:])
    right = np.array([vals[-1] * ratio])
    return np.concatenate([left, mids, right])


# ---------------------------------------------------------------------------
# Method B: column-by-column crossing of P = level, restricted to S > S_min
# ---------------------------------------------------------------------------

def extract_boundary_B(
    P: np.ndarray,           # shape (Na, NS),  P[ia, iS]
    a_vals: np.ndarray,
    S_vals: np.ndarray,
    level: float = 0.5,
    S_floor: float = 3.0,
    polarized_floor: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each column ia, walk the grid in increasing S and find the cells
    that bracket the value `level`.  Interpolate linearly in log(S).
    Restrict to S > S_floor.  If several crossings exist, keep the one
    with the largest S (upper boundary).

    Sanity filter — skip columns where the column never reaches a truly
    polarised state.  We require   max_S P(a_k, S) >= polarized_floor
    (default 0.85).  Otherwise the "boundary" the algorithm would find
    is between two weakly-polarised regimes, not between phases.
    Set polarized_floor=None to disable.

    Multiple-crossings rule:
        At high <k> the dynamics has a *second* polarised phase at
        large S (rigid agents = inertia).  Each column then has TWO
        crossings: the upper edge of the low-S polarised region (the
        physical analogue of the aS=c boundary in low <k>), and the
        lower edge of the high-S polarised region (a different physical
        boundary).  We want the FIRST one — the smallest-S crossing.

    Returns (a_pts, S_pts) — same length, only valid columns included.
    """
    Na, NS = P.shape
    assert NS == len(S_vals) and Na == len(a_vals)

    a_pts: list[float] = []
    S_pts: list[float] = []

    logS = np.log(S_vals)

    for ia in range(Na):
        col = P[ia]                       # P(a_ia, S_iS)  iS = 0..NS-1

        # Sanity filter: column must contain a polarised cell somewhere
        if polarized_floor is not None:
            col_max = np.nanmax(col)
            if not (col_max >= polarized_floor):
                continue

        # candidate crossings:  iS such that the segment (iS, iS+1)
        # straddles `level` AND both endpoints are inside S > S_floor.
        crossings: list[float] = []        # store S* values

        for iS in range(NS - 1):
            S_lo, S_hi = S_vals[iS], S_vals[iS + 1]
            # require the segment to lie (at least partially) above the floor
            if S_hi <= S_floor:
                continue

            p_lo, p_hi = col[iS], col[iS + 1]
            if np.isnan(p_lo) or np.isnan(p_hi):
                continue

            # Strict bracket: one side >= level, the other side <= level
            # (handles both up- and down-crossings)
            if (p_lo - level) * (p_hi - level) > 0:
                continue
            if p_hi == p_lo:               # flat segment exactly at level
                S_star = 0.5 * (S_lo + S_hi)
            else:
                # Linear interpolation in log-S space
                t = (level - p_lo) / (p_hi - p_lo)
                S_star = float(np.exp(logS[iS] + t * (logS[iS + 1] - logS[iS])))

            if S_star >= S_floor:
                crossings.append(S_star)

        if not crossings:
            continue

        # Largest-S crossing -- the upper boundary of the polarised
        # region (i.e. where P first drops below `level` as we walk
        # down from the unpolarised top).  For dense graphs (high <k>)
        # a column may have two crossings because of a second polarised
        # phase at very high S (rigid agents = inertia).  We want the
        # boundary that scales with a, not the constant-S "weak-regime"
        # cap; the upper crossing is the right one as long as the
        # cell at the *top* of the grid is unpolarised (see filter
        # below).
        S_star = max(crossings)
        a_pts.append(float(a_vals[ia]))
        S_pts.append(S_star)

    return np.asarray(a_pts), np.asarray(S_pts)


# ---------------------------------------------------------------------------
# Fit  log S = log c - alpha * log a
# ---------------------------------------------------------------------------

def fit_log_log(a_pts: np.ndarray, S_pts: np.ndarray):
    x = np.log(a_pts)
    y = np.log(S_pts)
    res = linregress(x, y)
    alpha       = -res.slope
    sigma_alpha = float(res.stderr)
    log_c       = float(res.intercept)
    sigma_log_c = float(res.intercept_stderr)
    c           = float(np.exp(log_c))
    sigma_c     = c * sigma_log_c
    r2          = float(res.rvalue ** 2)
    return alpha, sigma_alpha, c, sigma_c, r2


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _title(data, observable_tex: str) -> str:
    kind   = str(data["graph_kind"])
    n      = int(data["n"])
    k_avg  = int(data["k_avg"])
    pD     = float(data["pD"])
    w0     = float(data["w0"])
    T      = int(data["T"])
    n_runs = int(data["n_runs"])
    return (
        f"{observable_tex}     "
        f"[$\\langle k \\rangle = {k_avg}$]\n"
        f"{kind.upper()},  $N$={n},  "
        f"$p_D$={pD:.2f},  $w_0$={w0:.2f},  $T$={T},  runs/cell={n_runs}"
    )


def plot_main(
    out_png: Path,
    P: np.ndarray, a_vals: np.ndarray, S_vals: np.ndarray,
    a_pts: np.ndarray, S_pts: np.ndarray,
    alpha: float, sigma_alpha: float, c: float, sigma_c: float, r2: float,
    level: float, S_floor: float, data,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), P.T,
        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(_title(data, r"$P(T) = 1 - \bar{w}_\times(T)$"))
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    # Method-B points
    ax.plot(a_pts, S_pts, "o",
            mfc="white", mec="black", ms=5, lw=0,
            label=rf"$S^*(a)$  (method B, $P={level}$, $S>{S_floor:g}$)")

    # Fitted power law  S = c * a^{-alpha}
    a_line = np.geomspace(a_vals[0], a_vals[-1], 200)
    S_fit  = c * a_line ** (-alpha)
    inside = (S_fit >= S_vals[0]) & (S_fit <= S_vals[-1])
    ax.plot(
        a_line[inside], S_fit[inside], "r-", lw=2,
        label=(rf"fit:  $\alpha={alpha:.2f}\pm{sigma_alpha:.2f}$,  "
               rf"$c={c:.2g}\pm{sigma_c:.1g}$  ($R^2={r2:.2f}$)"),
    )

    # alpha = 1 reference, anchored at the median of the extracted points
    a_med = float(np.exp(np.median(np.log(a_pts))))
    S_med = float(np.exp(np.median(np.log(S_pts))))
    c_ref = a_med * S_med                     # so that aS = c_ref passes through median
    S_ref = c_ref / a_line
    inside_ref = (S_ref >= S_vals[0]) & (S_ref <= S_vals[-1])
    ax.plot(
        a_line[inside_ref], S_ref[inside_ref], "r--", lw=1.2, alpha=0.8,
        label=rf"reference $aS={c_ref:.2g}$  ($\alpha=1$, through median)",
    )

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")


def plot_variance(
    out_png: Path,
    V: np.ndarray, a_vals: np.ndarray, S_vals: np.ndarray,
    a_pts: np.ndarray, S_pts: np.ndarray,
    S_floor: float, data,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))

    vmax = float(np.nanmax(V))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    pcm = ax.pcolormesh(
        _log_edges(a_vals), _log_edges(S_vals), V.T,
        cmap="magma", vmin=0.0, vmax=vmax, shading="flat",
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open,  100 = rigid)")
    ax.set_title(_title(data, r"$\mathrm{Var}_{\mathrm{runs}}\,P(T)$  (susceptibility)"))
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"sample variance over runs  (ddof=1)")

    # Overlay the S*(a) curve from method B
    if len(a_pts) >= 2:
        ax.plot(a_pts, S_pts, "o-",
                mfc="white", mec="cyan", color="cyan",
                ms=5, lw=1.2,
                label=rf"$S^*(a)$ from method B  ($S>{S_floor:g}$)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[save]  {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(npz_path: Path, level: float, S_floor: float,
        polarized_floor: float) -> None:
    data = np.load(npz_path, allow_pickle=False)
    P      = data["grid"]
    a_vals = data["a_vals"]
    S_vals = data["S_vals"]

    a_pts, S_pts = extract_boundary_B(
        P, a_vals, S_vals,
        level=level, S_floor=S_floor,
        polarized_floor=polarized_floor,
    )
    if len(a_pts) < 5:
        raise RuntimeError(
            f"Method B kept only {len(a_pts)} columns — too few for a fit."
        )

    alpha, sa, c, sc, r2 = fit_log_log(a_pts, S_pts)
    z = (alpha - 1.0) / sa if sa > 0 else float("nan")

    print(f"\n[fit B]  P={level} crossings on  {npz_path.name}")
    print(f"          columns used         : {len(a_pts)} / {len(a_vals)}")
    print(f"          alpha                : {alpha:.4f}  +/-  {sa:.4f}")
    print(f"          c                    : {c:.4g}  +/-  {sc:.2g}")
    print(f"          R^2                  : {r2:.4f}")
    print(f"          z(alpha=1)           : {z:+.2f} sigma "
          f"({'compatible' if abs(z) < 2 else 'rejected'} with aS=c)")

    stem = npz_path.stem
    plot_main(
        npz_path.with_name(stem + "_fit_B.png"),
        P, a_vals, S_vals, a_pts, S_pts,
        alpha, sa, c, sc, r2,
        level, S_floor, data,
    )

    if "grid_var" in data.files:
        V = data["grid_var"]
        plot_variance(
            npz_path.with_name(stem + "_fit_B_var.png"),
            V, a_vals, S_vals, a_pts, S_pts, S_floor, data,
        )
    else:
        print("[skip]  grid_var not in file — variance heatmap omitted.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path(__file__).with_name(
            "heatmap_k_variable_results_er_k004.npz"
        ),
    )
    p.add_argument("--level",  type=float, default=0.5)
    p.add_argument("--S-min",  type=float, default=3.0, dest="S_floor",
                   help="ignore P=level crossings at S below this value")
    p.add_argument("--polarized-floor", type=float, default=0.85,
                   dest="polarized_floor",
                   help="skip columns whose max P stays below this value "
                        "(no real polarised phase). Use --polarized-floor 0 to disable.")
    args = p.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)

    pf = args.polarized_floor if args.polarized_floor > 0 else None
    run(args.npz, level=args.level, S_floor=args.S_floor,
        polarized_floor=pf)


if __name__ == "__main__":
    main()
