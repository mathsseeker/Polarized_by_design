"""
pick_regime_points.py
=====================

Pick 3-4 representative (S, a) cells from a heatmap_k_variable_results_*.npz:
  - one in the low-P regime (not on a corner; neighbours similar)
  - one in the high-P regime (deep yellow, all neighbours saturated)
  - one or two in the steep-gradient mid-P transition region

Approach:
  - Convert P to (a-index, S-index) coordinates.
  - Compute 3x3 neighbourhood mean and std for every interior cell. Low std
    means the cell sits in a "regime" rather than at a sharp boundary.
  - Compute |grad P| in (log a, log S) index space via central differences.
  - Score candidates per regime; pick the best non-corner cell.

Outputs:
  - a PNG annotated with the chosen points
  - a small table of (a, S, P) printed to stdout
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


def neighbourhood_stats(P: np.ndarray, half: int = 1):
    """3x3 (default) neighbourhood mean and std for each interior cell."""
    Na, NS = P.shape
    mean = np.full_like(P, np.nan)
    std  = np.full_like(P, np.nan)
    for ia in range(Na):
        for iS in range(NS):
            ia0, ia1 = max(0, ia - half), min(Na, ia + half + 1)
            iS0, iS1 = max(0, iS - half), min(NS, iS + half + 1)
            block = P[ia0:ia1, iS0:iS1]
            mean[ia, iS] = block.mean()
            std[ia, iS]  = block.std()
    return mean, std


def grad_magnitude(P: np.ndarray):
    """|grad P| in (index, index) space via central differences."""
    dPa, dPs = np.gradient(P)
    return np.sqrt(dPa ** 2 + dPs ** 2)


def pick_low(P, nb_mean, nb_std, exclude_corner: int = 2):
    """Cell with low P + low neighbourhood std, away from grid corners."""
    Na, NS = P.shape
    score = P + 2.0 * nb_std            # want both small
    # Mask out the outermost `exclude_corner` rings to avoid edges/corners.
    mask = np.ones_like(P, dtype=bool)
    mask[:exclude_corner, :] = False
    mask[-exclude_corner:, :] = False
    mask[:, :exclude_corner] = False
    mask[:, -exclude_corner:] = False
    score_masked = np.where(mask, score, np.inf)
    ia, iS = np.unravel_index(np.argmin(score_masked), score_masked.shape)
    return ia, iS


def pick_high(P, nb_mean, exclude_corner: int = 2, P_thr: float = 0.999):
    """A cell where the whole 3x3 neighbourhood is fully polarised (deep yellow)."""
    Na, NS = P.shape
    mask = np.ones_like(P, dtype=bool)
    mask[:exclude_corner, :] = False
    mask[-exclude_corner:, :] = False
    mask[:, :exclude_corner] = False
    mask[:, -exclude_corner:] = False
    # Prefer the deepest-in-yellow point: maximise nb_mean among cells with P>=P_thr.
    candidates = np.where(mask & (P >= P_thr), nb_mean, -np.inf)
    ia, iS = np.unravel_index(np.argmax(candidates), candidates.shape)
    return ia, iS


def pick_mid(P, grad_mag, target_P: float, exclude_corner: int = 2,
             tol: float = 0.10):
    """High-gradient cell with P close to target_P."""
    mask = np.ones_like(P, dtype=bool)
    mask[:exclude_corner, :] = False
    mask[-exclude_corner:, :] = False
    mask[:, :exclude_corner] = False
    mask[:, -exclude_corner:] = False
    near_target = (np.abs(P - target_P) <= tol) & mask
    if not near_target.any():
        # widen
        near_target = (np.abs(P - target_P) <= tol * 2) & mask
    score = np.where(near_target, grad_mag, -np.inf)
    ia, iS = np.unravel_index(np.argmax(score), score.shape)
    return ia, iS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path,
                        default=Path(__file__).with_name(
                            "heatmap_k_variable_results_er_k008.npz"))
    parser.add_argument("--n-mid", type=int, default=2, choices=[1, 2],
                        help="how many mid-regime points")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    P = d["grid"]
    a_vals = d["a_vals"]
    S_vals = d["S_vals"]
    Na, NS = P.shape

    nb_mean, nb_std = neighbourhood_stats(P)
    grad = grad_magnitude(P)

    chosen: list[tuple[str, int, int]] = []
    ia, iS = pick_low(P, nb_mean, nb_std)
    chosen.append(("low",  ia, iS))
    ia, iS = pick_high(P, nb_mean)
    chosen.append(("high", ia, iS))
    if args.n_mid == 2:
        # one closer to the low side, one closer to the high side
        ia, iS = pick_mid(P, grad, target_P=0.45)
        chosen.append(("mid (P~0.45)", ia, iS))
        ia, iS = pick_mid(P, grad, target_P=0.75)
        chosen.append(("mid (P~0.75)", ia, iS))
    else:
        ia, iS = pick_mid(P, grad, target_P=0.55)
        chosen.append(("mid (P~0.55)", ia, iS))

    # ----- Print summary table -----
    print()
    print(f"{'regime':<14} {'ia':>3} {'iS':>3}   {'a':>8}  {'S':>8}   {'P':>6}   notes")
    print("-" * 90)
    for label, ia, iS in chosen:
        a = a_vals[ia]; S = S_vals[iS]; p = P[ia, iS]
        ia0, ia1 = max(0, ia - 1), min(Na, ia + 2)
        iS0, iS1 = max(0, iS - 1), min(NS, iS + 2)
        block = P[ia0:ia1, iS0:iS1]
        note = (f"3x3 mean={block.mean():.3f} std={block.std():.3f} "
                f"|grad|={grad[ia, iS]:.3f}")
        print(f"{label:<14} {ia:>3} {iS:>3}   {a:>8.4f}  {S:>8.3f}   {p:>6.3f}   {note}")

    print()
    print("--- copy-paste form ---")
    print("regime,        a,         S,         P")
    for label, ia, iS in chosen:
        print(f"{label:<14}, {a_vals[ia]:9.5f}, {S_vals[iS]:9.5f}, {P[ia, iS]:6.4f}")

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(9, 7))
    pcm = ax.pcolormesh(_log_edges(a_vals), _log_edges(S_vals), P.T,
                        cmap="viridis", vmin=0.0, vmax=1.0, shading="flat")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$a$ — Hebbian weight step")
    ax.set_ylabel(r"$S$ — stubbornness")
    ax.set_title(
        f"$P(T)$ heatmap with regime points  "
        f"[{str(d['graph_kind']).upper()}, N={int(d['n'])}, "
        f"$\\langle k\\rangle$={int(d['k_avg'])}]"
    )
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r"$P(T)$  (0 = unpolarised, 1 = fully polarised)")

    # Tab10 color cycle for the points
    colors = {"low": "tab:red", "high": "tab:orange",
              "mid (P~0.45)": "tab:cyan", "mid (P~0.75)": "tab:pink",
              "mid (P~0.55)": "tab:cyan"}
    for label, ia, iS in chosen:
        a = a_vals[ia]; S = S_vals[iS]; p = P[ia, iS]
        ax.plot(a, S, marker="*", color=colors.get(label, "white"),
                mec="black", mew=1.0, ms=18, lw=0, zorder=6)
        # Offset the label so the star is visible
        ax.annotate(
            f"{label}\n(a={a:.3g}, S={S:.3g})\nP={p:.3f}",
            xy=(a, S), xytext=(8, 8), textcoords="offset points",
            color="black", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.5,
                      boxstyle="round,pad=0.3"),
            zorder=7,
        )

    plt.tight_layout()
    out_png = args.npz.with_name(args.npz.stem + "_regime_points.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[save] {out_png}")


if __name__ == "__main__":
    main()
