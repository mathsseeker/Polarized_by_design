#!/usr/bin/env python3
"""
Heterogeneity of S_i (Zheng et al.): three experiments on the polarized
model with per-node stubbornness S_i.

All experiments fix <S> = 10 and only vary the *shape* of the S_i
distribution, so the comparison isolates the effect of heterogeneity
(not of the average).

(A) Trajectories P(t) vs t for three distributions of S_i
    (const, normal, power-law) — same mean, different variance.

(B) Variance sweep: P_final vs Var(S_i) using truncated-normal S_i
    with mean=10 and varying std.

(C) Final edge-weight histograms split into
    same-opinion vs cross-opinion edges, for the three distributions of (A).

Outputs:
    experiment_hetS_Pt.png
    experiment_hetS_variance.png
    experiment_hetS_histograms.png
    experiment_hetS_results.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph
from polarized_hetS import (
    make_S_distribution,
    run_simulation_hetS,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

@dataclass
class Config:
    graph_kind: str = "ws"
    N: int = 200
    k: int = 8
    p_ws: float = 0.1
    a: float = 0.05
    w0: float = 0.5
    pD: float = 0.5
    T: int = 1000
    mean_S: float = 10.0
    # Experiment A / C
    n_runs_A: int = 30
    normal_std_A: float = 3.0
    # Experiment B
    n_runs_B: int = 20
    std_grid_B: tuple = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0)
    # Seeds
    graph_seed: int = 42
    state_seed_base: int = 1000
    sim_seed_base: int = 2000
    s_seed_base: int = 3000


# ---------------------------------------------------------------------------
# Experiment A:  P(t) trajectories for three distributions
# ---------------------------------------------------------------------------

DIST_LABELS = {
    "const":    r"const ($S_i = \langle S \rangle$, Var $=0$)",
    "normal":   r"normal ($\sigma = 3$)",
    "powerlaw": r"power-law (bounded Pareto)",
}
DIST_COLORS = {"const": "tab:blue", "normal": "tab:orange", "powerlaw": "tab:red"}


def experiment_A(cfg: Config):
    """Run T-step trajectories for 3 distributions × n_runs_A runs.

    Returns:
        traj      : dict[kind] -> (n_runs, T+1) array of P(t)
        final_W   : dict[kind] -> list of (final_W, final_state, A) tuples
        S_samples : dict[kind] -> (n_runs, N) array of S_i used per run
    """
    G = make_graph(cfg.graph_kind, cfg.N, cfg.k, p_ws=cfg.p_ws, seed=cfg.graph_seed)

    traj = {}
    final = {}
    S_samples = {}

    for kind in ("const", "normal", "powerlaw"):
        print(f"  [A] running {kind:9s} × {cfg.n_runs_A} runs, T={cfg.T}")
        t0 = time.time()
        P_runs = np.empty((cfg.n_runs_A, cfg.T + 1), dtype=np.float64)
        Ss = np.empty((cfg.n_runs_A, cfg.N), dtype=np.float64)
        finals = []
        for r in range(cfg.n_runs_A):
            rng_S = np.random.default_rng(cfg.s_seed_base + r)
            S_arr = make_S_distribution(
                cfg.N, kind=kind, mean=cfg.mean_S,
                std=cfg.normal_std_A, rng=rng_S,
            )
            res = run_simulation_hetS(
                G, S_arr=S_arr,
                a=cfg.a, w0=cfg.w0, pD=cfg.pD, T=cfg.T,
                init_seed=cfg.state_seed_base + r,
                sim_seed=cfg.sim_seed_base + r,
                record_traj=True,
            )
            P_runs[r] = res["P_t"]
            Ss[r] = S_arr
            finals.append((res["W"], res["state"], res["A"]))
        traj[kind] = P_runs
        final[kind] = finals
        S_samples[kind] = Ss
        print(f"      {time.time() - t0:.1f}s   "
              f"mean S = {Ss.mean():.2f}  std S = {Ss.std():.2f}  "
              f"P_final = {P_runs[:, -1].mean():.3f} ± {P_runs[:, -1].std():.3f}")

    return traj, final, S_samples


def plot_experiment_A(traj, cfg, out_png):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    t = np.arange(cfg.T + 1)
    for kind, P in traj.items():
        m = P.mean(axis=0)
        s = P.std(axis=0)
        c = DIST_COLORS[kind]
        ax.plot(t, m, color=c, lw=2, label=DIST_LABELS[kind])
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.18)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$P(t) = 1 - \bar{w}_\times(t)$")
    ax.set_ylim(0, 1)
    ax.set_title(
        rf"$P(t)$ under three $S_i$ distributions (same $\langle S \rangle = {cfg.mean_S:g}$)"
        "\n"
        f"{cfg.graph_kind.upper()},  $N$={cfg.N},  $\\langle k \\rangle$={cfg.k},  "
        f"$a$={cfg.a},  $w_0$={cfg.w0},  $p_D$={cfg.pD},  runs={cfg.n_runs_A}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_png}")


# ---------------------------------------------------------------------------
# Experiment B:  P_final vs Var(S) for truncated-normal S_i
# ---------------------------------------------------------------------------

def experiment_B(cfg: Config):
    """Sweep std of normal-distributed S_i (mean fixed) and record P_final."""
    G = make_graph(cfg.graph_kind, cfg.N, cfg.k, p_ws=cfg.p_ws, seed=cfg.graph_seed)

    stds = np.array(cfg.std_grid_B, dtype=np.float64)
    P_means = np.empty(len(stds))
    P_stds = np.empty(len(stds))
    var_emp = np.empty(len(stds))
    mean_emp = np.empty(len(stds))

    for j, std in enumerate(stds):
        t0 = time.time()
        kind = "const" if std == 0.0 else "normal"
        P_final = np.empty(cfg.n_runs_B)
        S_means = np.empty(cfg.n_runs_B)
        S_vars = np.empty(cfg.n_runs_B)
        for r in range(cfg.n_runs_B):
            rng_S = np.random.default_rng(cfg.s_seed_base + 100 + 10 * j + r)
            S_arr = make_S_distribution(
                cfg.N, kind=kind, mean=cfg.mean_S, std=float(std), rng=rng_S,
            )
            res = run_simulation_hetS(
                G, S_arr=S_arr,
                a=cfg.a, w0=cfg.w0, pD=cfg.pD, T=cfg.T,
                init_seed=cfg.state_seed_base + 100 + 10 * j + r,
                sim_seed=cfg.sim_seed_base + 100 + 10 * j + r,
                record_traj=False,
            )
            P_final[r] = res["P_final"]
            S_means[r] = S_arr.mean()
            S_vars[r] = S_arr.var()
        P_means[j] = P_final.mean()
        P_stds[j] = P_final.std()
        var_emp[j] = S_vars.mean()
        mean_emp[j] = S_means.mean()
        print(f"  [B] std={std:5.1f}  Var(S)≈{var_emp[j]:7.2f}  "
              f"<S>≈{mean_emp[j]:5.2f}  P_final={P_means[j]:.3f}±{P_stds[j]:.3f}  "
              f"({time.time()-t0:.1f}s)")

    return stds, P_means, P_stds, var_emp, mean_emp


def plot_experiment_B(stds, P_means, P_stds, var_emp, mean_emp, cfg, out_png):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(var_emp, P_means, yerr=P_stds, marker="o", lw=2,
                color="tab:blue", capsize=3, label=r"$\overline{P(T)}$")
    ax.set_xlabel(r"empirical Var($S_i$)")
    ax.set_ylabel(r"$P_\infty \approx P(T)$")
    stds_str = ", ".join(f"{float(s):g}" for s in stds)
    ax.set_title(
        rf"$P_\infty$ vs heterogeneity of $S_i$"
        "\n"
        rf"normal($\langle S \rangle$=10, $\sigma\in\{{{stds_str}\}}$), "
        f"{cfg.graph_kind.upper()}, $N$={cfg.N}, $T$={cfg.T}, runs={cfg.n_runs_B}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    # Annotate sigma at each point (above the marker so they don't collide
    # with overlapping x-axis labels at small variance).
    for v, s, p in zip(var_emp, stds, P_means):
        ax.annotate(rf"$\sigma$={float(s):g}", (v, p), xytext=(6, -2),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_png}")


# ---------------------------------------------------------------------------
# Experiment C:  histograms of final w_ij, split same vs cross-opinion edges
# ---------------------------------------------------------------------------

def collect_final_weights(final_runs, kind):
    """For all runs of one distribution kind, return two flat arrays of
    final edge weights: (same-opinion, cross-opinion)."""
    same_all = []
    cross_all = []
    for W, state, A in final_runs:
        # upper triangle of A to count each edge once
        upper = np.triu(A, k=1)
        # disagreement on each pair
        D = state[:, None] != state[None, :]
        same_mask = upper & ~D
        cross_mask = upper & D
        same_all.append(W[same_mask])
        cross_all.append(W[cross_mask])
    return np.concatenate(same_all), np.concatenate(cross_all)


def plot_experiment_C(final, cfg, out_png):
    import matplotlib.pyplot as plt

    kinds = ("const", "normal", "powerlaw")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    bins = np.linspace(0.0, 1.0, 41)

    for i, kind in enumerate(kinds):
        same, cross = collect_final_weights(final[kind], kind)
        ax_s = axes[i, 0]
        ax_c = axes[i, 1]
        ax_s.hist(same,  bins=bins, color="tab:green", alpha=0.85, density=True)
        ax_c.hist(cross, bins=bins, color="tab:red",   alpha=0.85, density=True)
        ax_s.set_title(f"{DIST_LABELS[kind]} — same-opinion edges  (mean w={same.mean():.2f})",
                       fontsize=10)
        ax_c.set_title(f"{DIST_LABELS[kind]} — cross-opinion edges (mean w={cross.mean():.2f})",
                       fontsize=10)
        for ax in (ax_s, ax_c):
            ax.grid(alpha=0.3)
            ax.set_ylabel("density")

    axes[-1, 0].set_xlabel(r"$w_{ij}(T)$")
    axes[-1, 1].set_xlabel(r"$w_{ij}(T)$")
    fig.suptitle(
        rf"Final edge-weight distribution, $T={cfg.T}$  "
        f"(aggregated over {cfg.n_runs_A} runs)"
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [save] {out_png}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", choices=["ws", "er", "ba"], default="ws")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--a", type=float, default=0.05)
    p.add_argument("--w0", type=float, default=0.5)
    p.add_argument("--pD", type=float, default=0.5)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--mean-S", type=float, default=10.0)
    p.add_argument("--runs-A", type=int, default=30)
    p.add_argument("--runs-B", type=int, default=20)
    p.add_argument("--normal-std-A", type=float, default=3.0)
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Tiny defaults for a fast smoke test (N=80, T=200, runs=5).")
    p.add_argument("--only", choices=["A", "B", "C", "all"], default="all")
    args = p.parse_args()

    cfg = Config(
        graph_kind=args.graph, N=args.N, k=args.k,
        a=args.a, w0=args.w0, pD=args.pD, T=args.T,
        mean_S=args.mean_S, n_runs_A=args.runs_A, n_runs_B=args.runs_B,
        normal_std_A=args.normal_std_A, graph_seed=args.graph_seed,
    )
    if args.quick:
        cfg.N = 80
        cfg.T = 200
        cfg.n_runs_A = 5
        cfg.n_runs_B = 4
        cfg.std_grid_B = (0.0, 2.0, 5.0, 10.0)

    print(f"[setup] graph={cfg.graph_kind} N={cfg.N} k={cfg.k} a={cfg.a} "
          f"w0={cfg.w0} pD={cfg.pD} T={cfg.T} <S>={cfg.mean_S}")
    print()

    traj = final = S_samples = None
    expB = None

    t_total0 = time.time()
    if args.only in ("A", "all"):
        print("== Experiment A: P(t) trajectories ==")
        traj, final, S_samples = experiment_A(cfg)
        plot_experiment_A(traj, cfg, "experiment_hetS_Pt.png")
        print()

    if args.only in ("B", "all"):
        print("== Experiment B: variance sweep ==")
        expB = experiment_B(cfg)
        plot_experiment_B(*expB, cfg, "experiment_hetS_variance.png")
        print()

    if args.only in ("C", "all"):
        if final is None:
            print("== Experiment C requires A — running A now ==")
            traj, final, S_samples = experiment_A(cfg)
        print("== Experiment C: final-weight histograms ==")
        plot_experiment_C(final, cfg, "experiment_hetS_histograms.png")
        print()

    # Save aggregated numerics
    save = {}
    if traj is not None:
        for k, P in traj.items():
            save[f"A_traj_{k}"] = P
        for k, runs in final.items():
            # store flat same/cross weights instead of full W (saves disk)
            same, cross = collect_final_weights(runs, k)
            save[f"C_same_{k}"]  = same
            save[f"C_cross_{k}"] = cross
        for k, Ss in S_samples.items():
            save[f"A_Ssamples_{k}"] = Ss
    if expB is not None:
        stds, Pm, Ps, ve, me = expB
        save["B_stds"] = stds
        save["B_P_mean"] = Pm
        save["B_P_std"] = Ps
        save["B_var_emp"] = ve
        save["B_mean_emp"] = me

    np.savez("experiment_hetS_results.npz", **save)
    print(f"[save] experiment_hetS_results.npz")
    print(f"[done] total {time.time() - t_total0:.1f}s")


if __name__ == "__main__":
    main()
