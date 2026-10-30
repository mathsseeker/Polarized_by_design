"""
regime_homophily_hist.py
=========================

Targeted resimulation at chosen regime points + per-node weighted homophily
h_i analysis, for the ER ⟨k⟩=8 sweep.

h_i = (sum_j w_ij  over neighbours j with x_j == x_i)
      / (sum_j w_ij  over all neighbours j)

We reproduce the exact RNG / seed convention of the original sweep
(`heatmap_k_variable.py:_cell_worker`):

    cell_seed = base_seed + 1_000_003 * idx_a + 1_009 * idx_S
    state[r]  ~ Bernoulli(pD), rng = default_rng(state_seed + r)
    dynamics rng: single default_rng(cell_seed), draws (n_runs, N) per step

so the resimulated cells are drawn from the SAME distribution as the heatmap.

Output:
    - regime_homophily_hist.png        (4 rows of histograms)
    - regime_homophily_summary.txt     (table per regime)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from polarized import make_graph


# ---------------------------------------------------------------------------
# Parameters — must match the sweep that produced the .npz
# ---------------------------------------------------------------------------
HEATMAP_NPZ = Path(__file__).with_name("heatmap_k_variable_results_er_k008.npz")
OUT_PNG     = Path(__file__).with_name("regime_homophily_hist.png")
OUT_TXT     = Path(__file__).with_name("regime_homophily_summary.txt")

# Regime points from pick_regime_points.py (ER, <k>=8)
POINTS = [
    # (label, a, S, P_grid_at_cell)
    ("low",            0.00734,  12.115,  0.114),
    ("mid (P≈0.45)",   0.00889,  68.129,  0.486),
    ("mid (P≈0.75)",   0.05000,  17.783,  0.705),
    ("high",           0.03406,  56.234,  1.000),
]


# ---------------------------------------------------------------------------
# Vectorised simulator that ALSO returns final (state, W) per run
# (adapted from polarized_vec.run_cell_vectorized — same model, same seeds)
# ---------------------------------------------------------------------------

def run_cell_keep_state(G: nx.Graph, n_runs: int, T: int, S: float, a: float,
                       w0: float, pD: float, base_seed: int, state_seed: int):
    n_nodes = G.number_of_nodes()
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        raise ValueError("Graph has no edges")
    u_idx = edges[:, 0]; v_idx = edges[:, 1]
    n_edges = edges.shape[0]

    # Incidence matrix for scatter-add.
    M = np.zeros((n_edges, n_nodes), dtype=np.float64)
    e_range = np.arange(n_edges)
    M[e_range, u_idx] = 1.0
    M[e_range, v_idx] = 1.0

    # Initial states — one independent draw per run.
    states = np.empty((n_runs, n_nodes), dtype=np.int8)
    for r in range(n_runs):
        rng_init = np.random.default_rng(state_seed + r)
        states[r] = (rng_init.random(n_nodes) < pD).astype(np.int8)

    weights = np.full((n_runs, n_edges), w0, dtype=np.float64)
    rng = np.random.default_rng(base_seed)

    for _ in range(T):
        u_st = states[:, u_idx]
        v_st = states[:, v_idx]
        disagree = u_st != v_st
        log_factor = np.where(disagree, np.log1p(-weights / S), 0.0)
        node_log_accum = log_factor @ M
        flip_probs = 1.0 - np.exp(node_log_accum)
        u01 = rng.random((n_runs, n_nodes))
        flips = u01 < flip_probs
        states = np.where(flips, 1 - states, states).astype(np.int8)

        u_st = states[:, u_idx]; v_st = states[:, v_idx]
        agree = u_st == v_st
        weights += np.where(agree, a, -a)
        np.clip(weights, 0.0, 1.0, out=weights)

    return states, weights, edges      # shapes: (n_runs, N), (n_runs, n_edges), (n_edges, 2)


# ---------------------------------------------------------------------------
# Compute h_i per node, per run
# ---------------------------------------------------------------------------

def homophily_per_run(states: np.ndarray, weights: np.ndarray, edges: np.ndarray):
    """Return (h, valid_mask) of shapes (n_runs, n_nodes), where h[r, i] is the
    weighted homophily of node i in run r, and valid_mask[r, i] is False for
    nodes that should be excluded (no neighbours OR total weight == 0)."""
    n_runs, n_nodes = states.shape
    u = edges[:, 0]; v = edges[:, 1]

    # Per-edge agreement bool, per run, then accumulate to nodes by endpoint.
    h = np.zeros((n_runs, n_nodes), dtype=np.float64)
    w_total_node = np.zeros((n_runs, n_nodes), dtype=np.float64)
    w_agree_node = np.zeros((n_runs, n_nodes), dtype=np.float64)

    for r in range(n_runs):
        w_e = weights[r]
        agree_e = states[r, u] == states[r, v]
        # add edge weight to BOTH endpoints
        np.add.at(w_total_node[r], u, w_e)
        np.add.at(w_total_node[r], v, w_e)
        w_agree_e = np.where(agree_e, w_e, 0.0)
        np.add.at(w_agree_node[r], u, w_agree_e)
        np.add.at(w_agree_node[r], v, w_agree_e)

    valid = w_total_node > 0
    np.divide(w_agree_node, w_total_node, out=h, where=valid)
    return h, valid, w_total_node


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    d = np.load(HEATMAP_NPZ, allow_pickle=False)
    a_vals = d["a_vals"]; S_vals = d["S_vals"]
    n = int(d["n"]); k_avg = int(d["k_avg"]); graph_kind = str(d["graph_kind"])
    pD = float(d["pD"]); w0 = float(d["w0"])
    T = int(d["T"]); n_runs = int(d["n_runs"])
    graph_seed = int(d["graph_seed"]); state_seed = int(d["state_seed"])
    base_seed_root = int(d["base_seed"])

    # Build the graph once (same for every cell in the sweep).
    G = make_graph(graph_kind, n, k_avg, seed=graph_seed)
    print(f"[graph]  {graph_kind} N={n} <k>={k_avg}  edges={G.number_of_edges()}")

    # Find which (ia, iS) cell each chosen point lives in (nearest in log-space).
    def nearest_idx(x, arr):
        return int(np.argmin(np.abs(np.log(arr) - np.log(x))))

    results = []   # list of dicts per regime

    for label, a_val, S_val, P_at_cell in POINTS:
        ia = nearest_idx(a_val, a_vals)
        iS = nearest_idx(S_val, S_vals)
        cell_seed = base_seed_root + 1_000_003 * ia + 1_009 * iS
        print(f"\n[run]    {label}: a={a_vals[ia]:.5f} S={S_vals[iS]:.3f}  "
              f"ia={ia} iS={iS}  cell_seed={cell_seed}")

        states, weights, edges = run_cell_keep_state(
            G, n_runs=n_runs, T=T, S=S_vals[iS], a=a_vals[ia],
            w0=w0, pD=pD, base_seed=cell_seed, state_seed=state_seed,
        )
        h, valid, w_total = homophily_per_run(states, weights, edges)

        # Sanity: report polarisation we measured here vs the heatmap.
        # P_run = 1 - mean cross-opinion weight  (same convention as the sweep).
        u = edges[:, 0]; v = edges[:, 1]
        P_runs = np.empty(n_runs)
        for r in range(n_runs):
            cross = states[r, u] != states[r, v]
            cw = weights[r][cross]
            P_runs[r] = 1.0 - float(cw.mean()) if cw.size > 0 else 1.0

        print(f"         resim P_runs: mean={P_runs.mean():.3f}  "
              f"std={P_runs.std():.3f}  (heatmap cell P={P_at_cell:.3f})")

        results.append(dict(
            label=label, a=float(a_vals[ia]), S=float(S_vals[iS]),
            P_cell=P_at_cell, P_resim=float(P_runs.mean()),
            states=states, valid=valid, h=h, w_total=w_total,
        ))

    # ----------------------------------------------------------------
    # Histogram grid
    # ----------------------------------------------------------------
    bins = np.linspace(0.0, 1.0, 21)        # 20 bins of width 0.05
    n_regimes = len(results)

    fig, axes = plt.subplots(n_regimes, 1, figsize=(8, 2.6 * n_regimes), sharex=True)
    if n_regimes == 1:
        axes = [axes]

    # Compute global y-max so panels share scale.
    panel_data = []   # store (h_F, h_D, label, stats) for each regime
    y_max = 0.0
    for r in results:
        states = r["states"]; valid = r["valid"]; h = r["h"]
        # Pool across runs, but keep F/D label.
        F_mask = (states == 0) & valid
        D_mask = (states == 1) & valid
        h_F = h[F_mask]
        h_D = h[D_mask]
        # Estimate the largest bin-height we'll plot (fraction).
        if h_F.size > 0:
            hf, _ = np.histogram(h_F, bins=bins, density=False)
            y_max = max(y_max, float(hf.max()) / h_F.size)
        if h_D.size > 0:
            hd, _ = np.histogram(h_D, bins=bins, density=False)
            y_max = max(y_max, float(hd.max()) / h_D.size)
        panel_data.append((h_F, h_D, r))
    y_top = y_max * 1.10 if y_max > 0 else 1.0

    for ax, (h_F, h_D, r) in zip(axes, panel_data):
        if h_F.size > 0:
            ax.hist(h_F, bins=bins, weights=np.full_like(h_F, 1.0/h_F.size),
                    color="tab:red", alpha=0.55, edgecolor="black", lw=0.4,
                    label=f"F (state=0)  n={h_F.size}")
        if h_D.size > 0:
            ax.hist(h_D, bins=bins, weights=np.full_like(h_D, 1.0/h_D.size),
                    color="tab:blue", alpha=0.55, edgecolor="black", lw=0.4,
                    label=f"D (state=1)  n={h_D.size}")

        # Threshold line + echo-chamber fraction.
        ax.axvline(0.9, color="black", lw=0.8, ls="--")
        h_all = np.concatenate([h_F, h_D]) if (h_F.size + h_D.size) > 0 else np.array([])
        frac_above = float((h_all > 0.9).sum()) / max(1, h_all.size)
        ax.text(0.905, y_top * 0.92,
                f">0.9 : {frac_above*100:.1f}% of nodes",
                fontsize=8, va="top", ha="left")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_top)
        ax.set_ylabel("fraction of nodes")
        ax.set_title(
            f"{r['label']}    a={r['a']:.4g},  S={r['S']:.4g}    "
            f"P_cell={r['P_cell']:.3f}  (resim mean P={r['P_resim']:.3f})",
            fontsize=9, loc="left",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel(r"$h_i$  (weighted homophily)")
    fig.suptitle(
        f"Weighted homophily $h_i$ at four regime points  "
        f"[ER, N={n}, ⟨k⟩={k_avg}, T={T}, runs/cell={n_runs}]  — final state only",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"\n[save] {OUT_PNG}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append("Regime homophily summary (ER, ⟨k⟩=8, N=200, T=1000, 20 runs/cell)")
    lines.append("=" * 78)
    lines.append("")
    header = (f"{'regime':<14} {'a':>8} {'S':>8} {'P':>6}  "
              f"{'mean h':>7}  {'>0.9':>6}  {'h=1':>6}  {'h=0':>6}  "
              f"{'excl':>5}  {'n_F':>4}  {'n_D':>4}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        h = r["h"]; valid = r["valid"]; states = r["states"]
        n_total = h.size
        n_excl  = int((~valid).sum())
        h_all = h[valid]
        mean_h = float(h_all.mean()) if h_all.size > 0 else float("nan")
        frac_gt09 = float((h_all > 0.9).sum()) / max(1, h_all.size)
        frac_eq1  = float((h_all == 1.0).sum()) / max(1, h_all.size)
        frac_eq0  = float((h_all == 0.0).sum()) / max(1, h_all.size)
        n_F = int((valid & (states == 0)).sum())
        n_D = int((valid & (states == 1)).sum())
        lines.append(
            f"{r['label']:<14} {r['a']:>8.4f} {r['S']:>8.3f} {r['P_cell']:>6.3f}  "
            f"{mean_h:>7.3f}  {frac_gt09*100:>5.1f}%  {frac_eq1*100:>5.1f}%  "
            f"{frac_eq0*100:>5.1f}%  {n_excl:>5d}  {n_F:>4d}  {n_D:>4d}"
        )

        # Broken down by opinion
        for op_name, op_val in [("F", 0), ("D", 1)]:
            mask = valid & (states == op_val)
            hh = h[mask]
            if hh.size == 0:
                lines.append(f"   {op_name}: <no nodes>")
                continue
            mh = float(hh.mean())
            gt = float((hh > 0.9).sum()) / hh.size
            eq1 = float((hh == 1.0).sum()) / hh.size
            eq0 = float((hh == 0.0).sum()) / hh.size
            lines.append(
                f"   {op_name}: mean h={mh:.3f}  >0.9={gt*100:.1f}%  "
                f"h=1={eq1*100:.1f}%  h=0={eq0*100:.1f}%  n={hh.size}"
            )
        lines.append("")

    OUT_TXT.write_text("\n".join(lines))
    print(f"[save] {OUT_TXT}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
