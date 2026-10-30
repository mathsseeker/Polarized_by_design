"""Sanity check: vectorised simulate_final_cross_weight should match
toy_model4's step_opinions/step_weights when fed the same RNG.

For NumPy's default_rng, rng.random() called n times returns the same
n values as one rng.random(n) call - so a synchronous update with the
same flip_probs and same RNG seed will produce an identical trajectory.
"""
import numpy as np
import networkx as nx
import sys
sys.path.insert(0, ".")

from toy_model4 import compute_flip_prob, step_weights, initialise, make_graph
from heatmap_experiment import (
    simulate_final_cross_weight,
    make_fixed_graph_arrays,
    make_fixed_initial_state,
)


def reference_run(G, state0, gamma, a, w0, T, seed):
    """Mirror of toy_model4 dynamics, but with synchronous flip-decision RNG
    drawn as a single rng.random(n) so it matches the vectorised core."""
    rng = np.random.default_rng(seed)
    state = state0.copy().astype(np.int64)  # toy_model4 uses int64
    for u, v in G.edges():
        G[u][v]["weight"] = w0
    n = state.size
    for _ in range(T):
        flip_probs = np.array([compute_flip_prob(i, G, state, gamma) for i in range(n)])
        u = rng.random(n)
        flips = u < flip_probs
        state[flips] ^= 1
        step_weights(G, state, a)
    final_cross = []
    for u, v in G.edges():
        if state[u] != state[v]:
            final_cross.append(G[u][v]["weight"])
    return float(np.mean(final_cross)) if final_cross else float("nan")


def vectorised_run(eu, ev, n, state0, gamma, a, w0, T, seed):
    return simulate_final_cross_weight(eu, ev, n, state0, gamma, a, w0, T, seed)


if __name__ == "__main__":
    n = 120
    avg_k = 8
    seed_g = 7
    G = make_graph("er", n, avg_k, seed=seed_g)
    eu, ev = make_fixed_graph_arrays(n, avg_k, seed=seed_g)
    # rebuild graph the same way the vectorised code does (same edge list)
    state0 = make_fixed_initial_state(n, seed=11)

    cases = [
        # (gamma, a, w0, T, sim_seed)
        (0.0, 0.1, 0.5, 200, 1),    # no flipping; weights split
        (0.5, 0.0, 0.5, 200, 2),    # no weight change; pure opinion
        (0.3, 0.05, 0.5, 200, 3),   # mixed
        (1.0, 0.5, 0.5, 200, 4),    # strong
    ]
    for gamma, a, w0, T, sd in cases:
        ref = reference_run(G, state0, gamma, a, w0, T, sd)
        vec = vectorised_run(eu, ev, n, state0, gamma, a, w0, T, sd)
        ok = (np.isnan(ref) and np.isnan(vec)) or abs(ref - vec) < 1e-9
        tag = "OK " if ok else "FAIL"
        print(f"  {tag}  gamma={gamma:.2f} a={a:.2f}  ref={ref:.6f}  vec={vec:.6f}  diff={abs(ref-vec):.2e}")
