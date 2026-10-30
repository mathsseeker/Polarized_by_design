#!/usr/bin/env python3
"""
Vectorised batch implementation of the *Polarized by Design* dynamics.

Given a graph G, runs ``n_runs`` independent simulations of T timesteps in
parallel using NumPy array operations on (n_runs, n_nodes) state matrices
and (n_runs, n_edges) edge-weight matrices.

Mathematically identical to looping over n_runs calls of
``polarized.step_opinions`` / ``polarized.step_weights`` — same model,
same statistics — but ~10–20× faster because the per-step Python loops
over nodes and edges are replaced by vectorised numpy ops.

The flip-probability identity used:

    flip_prob_i  = 1 - prod_{j ~ i, s_j != s_i} (1 - w_ij / S)
                 = 1 - exp( sum_{j ~ i, disagree} log(1 - w_ij / S) )

The sum-over-neighbours is computed by scatter-adding per-edge log factors
onto node accumulators via a (sparse) incidence matrix multiplication.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def run_cell_vectorized(
    G: nx.Graph,
    n_runs: int,
    T: int,
    S: float,
    a: float,
    w0: float,
    pD: float,
    base_seed: int,
    state_seed: int,
) -> np.ndarray:
    """Run ``n_runs`` independent simulations of (S, a, T) dynamics on G.

    Parameters
    ----------
    G          : fixed graph (networkx, undirected)
    n_runs     : number of independent simulations to batch
    T          : number of timesteps per simulation
    S          : stubbornness  (>= 1)
    a          : Hebbian weight step
    w0         : initial edge weight
    pD         : initial fraction of opinion-1 nodes  (per run)
    base_seed  : seed for the dynamical RNG  (single stream feeding all runs)
    state_seed : seed for the initial-state RNGs  (one per run: state_seed + r)

    Returns
    -------
    P_runs     : (n_runs,) array of final polarization
                 P(T) = 1 - mean( w_{ij}(T)  over cross-opinion edges ).
                 If at T there are no cross-opinion edges, P = 1 by convention.
    """
    n_nodes = G.number_of_nodes()
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return np.ones(n_runs, dtype=np.float64)

    u_idx = edges[:, 0]
    v_idx = edges[:, 1]
    n_edges = edges.shape[0]

    # ------------------------------------------------------------------
    # Incidence matrix  M ∈ {0,1}^(n_edges × n_nodes)
    #
    #   M[e, i] = 1   iff   node i is an endpoint of edge e
    #
    # Used to scatter-add per-edge log factors onto node accumulators:
    #   node_log_accum (n_runs, n_nodes)  =  log_factor (n_runs, n_edges) @ M
    # ------------------------------------------------------------------
    M = np.zeros((n_edges, n_nodes), dtype=np.float64)
    edge_range = np.arange(n_edges)
    M[edge_range, u_idx] = 1.0
    M[edge_range, v_idx] = 1.0

    # ------------------------------------------------------------------
    # Initial states (one independent draw per run, same Bernoulli(pD)).
    # ------------------------------------------------------------------
    states = np.empty((n_runs, n_nodes), dtype=np.int8)
    for r in range(n_runs):
        rng_init = np.random.default_rng(state_seed + r)
        states[r] = (rng_init.random(n_nodes) < pD).astype(np.int8)

    # All edge weights start at w0.
    weights = np.full((n_runs, n_edges), w0, dtype=np.float64)

    # Single master RNG for the dynamics; independence across runs comes from
    # drawing a fresh (n_runs, n_nodes) uniform matrix each timestep.
    rng = np.random.default_rng(base_seed)

    # ------------------------------------------------------------------
    # Time loop  (each iteration vectorised across all n_runs simulations)
    # ------------------------------------------------------------------
    for _ in range(T):
        # ---- step_opinions  (synchronous flip decision) ----
        u_st = states[:, u_idx]                       # (n_runs, n_edges)
        v_st = states[:, v_idx]
        disagree = u_st != v_st                       # (n_runs, n_edges)

        # log(1 - w/S) on disagreeing edges, 0 elsewhere.
        # log1p is numerically stable; w/S <= 1 since S >= 1 and w in [0,1].
        log_factor = np.where(disagree, np.log1p(-weights / S), 0.0)

        # Scatter-add log factors onto node endpoints via matmul.
        node_log_accum = log_factor @ M               # (n_runs, n_nodes)
        flip_probs = 1.0 - np.exp(node_log_accum)

        # Sample synchronous flips.
        u01 = rng.random((n_runs, n_nodes))
        flips = u01 < flip_probs
        states = np.where(flips, 1 - states, states).astype(np.int8)

        # ---- step_weights  (+a if endpoints agree, -a otherwise, clip [0,1]) ----
        u_st = states[:, u_idx]
        v_st = states[:, v_idx]
        agree = u_st == v_st
        weights += np.where(agree, a, -a)
        np.clip(weights, 0.0, 1.0, out=weights)

    # ------------------------------------------------------------------
    # Final polarization P(T) per run
    # ------------------------------------------------------------------
    u_st = states[:, u_idx]
    v_st = states[:, v_idx]
    cross_mask = u_st != v_st                          # (n_runs, n_edges)

    P_runs = np.empty(n_runs, dtype=np.float64)
    for r in range(n_runs):
        cw = weights[r][cross_mask[r]]
        P_runs[r] = 1.0 - float(cw.mean()) if cw.size > 0 else 1.0

    return P_runs
