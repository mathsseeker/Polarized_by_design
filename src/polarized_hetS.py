#!/usr/bin/env python3
"""
Heterogeneous stubbornness extension of the polarization model.

In polarized.py / polarized_fast.py the stubbornness S is a single scalar
shared by all nodes. Here S becomes a per-node array S_i and the flip
probability of node i uses S_i (not S_j of its neighbours):

    flip_prob_i = 1 - prod_{j ~ i, s_j != s_i} (1 - w_ij / S_i)

In the dense-matrix formulation (polarized_fast.step_opinions_fast):

    factor = np.where(M, 1.0 - W / S, 1.0)     # S scalar
    p_flip = 1.0 - factor.prod(axis=1)

becomes

    factor = np.where(M, 1.0 - W / S_arr[:, None], 1.0)   # S_arr shape (N,)
    p_flip = 1.0 - factor.prod(axis=1)

because broadcasting divides row i of W by S_arr[i] — exactly the S of the
deciding node. step_weights_fast and polarization_coefficient_fast are
unchanged (no S inside).

Equivalence with the scalar-S code is checked in `test_equivalence_hetS`:
with S_arr = S * ones(N), the dynamics must be bit-identical.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polarized import make_graph
from polarized_fast import (
    initialise_fast,
    step_opinions_fast,
    step_weights_fast,
    polarization_coefficient_fast,
)


# ---------------------------------------------------------------------------
# Heterogeneous-S dynamics
# ---------------------------------------------------------------------------

def step_opinions_hetS(A, W, state, S_arr, rng):
    """Synchronous flip update with per-node stubbornness S_arr.

    Identical to polarized_fast.step_opinions_fast except S is an (N,)
    array. Row i of W is divided by S_arr[i] via broadcasting, so node i
    uses its own S_i in the product of (1 - w/S_i) over disagreeing
    neighbours.
    """
    D = state[:, None] != state[None, :]
    M = A & D
    factor = np.where(M, 1.0 - W / S_arr[:, None], 1.0)
    p_flip = 1.0 - factor.prod(axis=1)
    flips = rng.random(state.size) < p_flip
    return np.where(flips, 1 - state, state).astype(np.int8)


# ---------------------------------------------------------------------------
# S_i distributions
# ---------------------------------------------------------------------------

def make_S_distribution(
    N: int,
    kind: str,
    mean: float,
    std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    lo: float = 1.0,
    hi: float = 100.0,
) -> np.ndarray:
    """Sample N values of S_i from one of three distributions.

    Parameters
    ----------
    N    : number of nodes
    kind : "const", "normal", or "powerlaw"
    mean : target population mean (clipping to [lo, hi] introduces a small
           empirical deviation; callers can inspect the returned array)
    std  : standard deviation (used by "normal" only; ignored otherwise)
    rng  : numpy Generator (default: np.random.default_rng())
    lo   : lower clip (default 1.0 — required by the model since w/S <= 1)
    hi   : upper clip (default 100.0 — same as the paper's S range)

    Distributions:
    - "const":    S_i = mean for all i.  Variance = 0.
    - "normal":   S_i ~ N(mean, std^2), clipped to [lo, hi].
    - "powerlaw": Bounded Pareto with shape alpha = mean / (mean - 1),
                  giving theoretical mean = `mean` (unbounded) and a heavy
                  right tail; samples clipped to [lo, hi].
    """
    if rng is None:
        rng = np.random.default_rng()

    if kind == "const":
        return np.full(N, mean, dtype=np.float64)

    if kind == "normal":
        if std < 0:
            raise ValueError("std must be >= 0")
        S = rng.normal(mean, std, size=N)
        return np.clip(S, lo, hi).astype(np.float64)

    if kind == "powerlaw":
        if not (lo < mean < hi):
            raise ValueError(f"powerlaw mean must lie strictly in ({lo}, {hi})")
        # Bounded Pareto on [lo, hi] with shape alpha, sampled by inverse
        # CDF. We pick alpha numerically so the *theoretical* bounded mean
        # equals `mean` — avoids the clip-bias of "unbounded Pareto + clip".
        alpha = _bounded_pareto_alpha_for_mean(mean, lo, hi)
        u = rng.uniform(0.0, 1.0, size=N)
        # F^-1(u) = lo / (1 - u * (1 - (lo/hi)^alpha))^(1/alpha)
        S = lo / (1.0 - u * (1.0 - (lo / hi) ** alpha)) ** (1.0 / alpha)
        return S.astype(np.float64)

    raise ValueError(f"unknown kind: {kind!r}")


def _bounded_pareto_mean(alpha: float, lo: float, hi: float) -> float:
    """Theoretical mean of a bounded Pareto on [lo, hi] with shape alpha."""
    if abs(alpha - 1.0) < 1e-10:
        return lo * hi * np.log(hi / lo) / (hi - lo)
    r = lo / hi
    num = lo - hi * r ** alpha
    den = 1.0 - r ** alpha
    return (alpha / (alpha - 1.0)) * num / den


def _bounded_pareto_alpha_for_mean(target: float, lo: float, hi: float) -> float:
    """Find shape alpha such that bounded Pareto on [lo, hi] has mean `target`.

    The mean is monotonically decreasing in alpha: alpha → 0+ pushes mass
    to the upper bound (mean → hi), alpha → ∞ concentrates near lo (mean → lo).
    """
    return brentq(
        lambda a: _bounded_pareto_mean(a, lo, hi) - target,
        1e-4, 1.0e3, xtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Single-run driver (with optional P(t) trajectory)
# ---------------------------------------------------------------------------

def run_simulation_hetS(
    G,
    S_arr: np.ndarray,
    a: float,
    w0: float,
    pD: float,
    T: int,
    init_seed: int,
    sim_seed: int,
    record_traj: bool = True,
):
    """Run one simulation on graph G with per-node stubbornness S_arr.

    Returns a dict with keys:
        P_t       : (T+1,) array of P at t=0..T   (only if record_traj)
        P_final   : scalar P(T)
        state     : (N,) int8 final opinions
        W         : (N, N) final weight matrix
        A         : (N, N) bool adjacency (for downstream slicing)
    """
    state, A, W = initialise_fast(G, w0=w0, pD=pD, seed=init_seed)
    rng = np.random.default_rng(sim_seed)

    if record_traj:
        P_t = np.empty(T + 1, dtype=np.float64)
        P_t[0] = polarization_coefficient_fast(A, W, state)
    else:
        P_t = None

    for t in range(1, T + 1):
        state = step_opinions_hetS(A, W, state, S_arr, rng)
        W = step_weights_fast(A, W, state, a)
        if record_traj:
            P_t[t] = polarization_coefficient_fast(A, W, state)

    P_final = polarization_coefficient_fast(A, W, state)

    return {
        "P_t": P_t,
        "P_final": P_final,
        "state": state,
        "W": W,
        "A": A,
    }


# ---------------------------------------------------------------------------
# Equivalence test: with S_arr = S * ones(N) the new code must reproduce
# polarized_fast.step_opinions_fast bit-for-bit (same rng order).
# ---------------------------------------------------------------------------

def test_equivalence_hetS(n=80, k=8, T=200, S=10.0, a=0.05,
                          init_seed=1, sim_seed=2, graph_seed=42,
                          verbose=True):
    """Run scalar-S and constant-array-S dynamics with identical RNG and
    confirm bit-identical state + weight trajectories."""
    G = make_graph("er", n, k, seed=graph_seed)

    # --- scalar-S baseline ---
    state_a, A_a, W_a = initialise_fast(G, w0=0.5, pD=0.5, seed=init_seed)
    rng_a = np.random.default_rng(sim_seed)
    for _ in range(T):
        state_a = step_opinions_fast(A_a, W_a, state_a, S, rng_a)
        W_a = step_weights_fast(A_a, W_a, state_a, a)

    # --- heterogeneous-S with constant array ---
    state_b, A_b, W_b = initialise_fast(G, w0=0.5, pD=0.5, seed=init_seed)
    rng_b = np.random.default_rng(sim_seed)
    S_arr = np.full(n, S, dtype=np.float64)
    for _ in range(T):
        state_b = step_opinions_hetS(A_b, W_b, state_b, S_arr, rng_b)
        W_b = step_weights_fast(A_b, W_b, state_b, a)

    state_eq = bool(np.array_equal(state_a, state_b))
    w_max_diff = float(np.abs(W_a - W_b).max())
    passed = state_eq and (w_max_diff == 0.0)

    if verbose:
        print(f"[equiv]  N={n} k={k} T={T} S={S} a={a}")
        print(f"         states bit-equal     : {state_eq}")
        print(f"         max |W_fast - W_hetS| : {w_max_diff:.2e}")
        print(f"         result               : {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("Heterogeneous-S vs scalar-S equivalence tests")
    print("=" * 60)
    cases = [
        dict(n=50,  k=8, T=50,  S=5.0,  a=0.05, init_seed=1, sim_seed=10, graph_seed=42),
        dict(n=80,  k=8, T=100, S=20.0, a=0.10, init_seed=2, sim_seed=11, graph_seed=43),
        dict(n=100, k=8, T=200, S=2.0,  a=0.30, init_seed=3, sim_seed=12, graph_seed=44),
        dict(n=100, k=8, T=200, S=50.0, a=0.01, init_seed=4, sim_seed=13, graph_seed=45),
        dict(n=150, k=8, T=300, S=10.0, a=0.05, init_seed=5, sim_seed=14, graph_seed=46),
    ]
    all_pass = True
    for c in cases:
        all_pass = test_equivalence_hetS(**c) and all_pass
    print()
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Quick sanity demo of the three distributions
    print()
    print("=" * 60)
    print("S_i distribution sanity check (N=10000, mean=10)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    for kind in ("const", "normal", "powerlaw"):
        kwargs = dict(N=10000, kind=kind, mean=10.0, rng=rng)
        if kind == "normal":
            kwargs["std"] = 3.0
        S = make_S_distribution(**kwargs)
        print(f"  {kind:9s}  empirical mean={S.mean():6.3f}  std={S.std():6.3f}  "
              f"min={S.min():.2f}  max={S.max():.2f}")
