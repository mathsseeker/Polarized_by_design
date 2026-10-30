#!/usr/bin/env python3
"""
Broom-only at a fixed (a, S) cell — no sweep.

Re-uses run_broom / plot_broom / plot_hist / bimodality_report / write_summary
from bistability_experiment.py (which already imports polarized_fast).

Defaults match the cell shown in the user's previous bistab_broom.png:
    ER, N=200, k=4, a=0.0509, S=16.16, T=10000, 50 runs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from bistability_experiment import (
    run_broom, plot_broom, plot_hist,
    bimodality_report, write_summary,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topo", default="er")
    p.add_argument("--n",    type=int,   default=200)
    p.add_argument("--k",    type=int,   default=4)
    p.add_argument("--pD",   type=float, default=0.5)
    p.add_argument("--w0",   type=float, default=0.5)
    p.add_argument("--T",    type=int,   default=10000)
    p.add_argument("--a",    type=float, default=0.0509)
    p.add_argument("--S",    type=float, default=16.16)
    p.add_argument("--n-demo", type=int, default=50)
    p.add_argument("--graph-seed", type=int, default=42)
    p.add_argument("--state-seed", type=int, default=123)
    p.add_argument("--base-seed",  type=int, default=2024)
    p.add_argument("--out-prefix", default="bistab_er_T10000")
    args = p.parse_args()

    print(f"[setup] {args.topo.upper()}  N={args.n}  k={args.k}  "
          f"a={args.a}  S={args.S}  T={args.T}  n_demo={args.n_demo}")
    t0 = time.time()
    traces = run_broom(args.topo, args.n, args.k, args.pD, args.w0, args.T,
                       args.a, args.S, args.n_demo,
                       args.graph_seed, args.state_seed, args.base_seed)
    elapsed = time.time() - t0
    print(f"[done]  broom in {elapsed:.1f}s  "
          f"({elapsed / args.n_demo:.2f} s/run)")

    plot_broom(traces, args.a, args.S, f"{args.out_prefix}_broom.png")

    P_finals = np.array([tr[-1] for tr in traces])
    plot_hist(P_finals, args.a, args.S, f"{args.out_prefix}_hist.png")

    report = bimodality_report(P_finals, args.a, args.S)
    # sigma_r_max isn't computed here (no sweep) — pass NaN.
    write_summary(report, float("nan"), f"{args.out_prefix}_summary.txt")


if __name__ == "__main__":
    main()
