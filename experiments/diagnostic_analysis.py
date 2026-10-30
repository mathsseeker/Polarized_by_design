"""Diagnostic analysis on collapse_test.csv.

Tests whether within-bin variance of each observable at fixed kappa = a*S is
stochastic noise (more runs would tighten it) or systematic structure (a real
two-parameter dependence beyond kappa).

For each of P_mean, f_E_cross_mean, consensus_mean, w_x_density_mean:
  - bin cells by log10(kappa) into ~10 equal-width bins (skip bins with <4 cells)
  - in each bin, regress observable vs log10(a) and vs log10(a/S)
  - compare within-bin std of cell means to median per-cell run-to-run std / sqrt(n_runs)
  - summarize beta_1 vs kappa across bins

Outputs:
  ./diagnostic_plots/*.png
  ./diagnostic_summary.md
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collapse_test.csv")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_plots")
SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_summary.md")

N_BINS = 10
MIN_CELLS_PER_BIN = 4
ALPHA = 0.05

# observables: (mean_col, std_col, pretty_label)
OBSERVABLES = [
    ("P_mean", "P_std", "P(T)"),
    ("f_E_cross_mean", "f_E_cross_std", "f_E_cross"),
    ("consensus_mean", "consensus_std", "consensus"),
    ("w_x_density_mean", "w_x_density_std", "w_x_density"),
]


def load_and_inspect(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {path}")
    print(f"[load] shape = {df.shape}")
    print(f"[load] columns = {list(df.columns)}")
    print("[load] head:")
    print(df.head().to_string())
    # confirm one row per (a, S)
    n_unique = df.groupby(["a", "S"]).ngroups
    print(f"[load] unique (a,S) cells = {n_unique}; total rows = {len(df)}")
    if n_unique == len(df):
        print("[load] CSV already aggregated: one row per (a,S) cell with mean/std fields.")
    else:
        print("[load] WARNING: multiple rows per (a,S). Aggregating to per-cell mean/std.")
        agg = {}
        for mean_col, std_col, _ in OBSERVABLES:
            base = mean_col.replace("_mean", "")
            if base in df.columns:
                agg[mean_col] = (base, "mean")
                agg[std_col] = (base, "std")
        df = df.groupby(["a", "S"]).agg(**agg).reset_index()
        df["kappa"] = df["a"] * df["S"]
        df["n_runs"] = df.groupby(["a", "S"]).size().reset_index(name="n")["n"]
    return df


def assign_kappa_bins(df: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, np.ndarray]:
    log_k = np.log10(df["kappa"].to_numpy())
    edges = np.linspace(log_k.min(), log_k.max() + 1e-12, n_bins + 1)
    df = df.copy()
    df["log10_kappa"] = log_k
    df["log10_a"] = np.log10(df["a"].to_numpy())
    df["log10_aoverS"] = np.log10(df["a"].to_numpy() / df["S"].to_numpy())
    df["kbin"] = np.digitize(log_k, edges[1:-1])  # 0..n_bins-1
    return df, edges


def regress_bin(
    bin_df: pd.DataFrame, x_col: str, y_col: str
) -> dict:
    x = bin_df[x_col].to_numpy()
    y = bin_df[y_col].to_numpy()
    if len(x) < 2 or np.unique(x).size < 2:
        return {"slope": np.nan, "stderr": np.nan, "pvalue": np.nan, "intercept": np.nan, "n": len(x)}
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "stderr": float(res.stderr),
        "pvalue": float(res.pvalue),
        "intercept": float(res.intercept),
        "n": int(len(x)),
    }


def per_bin_results(
    df: pd.DataFrame,
    edges: np.ndarray,
    mean_col: str,
    std_col: str,
    x_col: str,
) -> list[dict]:
    results = []
    n_bins = len(edges) - 1
    for b in range(n_bins):
        bin_df = df[df["kbin"] == b]
        n_cells = len(bin_df)
        if n_cells < MIN_CELLS_PER_BIN:
            results.append(
                {
                    "bin": b,
                    "n_cells": n_cells,
                    "log_kappa_lo": float(edges[b]),
                    "log_kappa_hi": float(edges[b + 1]),
                    "kappa_center": float(10 ** (0.5 * (edges[b] + edges[b + 1]))),
                    "skipped": True,
                }
            )
            continue
        fit = regress_bin(bin_df, x_col, mean_col)
        n_runs = bin_df["n_runs"].to_numpy() if "n_runs" in bin_df.columns else np.full(n_cells, 8)
        sem_per_cell = bin_df[std_col].to_numpy() / np.sqrt(np.maximum(n_runs, 1))
        results.append(
            {
                "bin": b,
                "n_cells": n_cells,
                "log_kappa_lo": float(edges[b]),
                "log_kappa_hi": float(edges[b + 1]),
                "kappa_center": float(10 ** (0.5 * (edges[b] + edges[b + 1]))),
                "skipped": False,
                "slope": fit["slope"],
                "stderr": fit["stderr"],
                "pvalue": fit["pvalue"],
                "intercept": fit["intercept"],
                "within_bin_std_of_means": float(np.std(bin_df[mean_col].to_numpy(), ddof=1)),
                "median_run_std": float(np.median(bin_df[std_col].to_numpy())),
                "median_sem": float(np.median(sem_per_cell)),
                "cells": bin_df,
            }
        )
    return results


def plot_grid(
    results: list[dict],
    mean_col: str,
    std_col: str,
    x_col: str,
    pretty_label: str,
    out_path: str,
) -> None:
    active = [r for r in results if not r["skipped"]]
    if not active:
        return
    n = len(active)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, active):
        bin_df = r["cells"]
        n_runs = bin_df["n_runs"].to_numpy() if "n_runs" in bin_df.columns else np.full(len(bin_df), 8)
        sem = bin_df[std_col].to_numpy() / np.sqrt(np.maximum(n_runs, 1))
        x = bin_df[x_col].to_numpy()
        y = bin_df[mean_col].to_numpy()
        ax.errorbar(x, y, yerr=sem, fmt="o", ms=4, capsize=2, alpha=0.85)
        # fit line
        if np.isfinite(r["slope"]):
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, r["intercept"] + r["slope"] * xs, "-", color="C3", lw=1.2)
        sig = "*" if (np.isfinite(r["pvalue"]) and r["pvalue"] < ALPHA) else ""
        ax.set_title(
            f"k_c={r['kappa_center']:.3g}  n={r['n_cells']}\n"
            f"beta1={r['slope']:.3g}+/-{r['stderr']:.2g}  p={r['pvalue']:.2g}{sig}",
            fontsize=8,
        )
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(pretty_label, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(active):]:
        ax.axis("off")
    fig.suptitle(f"{pretty_label} vs {x_col}, binned by log10(kappa) — error bars = SEM (std/sqrt(n_runs))", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_beta_vs_kappa(
    results: list[dict],
    pretty_label: str,
    x_col: str,
    out_path: str,
) -> None:
    active = [r for r in results if not r["skipped"]]
    if not active:
        return
    centers = np.array([r["kappa_center"] for r in active])
    slopes = np.array([r["slope"] for r in active])
    errs = np.array([r["stderr"] for r in active])
    pvals = np.array([r["pvalue"] for r in active])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["C3" if p < ALPHA else "C0" for p in pvals]
    for c, s, e, col in zip(centers, slopes, errs, colors):
        ax.errorbar(c, s, yerr=e, fmt="o", color=col, capsize=3)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("bin-center kappa (log scale)")
    ax.set_ylabel(f"beta_1  (slope of {pretty_label} vs {x_col})")
    ax.set_title(f"{pretty_label}: beta_1 vs kappa (red = p<{ALPHA})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def classify(results: list[dict], pretty_label: str) -> tuple[str, dict]:
    """Return (case_letter, stats_dict) for an observable."""
    active = [r for r in results if not r["skipped"]]
    if not active:
        return "?", {}
    slopes = np.array([r["slope"] for r in active])
    pvals = np.array([r["pvalue"] for r in active])
    sigs = pvals < ALPHA
    n_sig = int(sigs.sum())
    frac_sig = float(n_sig) / len(active)
    n_pos = int((slopes > 0).sum())
    n_neg = int((slopes < 0).sum())
    sign_consistent = (n_pos >= 0.75 * len(active)) or (n_neg >= 0.75 * len(active))

    # noise floor comparison: within-bin std of cell means vs median SEM
    excess_ratios = np.array(
        [r["within_bin_std_of_means"] / r["median_sem"] if r["median_sem"] > 0 else np.nan
         for r in active]
    )
    median_excess = float(np.nanmedian(excess_ratios))

    median_abs_slope = float(np.median(np.abs(slopes)))

    # heuristic classification
    if frac_sig <= 0.1 and median_excess < 1.5:
        case = "A"  # noise-like
    elif frac_sig >= 0.5 and sign_consistent and median_excess > 2.0:
        case = "B"  # structure
    elif n_sig >= 2 and not (frac_sig >= 0.5 and sign_consistent):
        case = "C"  # mixed
    elif median_excess > 1.5:
        case = "C"
    else:
        case = "A"

    stats_dict = {
        "n_bins_active": len(active),
        "n_sig": n_sig,
        "frac_sig": frac_sig,
        "median_abs_slope": median_abs_slope,
        "median_excess_ratio": median_excess,
        "sign_consistent": sign_consistent,
        "n_positive_slopes": n_pos,
        "n_negative_slopes": n_neg,
    }
    return case, stats_dict


def write_summary(
    bin_counts: list[tuple[int, float, float, int]],
    results_logA: dict[str, list[dict]],
    results_logAoverS: dict[str, list[dict]],
    cases: dict[str, dict],
    df_shape: tuple[int, int],
) -> None:
    lines = []
    lines.append("# Diagnostic analysis of collapse_test.csv\n")
    lines.append(f"- input shape: {df_shape[0]} rows x {df_shape[1]} cols")
    lines.append(f"- bins: {N_BINS} equal-width bins of log10(kappa); skip bins with <{MIN_CELLS_PER_BIN} cells")
    lines.append(f"- significance threshold: p < {ALPHA}")
    lines.append("- error bars in per-bin plots: SEM = per-cell std / sqrt(n_runs)\n")

    lines.append("## Bin populations (log10(kappa) range, cell count)\n")
    lines.append("| bin | log10(kappa) range | kappa range | n_cells |")
    lines.append("|---|---|---|---|")
    for b, lo, hi, n in bin_counts:
        lines.append(f"| {b} | [{lo:.3f}, {hi:.3f}] | [{10**lo:.3g}, {10**hi:.3g}] | {n} |")
    lines.append("")

    # Comparison table
    lines.append("## Comparison table across observables\n")
    lines.append(
        "| observable | x-axis | active bins | n_sig | frac_sig | median \\|beta_1\\| | median excess ratio | sign consistent | case |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mean_col, _, label in OBSERVABLES:
        for x_label, table in (("log10(a)", "logA"), ("log10(a/S)", "logAoverS")):
            key = f"{mean_col}__{table}"
            c = cases[key]
            lines.append(
                f"| {label} | {x_label} | {c['n_bins_active']} | {c['n_sig']} | {c['frac_sig']:.2f} "
                f"| {c['median_abs_slope']:.3g} | {c['median_excess_ratio']:.2f} "
                f"| {'yes' if c['sign_consistent'] else 'no'} | {c['case']} |"
            )
    lines.append("")

    # Per-observable per-bin tables
    for mean_col, std_col, label in OBSERVABLES:
        lines.append(f"## {label}\n")
        for x_label, x_col, results in (
            ("log10(a)", "log10_a", results_logA[mean_col]),
            ("log10(a/S)", "log10_aoverS", results_logAoverS[mean_col]),
        ):
            lines.append(f"### vs {x_label}\n")
            lines.append(
                "| bin | kappa_center | n_cells | beta_1 | stderr | p | within-bin std(means) | median run std | median SEM | excess ratio |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in results:
                if r["skipped"]:
                    lines.append(
                        f"| {r['bin']} | {r['kappa_center']:.3g} | {r['n_cells']} | -- | -- | -- | -- | -- | -- | -- |"
                    )
                    continue
                excess = (
                    r["within_bin_std_of_means"] / r["median_sem"] if r["median_sem"] > 0 else float("nan")
                )
                sig = "*" if r["pvalue"] < ALPHA else ""
                lines.append(
                    f"| {r['bin']} | {r['kappa_center']:.3g} | {r['n_cells']} | "
                    f"{r['slope']:.3g}{sig} | {r['stderr']:.2g} | {r['pvalue']:.2g} | "
                    f"{r['within_bin_std_of_means']:.3g} | {r['median_run_std']:.3g} | "
                    f"{r['median_sem']:.3g} | {excess:.2f} |"
                )
            lines.append("")

    # Conclusions per observable
    lines.append("## Case assignment per observable\n")
    for mean_col, _, label in OBSERVABLES:
        cA = cases[f"{mean_col}__logA"]
        cB = cases[f"{mean_col}__logAoverS"]
        lines.append(
            f"- **{label}**: vs log10(a) -> CASE {cA['case']}; vs log10(a/S) -> CASE {cB['case']}. "
            f"(median |beta_1| = {cA['median_abs_slope']:.3g}/{cB['median_abs_slope']:.3g}, "
            f"frac_sig = {cA['frac_sig']:.2f}/{cB['frac_sig']:.2f}, "
            f"median excess ratio = {cA['median_excess_ratio']:.2f}/{cB['median_excess_ratio']:.2f})"
        )
    lines.append("")

    # Plain English conclusion (heuristic, based on cases of P_mean)
    P_logA = cases["P_mean__logA"]
    P_logAoverS = cases["P_mean__logAoverS"]
    fE_logA = cases["f_E_cross_mean__logA"]
    cons_logA = cases["consensus_mean__logA"]
    wxd_logA = cases["w_x_density_mean__logA"]

    lines.append("## Plain-English conclusion\n")
    paragraphs = []
    p1 = (
        f"For P(T), the regression of mean P vs log10(a) at fixed kappa-bin yields case "
        f"{P_logA['case']} (frac_sig={P_logA['frac_sig']:.2f}, median |beta_1|={P_logA['median_abs_slope']:.3g}, "
        f"median excess ratio of within-bin std-of-means to SEM = {P_logA['median_excess_ratio']:.2f}). "
        f"Against log10(a/S) the case is {P_logAoverS['case']} (frac_sig={P_logAoverS['frac_sig']:.2f}, "
        f"median |beta_1|={P_logAoverS['median_abs_slope']:.3g}, "
        f"median excess ratio = {P_logAoverS['median_excess_ratio']:.2f})."
    )
    if P_logA["case"] == "A" and P_logAoverS["case"] == "A":
        verdict = (
            "The residual within-bin variance in P(T) is consistent with stochastic noise from "
            "the 8-run averaging; more runs should tighten the data collapse."
        )
    elif P_logA["case"] == "B" or P_logAoverS["case"] == "B":
        direction = "a/S" if P_logAoverS["case"] == "B" else "a"
        verdict = (
            f"The residual variance in P(T) carries real systematic structure along the {direction} "
            f"direction — kappa alone does not determine P(T); more runs will NOT collapse it further."
        )
    else:
        verdict = (
            "The residual structure in P(T) is partial: significant in some kappa regimes (likely "
            "the transition zone) but not others. Inspect the per-bin table to see where."
        )
    p1 = p1 + " " + verdict
    paragraphs.append(p1)

    p2 = (
        f"For comparison: f_E_cross is case {fE_logA['case']}, consensus is case {cons_logA['case']}, "
        f"w_x_density is case {wxd_logA['case']} (all vs log10(a)). "
        "These three were claimed in the original report to collapse tightly on kappa; "
        "consult the comparison table above to see whether that holds here."
    )
    paragraphs.append(p2)

    for p in paragraphs:
        lines.append(p + "\n")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[write] {SUMMARY_PATH}")


def main() -> int:
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_and_inspect(CSV_PATH)

    if "kappa" not in df.columns:
        df["kappa"] = df["a"] * df["S"]
    if "n_runs" not in df.columns:
        df["n_runs"] = 8

    df, edges = assign_kappa_bins(df, N_BINS)

    print("\n[bins] log10(kappa) edges:", np.round(edges, 4).tolist())
    bin_counts = []
    for b in range(N_BINS):
        n = int((df["kbin"] == b).sum())
        lo, hi = float(edges[b]), float(edges[b + 1])
        marker = "" if n >= MIN_CELLS_PER_BIN else "  (SKIP)"
        print(f"[bins] bin {b}: log10(kappa) in [{lo:.3f}, {hi:.3f}], n_cells={n}{marker}")
        bin_counts.append((b, lo, hi, n))

    results_logA: dict[str, list[dict]] = {}
    results_logAoverS: dict[str, list[dict]] = {}
    cases: dict[str, dict] = {}

    for mean_col, std_col, label in OBSERVABLES:
        print(f"\n[obs] {label} ({mean_col})")
        # vs log10(a)
        rA = per_bin_results(df, edges, mean_col, std_col, "log10_a")
        results_logA[mean_col] = rA
        plot_grid(
            rA, mean_col, std_col, "log10_a", label,
            os.path.join(PLOT_DIR, f"{mean_col}_grid_log10a.png"),
        )
        plot_beta_vs_kappa(
            rA, label, "log10(a)",
            os.path.join(PLOT_DIR, f"{mean_col}_beta_vs_kappa_log10a.png"),
        )
        caseA, statsA = classify(rA, label)
        statsA["case"] = caseA
        cases[f"{mean_col}__logA"] = statsA
        print(f"  vs log10(a):     case={caseA}  n_sig={statsA['n_sig']}/{statsA['n_bins_active']}  "
              f"|beta1|_med={statsA['median_abs_slope']:.3g}  excess={statsA['median_excess_ratio']:.2f}")

        # vs log10(a/S)
        rB = per_bin_results(df, edges, mean_col, std_col, "log10_aoverS")
        results_logAoverS[mean_col] = rB
        plot_grid(
            rB, mean_col, std_col, "log10_aoverS", label,
            os.path.join(PLOT_DIR, f"{mean_col}_grid_log10aoverS.png"),
        )
        plot_beta_vs_kappa(
            rB, label, "log10(a/S)",
            os.path.join(PLOT_DIR, f"{mean_col}_beta_vs_kappa_log10aoverS.png"),
        )
        caseB, statsB = classify(rB, label)
        statsB["case"] = caseB
        cases[f"{mean_col}__logAoverS"] = statsB
        print(f"  vs log10(a/S):   case={caseB}  n_sig={statsB['n_sig']}/{statsB['n_bins_active']}  "
              f"|beta1|_med={statsB['median_abs_slope']:.3g}  excess={statsB['median_excess_ratio']:.2f}")

    write_summary(bin_counts, results_logA, results_logAoverS, cases, df.shape)
    print("\n[done] diagnostic complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
