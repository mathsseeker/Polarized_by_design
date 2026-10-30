#!/usr/bin/env python3
"""
Reanalysis of universality_sweep.csv using P(T) = 1 - w̄_×(T) as the
primary polarization observable, with C as the auxiliary diagnostic.

Steps 2–5. No new simulations — pure post-processing.
Output:
  universality_collapse_P.png   (headline collapse figure)
  regime_map_P_C.png            (regime map on (a, S) plane)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

CSV       = "universality_sweep.csv"
W0        = 0.5
KAPPA_STAR = W0**2 / 2        # = 0.125  (eq. heuristic threshold)

GRAPH_TYPES  = ("er", "ws", "ba")
MEAN_DEGREES = (4, 8, 16)
COLOUR = {"er": "C0", "ws": "C1", "ba": "C2"}
LABEL  = {"er": "ER",  "ws": "WS",  "ba": "BA"}

# Regime codes
MIXED      = 0
TRANSITION = 1
POLARIZED  = 2
CONSENSUS  = 3
REGIME_NAME = {MIXED: "Mixed", TRANSITION: "Transition",
               POLARIZED: "Polarized", CONSENSUS: "Consensus"}
REGIME_CLR  = ["#2166ac", "#fdae61", "#d73027", "#1a9850"]


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV)
missing = [c for c in ("P_mean", "P_std", "C_mean", "C_std") if c not in df.columns]
if missing:
    sys.exit(f"STOP: columns missing from CSV: {missing}. Re-run the sweep first.")

print(f"[load]  {CSV}  —  {len(df)} rows, columns: {df.columns.tolist()}")
print(f"        P_mean range: [{df.P_mean.min():.3f}, {df.P_mean.max():.3f}]")
print(f"        C_mean range: [{df.C_mean.min():.3f}, {df.C_mean.max():.3f}]")


# ---------------------------------------------------------------------------
# Helper: bin by log(κ)
# ---------------------------------------------------------------------------
def bin_by_log_kappa(kappa, values, n_bins=20):
    log_edges = np.linspace(np.log10(kappa.min()), np.log10(kappa.max()), n_bins + 1)
    idx = np.clip(np.digitize(np.log10(kappa), log_edges) - 1, 0, n_bins - 1)
    c, m, s = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 2:
            continue
        c.append(10 ** (0.5 * (log_edges[b] + log_edges[b + 1])))
        m.append(values[sel].mean())
        s.append(values[sel].std(ddof=1))
    return np.array(c), np.array(m), np.array(s)


# ---------------------------------------------------------------------------
# STEP 2 — Headline collapse plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
row_specs = [
    ("P_mean", r"$P(T) = 1 - \bar{w}_\times(T)$",  "polarization (primary)"),
    ("C_mean", r"$C = |2f_D - 1|$",                  "consensus (auxiliary)"),
]

for row_i, (col, ylabel, sub_title) in enumerate(row_specs):
    for col_i, k in enumerate(MEAN_DEGREES):
        ax = axes[row_i, col_i]
        for g in GRAPH_TYPES:
            sub = df[(df.graph_type == g) & (df.mean_degree == k)]
            cx, cm, cs = bin_by_log_kappa(sub.kappa.values, sub[col].values)
            ax.plot(cx, cm, "-", color=COLOUR[g], lw=2, label=LABEL[g])
            ax.fill_between(cx, cm - cs, cm + cs, color=COLOUR[g], alpha=0.20)
        ax.axvline(KAPPA_STAR, ls="--", color="red", lw=1.2,
                   label=r"$\kappa^*=w_0^2/2=0.125$")
        ax.axhline(0.5, ls=":", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        if row_i == 0:
            ax.set_title(fr"$\langle k\rangle = {k}$ — {sub_title}", fontsize=10)
        else:
            ax.set_title(sub_title, fontsize=10)
        if row_i == 1:
            ax.set_xlabel(r"$\kappa = a \cdot S$", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.25, lw=0.4)

fig.suptitle(
    r"Universality collapse:  $P(T)$ (primary) and $C$ (auxiliary)  vs  $\kappa = a \cdot S$",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig("universality_collapse_P.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[save]  universality_collapse_P.png")


# ---------------------------------------------------------------------------
# STEP 3 — Regime classification + map
# ---------------------------------------------------------------------------
def classify(P, C):
    if P < 0.5:
        return MIXED
    elif P >= 0.7 and C < 0.3:
        return POLARIZED
    elif P >= 0.7 and C >= 0.7:
        return CONSENSUS
    else:
        return TRANSITION

df["regime"] = df.apply(lambda r: classify(r.P_mean, r.C_mean), axis=1)

a_vals = np.sort(df.a.unique())
S_vals = np.sort(df.S.unique())

cmap4 = ListedColormap(REGIME_CLR)
norm4 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap4.N)

fig2, axes2 = plt.subplots(3, 3, figsize=(13, 12), sharex=True, sharey=True)

for row_i, g in enumerate(GRAPH_TYPES):
    for col_i, k in enumerate(MEAN_DEGREES):
        ax = axes2[row_i, col_i]
        sub = df[(df.graph_type == g) & (df.mean_degree == k)].copy()

        grid = np.full((len(S_vals), len(a_vals)), np.nan)
        for _, row in sub.iterrows():
            ia = int(np.argmin(np.abs(a_vals - row.a)))
            iS = int(np.argmin(np.abs(S_vals - row.S)))
            grid[iS, ia] = row.regime

        ax.pcolormesh(a_vals, S_vals, grid,
                      cmap=cmap4, norm=norm4, shading="nearest")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{LABEL[g]}, ⟨k⟩={k}", fontsize=10)
        if row_i == 2:
            ax.set_xlabel(r"$a$", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(r"$S$", fontsize=10)

legend_patches = [Patch(color=REGIME_CLR[r], label=REGIME_NAME[r])
                  for r in [MIXED, TRANSITION, POLARIZED, CONSENSUS]]
fig2.legend(handles=legend_patches, loc="lower center", ncol=4,
            fontsize=11, bbox_to_anchor=(0.5, -0.03))
fig2.suptitle(
    r"Regime map on $(a, S)$ plane — classified by $(P, C)$   "
    r"[blue=Mixed, orange=Transition, red=Polarized, green=Consensus]",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("regime_map_P_C.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("[save]  regime_map_P_C.png")


# ---------------------------------------------------------------------------
# STEP 4 — Quantitative summary
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("  REGIME FRACTIONS  (one row per graph × ⟨k⟩)")
print("=" * 76)
print(f"  {'graph':<6} {'⟨k⟩':<5} {'Mixed':>10} {'Polarized':>12} "
      f"{'Consensus':>12} {'Transition':>12}")
print("  " + "-" * 55)
for g in GRAPH_TYPES:
    for k in MEAN_DEGREES:
        sub = df[(df.graph_type == g) & (df.mean_degree == k)]
        n = len(sub)
        fmix = (sub.regime == MIXED).sum()      / n
        fpol = (sub.regime == POLARIZED).sum()  / n
        fcon = (sub.regime == CONSENSUS).sum()  / n
        ftra = (sub.regime == TRANSITION).sum() / n
        print(f"  {LABEL[g]:<6} {k:<5} {fmix:>10.3f} {fpol:>12.3f} "
              f"{fcon:>12.3f} {ftra:>12.3f}")

# κ_½_P: where P_mean crosses 0.5 in the binned curve
kappa_half_vals = []
for g in GRAPH_TYPES:
    for k in MEAN_DEGREES:
        sub = df[(df.graph_type == g) & (df.mean_degree == k)]
        cx, cm, _ = bin_by_log_kappa(sub.kappa.values, sub.P_mean.values)
        for i in range(1, len(cm)):
            if (cm[i - 1] - 0.5) * (cm[i] - 0.5) <= 0:
                x0, x1 = np.log10(cx[i - 1]), np.log10(cx[i])
                y0, y1 = cm[i - 1], cm[i]
                if y1 != y0:
                    xc = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
                    kappa_half_vals.append(10 ** xc)
                break

print()
if kappa_half_vals:
    kh_med = float(np.median(kappa_half_vals))
    ratio  = kh_med / KAPPA_STAR
    print(f"  κ_½_P  (median across {len(kappa_half_vals)} configurations): {kh_med:.4f}")
    print(f"  Heuristic κ* = {KAPPA_STAR:.4f}")
    print(f"  Ratio  κ_½_P / κ* = {ratio:.2f}  →  "
          + ("within factor 2–3 ✓" if 0.33 <= ratio <= 3.0 else "outside factor 2–3 ✗"))
    # Transition width
    widths = []
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            sub = df[(df.graph_type == g) & (df.mean_degree == k)]
            cx, cm, _ = bin_by_log_kappa(sub.kappa.values, sub.P_mean.values)
            trans = cx[(cm > 0.3) & (cm < 0.7)]
            if len(trans) >= 2:
                widths.append(trans.max() / trans.min())
    if widths:
        print(f"  Transition width  (κ_high/κ_low where 0.3<P<0.7): "
              f"median ratio = {np.median(widths):.2f}")
else:
    print("  κ_½_P: P_mean does not cross 0.5 within the sweep range.")
    print(f"         Lowest binned P_mean is already above 0.5 — κ_½_P < κ_min of sweep.")
    print(f"  Heuristic κ* = {KAPPA_STAR:.4f}")
    # Still report transition width if available
    widths = []
    for g in GRAPH_TYPES:
        for k in MEAN_DEGREES:
            sub = df[(df.graph_type == g) & (df.mean_degree == k)]
            cx, cm, _ = bin_by_log_kappa(sub.kappa.values, sub.P_mean.values)
            trans = cx[(cm > 0.3) & (cm < 0.7)]
            if len(trans) >= 2:
                widths.append(trans.max() / trans.min())
    if widths:
        print(f"  Transition width  (κ range where 0.3<P<0.7): "
              f"median κ_high/κ_low = {np.median(widths):.2f}")
    else:
        print("  Transition width: no bins in (0.3, 0.7) — P stays above 0.7 throughout.")

# Within-bin collapse quality for P(T)
print()
print("  Within-subplot collapse quality for P(T)  (spread across graph types):")
print(f"  {'⟨k⟩':<6} {'median bin-std':>18} {'worst bin-std':>18} {'verdict':>14}")
all_med = []
for k in MEAN_DEGREES:
    ref_sub = df[(df.graph_type == "ws") & (df.mean_degree == k)]
    ref_x, _, _ = bin_by_log_kappa(ref_sub.kappa.values, ref_sub.P_mean.values)
    per_bin = []
    for xt in ref_x:
        pts = []
        for g in GRAPH_TYPES:
            sub = df[(df.graph_type == g) & (df.mean_degree == k)]
            cx, cm, _ = bin_by_log_kappa(sub.kappa.values, sub.P_mean.values)
            if cx.size and cx.min() <= xt <= cx.max():
                pts.append(float(np.interp(np.log10(xt), np.log10(cx), cm)))
        if len(pts) >= 2:
            per_bin.append(float(np.std(pts, ddof=1)))
    if per_bin:
        med = float(np.median(per_bin))
        wrs = float(np.max(per_bin))
        all_med.append(med)
        v = "TIGHT" if med < 0.05 else ("MODERATE" if med < 0.12 else "POOR")
        print(f"  {k:<6} {med:>18.3f} {wrs:>18.3f} {v:>14}")


# ---------------------------------------------------------------------------
# STEP 5 — Honest verdict
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("  HONEST VERDICT")
print("=" * 76)

if all_med:
    omed = float(np.median(all_med))
    collapse_q = "Yes" if omed < 0.05 else ("Moderate" if omed < 0.12 else "No")
    print(f"  Does P(T) collapse on κ?          {collapse_q}  "
          f"(median within-bin std = {omed:.3f})")
    univ = ("Yes — topology and ⟨k⟩ do not split the P(T) vs κ curve."
            if omed < 0.05 else
            "Moderate — small topology-dependent offsets remain."
            if omed < 0.12 else
            "No — curves separate substantially across graph types.")
    print(f"  Universal across 9 configurations? {univ}")

print()
print("  Dominant regime per (graph, ⟨k⟩):")
print(f"  {'graph':<6} {'⟨k⟩':<5} {'dominant regime':<18} {'fraction':>10}")
for g in GRAPH_TYPES:
    for k in MEAN_DEGREES:
        sub = df[(df.graph_type == g) & (df.mean_degree == k)]
        counts = sub.regime.value_counts()
        dom  = int(counts.idxmax())
        frac = counts.max() / len(sub)
        print(f"  {LABEL[g]:<6} {k:<5} {REGIME_NAME[dom]:<18} {frac:>10.2f}")

print("=" * 76)
