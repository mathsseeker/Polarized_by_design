import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).parent

NPZ_FILES = [
    "heatmap_k_variable_results_er_k008.npz",
    "heatmap_k_variable_results_ws_k008.npz",
    "heatmap_k_variable_results_ba_k008.npz",
    "heatmap_k_variable_results_ba_k016.npz",
    "heatmap_k_variable_results_ws_k016.npz",
]

GRAPH_LABELS = {"er": "ER", "ws": "WS", "ba": "BA"}


def plot_one(npz_path: Path) -> Path:
    data = np.load(npz_path, allow_pickle=False)
    grid = data["grid"]
    a_vals = data["a_vals"]
    S_vals = data["S_vals"]
    graph_kind = str(data["graph_kind"])
    n = int(data["n"])
    k_avg = int(data["k_avg"])
    pD = float(data["pD"])
    w0 = float(data["w0"])
    T = int(data["T"])
    n_runs = int(data["n_runs"])

    graph_label = GRAPH_LABELS.get(graph_kind.lower(), graph_kind.upper())

    fig, ax = plt.subplots(figsize=(9, 7))

    mesh = ax.pcolormesh(
        a_vals,
        S_vals,
        grid,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )

    ax.set_xlabel(r"$a$ — Hebbian weight step (eq. 5)")
    ax.set_ylabel(r"$S$ — stubbornness  (1 = open, 100 = rigid)")

    ax.set_xlim(a_vals.min(), a_vals.max())
    ax.set_ylim(S_vals.min(), 100.0)

    ax.set_title(
        rf"$P(T) = 1 - \bar{{w}}_\times(T)$    $[\langle k \rangle = {k_avg}]$"
        "\n"
        rf"{graph_label}, $N={n}$, $p_D={pD:.2f}$, $w_0={w0:.2f}$, $T={T}$, runs/cell={n_runs}"
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$P(T)$ :  0 = unpolarised,  1 = fully polarised")

    fig.tight_layout()

    out_path = npz_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    for name in NPZ_FILES:
        npz_path = here / name
        if not npz_path.exists():
            print(f"Skipping (missing): {npz_path}")
            continue
        out = plot_one(npz_path)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
