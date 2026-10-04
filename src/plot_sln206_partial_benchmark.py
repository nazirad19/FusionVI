"""Plot the frozen SLN206 partial-panel benchmark from versioned CSV outputs."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "calibration_summary_sln206_partial.csv"
OUTPUT = ROOT / "results" / "figures" / "sln206_partial_benchmark.png"


def main() -> None:
    frame = pd.read_csv(INPUT).set_index(["arm", "readout"])
    models = [
        ("FusionVI", "fusionvi", "calibrated", "#2563EB"),
        ("same-width\ntotalVI", "totalvi_w128", "calibrated", "#94A3B8"),
        ("RNA + panel\nridge", "rna_panel_ridge", "ridge", "#F97316"),
        ("RNA-only\nridge", "rna_ridge", "ridge", "#CBD5E1"),
    ]
    metrics = [
        ("rmsle", "Calibrated RMSLE", "lower is better"),
        ("spearman", "Global Spearman", "higher is better"),
        ("within_celltype_spearman", "Within-cell-type Spearman", "higher is better"),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), constrained_layout=True)
    x = np.arange(len(models))

    for axis, (metric, title, direction) in zip(axes, metrics):
        values = [float(frame.loc[(arm, readout), metric]) for _, arm, readout, _ in models]
        colors = [color for _, _, _, color in models]
        bars = axis.bar(x, values, color=colors, width=0.68)
        axis.set_title(title, fontsize=12, fontweight="bold", color="#0F172A", pad=16)
        axis.text(0.5, 1.01, direction, transform=axis.transAxes, ha="center", va="bottom", fontsize=8.5, color="#64748B")
        axis.set_xticks(x, [name for name, *_ in models], fontsize=8.3)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#CBD5E1")
        axis.tick_params(axis="y", length=0, colors="#475569")
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8.2, color="#0F172A")
        low, high = min(values), max(values)
        if metric == "rmsle":
            axis.set_ylim(max(0, low - 0.012), high + 0.012)
        else:
            axis.set_ylim(0, high * 1.22)

    fig.suptitle(
        "Partial-panel transfer: fusion improves ranking, ridge remains strongest",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
