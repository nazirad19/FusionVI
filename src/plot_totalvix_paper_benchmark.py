"""Plot the totalVI learned-protein-efficiency readout audit."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "results" / "totalvix_paper_seed_summary.csv"
BENCHMARK = ROOT / "results" / "totalvi_efficiency_method_benchmark.csv"
OUTPUT = ROOT / "results" / "figures" / "totalvix_paper_benchmark.png"

HELPER = "totalVI helper-uncorrected"
CORRECT = "totalVI likelihood-consistent"
AFFINE = "D1 affine head"
CORRECT_HEAD = "totalVI likelihood-consistent + D1 head"
COLORS = {HELPER: "#94A3B8", CORRECT: "#059669", AFFINE: "#F97316", CORRECT_HEAD: "#7C3AED"}


def main() -> None:
    frame = pd.read_csv(SEEDS)
    means = frame.groupby("readout").mean(numeric_only=True)
    piv = frame.pivot(index="seed", columns="readout", values="rmsle")
    benchmark = pd.read_csv(BENCHMARK).sort_values("rmsle_rank")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)

    for _, row in piv.iterrows():
        axes[0].plot([0, 1], [row[HELPER], row[CORRECT]], color="#CBD5E1", lw=1)
        axes[0].scatter(0, row[HELPER], color=COLORS[HELPER], s=23, zorder=2)
        axes[0].scatter(1, row[CORRECT], color=COLORS[CORRECT], s=23, zorder=2)
    axes[0].set_xticks([0, 1], ["helper output\nwithout efficiency", "likelihood-consistent\ntotalVI"])
    axes[0].set_ylabel("Mean protein RMSLE")
    axes[0].set_title("A  Readout correction", loc="left", fontweight="bold")

    wanted = [
        "Likelihood-consistent totalVI",
        "D1 affine head",
        "Likelihood-consistent totalVI + D1 head",
        "Official Seurat v3",
        "RNA ridge + D1 calibration",
    ]
    table = benchmark.set_index("method").loc[wanted]
    labels = ["totalVI\ncorrect", "affine X", "correct +\nhead", "Seurat v3", "RNA ridge"]
    colors = ["#059669", "#F97316", "#7C3AED", "#2563EB", "#64748B"]
    bars = axes[1].bar(np.arange(len(table)), table["rmsle"], color=colors, width=0.72)
    axes[1].set_xticks(np.arange(len(table)), labels, rotation=15, ha="right")
    axes[1].set_ylim(0, 0.72)
    axes[1].bar_label(bars, labels=[f"{x:.3f}" for x in table["rmsle"]], padding=3, fontsize=8)
    axes[1].set_title("B  Paper benchmark", loc="left", fontweight="bold")

    order = [HELPER, CORRECT, AFFINE]
    x = np.arange(2)
    width = 0.24
    for i, name in enumerate(order):
        axes[2].bar(x + (i - 1) * width, [means.loc[name, "abs_bias"], means.loc[name, "residual_sd"]],
                    width, label={HELPER: "helper", CORRECT: "correct totalVI", AFFINE: "affine X"}[name],
                    color=COLORS[name])
    axes[2].set_xticks(x, ["Absolute bias", "Residual SD"])
    axes[2].set_title("C  Scale, not biology", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", color="#E2E8F0", lw=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#CBD5E1")
        axis.tick_params(axis="y", length=0, colors="#475569")

    fig.suptitle("A learned efficiency—not a new head—restores totalVI protein imputation",
                 fontsize=14, fontweight="bold", color="#0F172A")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
