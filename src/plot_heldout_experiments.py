"""Create neutral, publication-style summaries for Experiments 1--3.

The script uses only committed aggregate result files.  It deliberately avoids
claim-like titles and uses a colour-blind-friendly palette consistently across
the three figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

NAVY = "#19324D"
BLUE = "#4C78A8"
TEAL = "#159D8C"
ORANGE = "#F28E2B"
PURPLE = "#756BB1"
GREY = "#9AA5B1"
GRID = "#DDE3EA"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5D6772",
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#172B4D",
        }
    )


def _label_bars(ax: plt.Axes, bars, digits: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
        )


def plot_experiment_1(summary: dict) -> None:
    exp = summary["experiments"][0]
    labels = ["CD25\nT cells", "CD69\nT cells", "HLA-DR\nmonocytes"]
    series = [
        ("totalVI decoder", exp["native_totalvi_spearman"], GREY),
        ("FusionVI decoder", exp["native_fusionvi_spearman"], NAVY),
        ("totalVI-X", exp["totalvi_xmodal_spearman"], BLUE),
        ("FusionVI-X", exp["fusionvi_xmodal_spearman"], TEAL),
    ]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for idx, (name, values, color) in enumerate(series):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        _label_bars(ax, bars)
    ax.set_title("Experiment 1 · Donor-held-out hidden-marker recovery", loc="left", pad=16)
    ax.text(
        0,
        1.02,
        "Lawlor PBMC CITE-seq · 10 leave-one-donor-out folds",
        transform=ax.transAxes,
        fontsize=10,
        color="#5D6772",
    )
    ax.set_ylabel("Spearman correlation")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "lawlor_marker_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_2(summary: dict) -> None:
    exp = summary["experiments"][1]
    metrics = exp["metrics"]
    methods = [
        ("CD274 RNA", "cd274_rna_only", GREY),
        ("totalVI decoder", "totalvi_decoder", BLUE),
        ("FusionVI decoder", "fusionvi_decoder", NAVY),
        ("totalVI-X", "totalvi_xmodal", PURPLE),
        ("FusionVI-X", "fusionvi_xmodal", TEAL),
    ]
    labels = [m[0] for m in methods]
    colors = [m[2] for m in methods]
    spearman = [metrics[m[1]]["effect_spearman"] for m in methods]
    direction = [metrics[m[1]]["direction_accuracy"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), sharey=True)
    for ax, values, title in zip(
        axes,
        [spearman, direction],
        ["A  Effect ranking", "B  Effect direction"],
    ):
        bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.72)
        _label_bars(ax, bars)
        ax.set_title(title, loc="left", fontsize=12, pad=10)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    fig.suptitle(
        "Experiment 2 · PD-L1 prediction for unseen CRISPR targets",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#172B4D",
    )
    fig.text(
        0.06,
        0.91,
        "Papalexi ECCITE-seq · RNA and three other proteins observed; PD-L1 hidden",
        fontsize=10,
        color="#5D6772",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGURES / "papalexi_pdl1_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_3() -> None:
    df = pd.read_csv(RESULTS / "cross_mouse_baselines.csv")
    order = [
        "rna_proxy",
        "protein_context",
        "protein_context_plus_rna",
        "totalvi_xmodal",
        "fusionvi_xmodal",
    ]
    labels = [
        "RNA proxy",
        "Protein\ncontext",
        "Protein context\n+ matching RNA",
        "totalVI-X",
        "FusionVI-X",
    ]
    colors = [GREY, ORANGE, BLUE, PURPLE, TEAL]
    means = df.groupby("readout")["spearman"].mean().reindex(order)
    fold_means = (
        df.groupby(["heldout_mouse", "readout"])["spearman"]
        .mean()
        .unstack("readout")
        .reindex(columns=order)
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(order))
    bars = ax.bar(x, means.to_numpy(), color=colors, width=0.67)
    _label_bars(ax, bars)
    for _, row in fold_means.iterrows():
        ax.plot(x, row.to_numpy(), color=NAVY, alpha=0.35, linewidth=1.2, marker="o", markersize=4)
    ax.set_title("Experiment 3 · Hidden-marker recovery across mice", loc="left", pad=16)
    ax.text(
        0,
        1.02,
        "SLN111 mouse CITE-seq · mean across four jointly masked markers; lines show held-out mice",
        transform=ax.transAxes,
        fontsize=10,
        color="#5D6772",
    )
    ax.set_ylabel("Mean Spearman correlation")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "totalvi_original_marker_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "fusionvi_experiments_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    plot_experiment_1(summary)
    plot_experiment_2(summary)
    plot_experiment_3()
    print("Wrote Experiment 1--3 figures to", FIGURES)


if __name__ == "__main__":
    main()
