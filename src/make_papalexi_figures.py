"""Publication-style figures for the Papalexi external validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiment2_papalexi"
FIGURES = RESULTS / "figures"
COLORS = {
    "rna_proxy_only": "#9CA3AF",
    "totalvi": "#2563EB",
    "totalvi_xmodal": "#06B6D4",
    "fusionvi": "#7C3AED",
    "fusionvi_xmodal": "#EC4899",
}
LABELS = {
    "rna_proxy_only": "CD274 RNA",
    "totalvi": "totalVI decoder",
    "totalvi_xmodal": "totalVI-X",
    "fusionvi": "FusionVI decoder",
    "fusionvi_xmodal": "FusionVI-X",
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    metrics = pd.read_csv(RESULTS / "summary_metrics.csv")
    biological = pd.read_csv(RESULTS / "biological_effects.csv")

    models = ["rna_proxy_only", "totalvi", "totalvi_xmodal", "fusionvi", "fusionvi_xmodal"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ["effect_spearman", "direction_accuracy"],
        ["Rank unseen perturbation effects", "Recover effect direction"],
    ):
        values = metrics[metrics["metric"] == metric].set_index("model").loc[models, "value"]
        axis.bar(
            [LABELS[name] for name in models],
            values,
            color=[COLORS[name] for name in models],
            edgecolor="white",
            linewidth=1.5,
        )
        axis.set_ylim(0, 1)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("Spearman ρ" if metric == "effect_spearman" else "Direction accuracy")
        axis.tick_params(axis="x", rotation=25)
        for i, value in enumerate(values):
            axis.text(i, value + 0.025, f"{value:.2f}", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Experiment 2 · PD-L1 prediction for unseen CRISPR targets", fontsize=19, fontweight="bold")
    fig.savefig(FIGURES / "papalexi_external_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "papalexi_external_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("cd274_rna_effect", "CD274 RNA", "#9CA3AF"),
        ("totalvi_x_predicted_effect", "totalVI-X", "#06B6D4"),
        ("fusionvi_x_predicted_effect", "FusionVI-X", "#EC4899"),
    ]
    lower = min(biological["pdl1_protein_effect"].min(), *(biological[c].min() for c, _, _ in panels)) - 0.08
    upper = max(biological["pdl1_protein_effect"].max(), *(biological[c].max() for c, _, _ in panels)) + 0.08
    highlights = {"JAK2", "IFNGR1", "IFNGR2", "STAT1", "CUL3", "BRD4", "CMTM6", "SPI1"}
    for axis, (column, title, color) in zip(axes, panels):
        axis.axhline(0, color="#CBD5E1", linewidth=1)
        axis.axvline(0, color="#CBD5E1", linewidth=1)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#64748B", linewidth=1.2)
        axis.scatter(
            biological["pdl1_protein_effect"],
            biological[column],
            s=60,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        for row in biological.itertuples(index=False):
            if row.gene in highlights:
                axis.annotate(row.gene, (row.pdl1_protein_effect, getattr(row, column)), xytext=(4, 4), textcoords="offset points", fontsize=8)
        rho = biological[["pdl1_protein_effect", column]].corr(method="spearman").iloc[0, 1]
        axis.set_title(f"{title}\nρ = {rho:.2f}", fontweight="bold")
        axis.set_xlabel("Observed surface PD-L1 effect")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    axes[0].set_ylabel("Predicted or RNA effect")
    fig.suptitle("Cross-modal context recovers pharmacologically relevant PD-L1 regulation", fontsize=18, fontweight="bold")
    fig.savefig(FIGURES / "papalexi_gene_effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "papalexi_gene_effects.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
