"""Create publication-style figures for the nested cross-modal readout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

LABELS = {
    "CD25_T": "CD25\nT cells",
    "CD69_T": "CD69\nT cells",
    "HLA_DR_Monocyte": "HLA-DR\nmonocytes",
}
PALETTE = {"totalvi": "#5267D9", "fusionvi_xmodal": "#F15B64"}


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(RESULTS / "cv_metrics.csv")
    base = base[
        (base["model"] == "totalvi")
        & (base["analysis"] == "heldout_marker")
        & base["metric"].isin(["spearman_true_pred", "discordant_spearman"])
    ][["model", "donor", "target", "metric", "value"]]
    xmodal = pd.read_csv(RESULTS / "cross_modal_head_metrics.csv")
    xmodal = xmodal[xmodal["model"] == "fusionvi_xmodal"]
    data = pd.concat([base, xmodal], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    for row, metric in enumerate(["spearman_true_pred", "discordant_spearman"]):
        for col, target in enumerate(LABELS):
            ax = axes[row, col]
            sub = data[(data["target"] == target) & (data["metric"] == metric)]
            wide = sub.pivot(index="donor", columns="model", values="value").dropna()
            for _, donor_values in wide.iterrows():
                ax.plot([0, 1], [donor_values["totalvi"], donor_values["fusionvi_xmodal"]], color="#AEB8CA", alpha=0.55, lw=1)
                ax.scatter(0, donor_values["totalvi"], color=PALETTE["totalvi"], alpha=0.7, s=24, zorder=2)
                ax.scatter(1, donor_values["fusionvi_xmodal"], color=PALETTE["fusionvi_xmodal"], alpha=0.7, s=24, zorder=2)
            means = wide.mean()
            ax.scatter([0, 1], [means["totalvi"], means["fusionvi_xmodal"]], marker="D", s=105, edgecolor="#17223B", linewidth=0.7, color=[PALETTE["totalvi"], PALETTE["fusionvi_xmodal"]], zorder=4)
            ax.set_title(LABELS[target], fontsize=14, weight="bold")
            ax.set_xticks([0, 1], ["totalVI", "FusionVI-X"])
            ax.set_ylim((-0.55, 1.0) if row else (0.25, 1.0))
            if col == 0:
                ax.set_ylabel("Global Spearman rho" if row == 0 else "Discordant-cell Spearman rho")
            else:
                ax.set_ylabel("")
            ax.text(0.5, 0.04, f"mean Δ = {means['fusionvi_xmodal'] - means['totalvi']:+.3f}", transform=ax.transAxes, ha="center", fontsize=10, color="#17223B")
    fig.suptitle("A donor-safe cross-modal readout improves held-out marker recovery", fontsize=20, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "cross_modal_improvement.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    means = pd.read_csv(RESULTS / "cross_modal_head_means.csv")
    models = ["rna_proxy_only", "protein_context_only", "context_no_latent", "totalvi_xmodal", "fusionvi_xmodal"]
    display = ["RNA proxy", "Other ADTs", "RNA + ADTs", "totalVI-X", "FusionVI-X"]
    cols = []
    matrix = []
    for metric, suffix in [("spearman_true_pred", "global"), ("discordant_spearman", "discordant")]:
        for target in LABELS:
            cols.append(f"{LABELS[target].replace(chr(10), ' ')}\n{suffix}")
    for model in models:
        row = []
        for metric in ["spearman_true_pred", "discordant_spearman"]:
            for target in LABELS:
                value = means[(means.model == model) & (means.target == target) & (means.metric == metric)]["value"].iloc[0]
                row.append(value)
        matrix.append(row)
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="mako", vmin=-0.75, vmax=0.9, center=0, xticklabels=cols, yticklabels=display, cbar_kws={"label": "Mean held-out-donor Spearman rho"}, ax=ax)
    ax.set_title("Ablation: protein context rescues markers when RNA evidence disagrees", weight="bold", pad=12)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_modal_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
