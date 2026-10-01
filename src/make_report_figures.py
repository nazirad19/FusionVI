"""Create compact scientific figures for the FusionVI report and talk."""

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
COLORS = {"totalvi": "#4F67D8", "fusionvi": "#F15B64"}
MODEL_LABELS = {
    "rna_pca": "RNA PCA",
    "protein_pca": "Protein PCA",
    "equal_fusion_pca": "Equal fusion PCA",
    "totalvi": "totalVI",
    "fusionvi": "FusionVI",
}
TARGET_LABELS = {
    "T_cell_direct": "T cells\nanti-CD3/CD28",
    "Monocyte_direct": "Monocytes\nLPS",
    "NK_indirect": "NK cells\nanti-CD3/CD28",
    "B_cell_indirect": "B cells\nanti-CD3/CD28",
    "CD25_T": "CD25 in T cells",
    "CD69_T": "CD69 in T cells",
    "HLA_DR_Monocyte": "HLA-DR in monocytes",
}


def paired_axes(ax, data: pd.DataFrame, title: str, ylabel: str, ylim=None) -> None:
    wide = data.pivot(index="donor", columns="model", values="value").dropna()
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row["totalvi"], row["fusionvi"]], color="#C8CCD8", linewidth=1.1, zorder=1)
    for x, model in enumerate(["totalvi", "fusionvi"]):
        values = wide[model]
        ax.scatter(np.repeat(x, len(values)), values, s=36, color=COLORS[model], edgecolor="white", linewidth=0.5, zorder=2)
        ax.scatter(x, values.mean(), s=120, color=COLORS[model], edgecolor="#17223B", marker="D", linewidth=0.8, zorder=3)
    ax.set_xticks([0, 1], ["totalVI", "FusionVI"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "cv_metrics.csv")
    sns.set_theme(style="whitegrid", context="talk")

    # Direct and indirect stimulation classification in held-out donors.
    response = metrics[(metrics["analysis"] == "latent_response") & (metrics["metric"] == "auc") & metrics["model"].isin(COLORS)]
    targets = ["T_cell_direct", "Monocyte_direct", "NK_indirect", "B_cell_indirect"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.3), sharey=True)
    for ax, target in zip(axes, targets):
        paired_axes(ax, response[response["target"] == target], TARGET_LABELS[target], "ROC AUC", (0.94, 1.005))
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle("Stimulation state prediction is already at ceiling", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "response_auc_ceiling.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Held-out marker reconstruction in all relevant and discordant cells.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for col, target in enumerate(["CD25_T", "CD69_T", "HLA_DR_Monocyte"]):
        global_data = metrics[(metrics["analysis"] == "heldout_marker") & (metrics["target"] == target) & (metrics["metric"] == "spearman_true_pred") & metrics["model"].isin(COLORS)]
        paired_axes(axes[0, col], global_data, TARGET_LABELS[target], "Spearman rho", (-0.35, 1.0))
        discordant = metrics[(metrics["analysis"] == "heldout_marker") & (metrics["target"] == target) & (metrics["metric"] == "discordant_spearman") & metrics["model"].isin(COLORS)]
        paired_axes(axes[1, col], discordant, "RNA-protein discordant cells", "Spearman rho", (-0.75, 1.0))
    fig.suptitle("FusionVI changes marker recovery only modestly and inconsistently", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_marker_recovery.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Donor-level gate behavior.
    gates = pd.read_csv(RESULTS / "fusion_gates_all_cells.csv", index_col=0)
    donor_condition = gates.groupby(["donor", "condition"], as_index=False)["protein_gate"].mean()
    gate_discordance = pd.read_csv(RESULTS / "gate_discordance_by_donor.csv")
    gate_wide = gate_discordance.pivot(index=["donor", "target"], columns="stratum", values="mean_protein_gate").reset_index()
    gate_wide["difference"] = gate_wide["discordant"] - gate_wide["concordant"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.boxplot(data=donor_condition, x="condition", y="protein_gate", color="#CDECEE", width=0.55, showfliers=False, ax=axes[0])
    sns.stripplot(data=donor_condition, x="condition", y="protein_gate", color="#167D8D", size=6, jitter=0.12, ax=axes[0])
    axes[0].set_title("Protein weight by stimulation", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean protein branch weight per donor")
    target_order = ["CD25_T", "CD69_T", "HLA_DR_Monocyte"]
    gate_wide["target_label"] = gate_wide["target"].map(TARGET_LABELS)
    sns.boxplot(data=gate_wide, x="target_label", y="difference", order=[TARGET_LABELS[t] for t in target_order], color="#FFD8B8", width=0.55, showfliers=False, ax=axes[1])
    sns.stripplot(data=gate_wide, x="target_label", y="difference", order=[TARGET_LABELS[t] for t in target_order], color="#D46813", size=6, jitter=0.12, ax=axes[1])
    axes[1].axhline(0, color="#26324B", linewidth=1)
    axes[1].set_title("Gate response to RNA-protein discordance", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Discordant minus concordant protein weight")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_gate_findings.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Compact baseline comparison heatmaps.
    order = ["rna_pca", "protein_pca", "equal_fusion_pca", "totalvi", "fusionvi"]
    response_table = response.copy()
    # Restore classical models for this figure.
    response_table = metrics[(metrics["analysis"] == "latent_response") & (metrics["metric"] == "auc")]
    response_table = response_table.groupby(["target", "model"])["value"].mean().unstack().reindex(index=targets, columns=order)
    marker_table = metrics[(metrics["analysis"] == "heldout_marker") & (metrics["metric"] == "spearman_true_pred")]
    marker_table = marker_table.groupby(["target", "model"])["value"].mean().unstack().reindex(index=["CD25_T", "CD69_T", "HLA_DR_Monocyte"], columns=order)
    response_table.index = [TARGET_LABELS[x].replace("\n", " ") for x in response_table.index]
    marker_table.index = [TARGET_LABELS[x] for x in marker_table.index]
    response_table.columns = [MODEL_LABELS[x] for x in response_table.columns]
    marker_table.columns = [MODEL_LABELS[x] for x in marker_table.columns]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    sns.heatmap(response_table, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.65, vmax=1.0, cbar_kws={"label": "ROC AUC"}, ax=axes[0])
    axes[0].set_title("Held-out donor stimulation classification", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(marker_table, annot=True, fmt=".3f", cmap="mako", vmin=0.0, vmax=0.85, cbar_kws={"label": "Spearman rho"}, ax=axes[1])
    axes[1].set_title("Held-out activation-marker reconstruction", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
