"""Plot two-replicate cross-mouse recovery on the original totalVI dataset."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = ROOT / "results" / "experiment3_totalvi_original"
OUT = RESULTS / "figures"

LABELS = {
    "ADT_CD20_A0192": "CD20",
    "ADT_CD28_A0204": "CD28",
    "ADT_CD4_A0001": "CD4",
    "ADT_CD8a_A0002": "CD8a",
}
COLORS = {
    "rna_proxy": "#93C5FD",
    "protein_context": "#F9A8D4",
    "totalvi": "#94A3B8",
    "fusionvi": "#14B8A6",
    "totalvi_xmodal": "#6366F1",
    "fusionvi_xmodal": "#F97316",
}


def panel(ax: plt.Axes, summary: pd.DataFrame, raw: pd.DataFrame,
          models: list[str], title: str) -> None:
    targets = list(LABELS)
    x = np.arange(len(targets))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        subset = summary[summary["model"] == model].set_index("target")
        means = [subset.loc[target, "mean_fold_spearman"] for target in targets]
        position = x - 0.4 + width / 2 + index * width
        ax.bar(position, means, width, color=COLORS[model], label=model.replace("_", "-"))
        for target_index, target in enumerate(targets):
            values = raw[(raw["target"] == target) & (raw["model"] == model)]["spearman"]
            ax.scatter(
                np.repeat(position[target_index], len(values)), values,
                s=19, color="white", edgecolor="#172554", linewidth=0.7, zorder=3,
            )
    ax.set_xticks(x, [LABELS[target] for target in targets])
    ax.set_ylim(0.15, 0.80)
    ax.set_ylabel("Spearman correlation")
    ax.set_title(title, loc="left", weight="bold", color="#172554")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")


def main() -> None:
    summary = pd.read_csv(RESULTS / "fold_summary_metrics.csv")
    raw = pd.read_csv(RESULTS / "cross_mouse_metrics.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    panel(axes[0], summary, raw, ["totalvi", "fusionvi"], "A  Native decoder")
    panel(
        axes[1], summary, raw,
        ["rna_proxy", "protein_context", "totalvi_xmodal", "fusionvi_xmodal"],
        "B  Fixed cross-modal readout",
    )
    fig.suptitle(
        "Original totalVI dataset: hidden-marker recovery across mice",
        fontsize=16, weight="bold", color="#172554", y=0.96,
    )
    fig.text(
        0.5, 0.035,
        "Bars: mean of two mouse-held-out folds • dots: individual folds • 16,813 cells",
        ha="center", color="#475569", fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.82, wspace=0.19)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "totalvi_original_cross_mouse.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "totalvi_original_cross_mouse.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
