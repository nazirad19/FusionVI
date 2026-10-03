"""Plot the final totalVI versus FusionVI comparison."""

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


RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
LABELS = {
    "ADT_CD20_A0192": "CD20",
    "ADT_CD28_A0204": "CD28",
    "ADT_CD4_A0001": "CD4",
    "ADT_CD8a_A0002": "CD8a",
}
COLORS = {"totalVI": "#64748B", "FusionVI": "#F97316"}


def main() -> None:
    summary = pd.read_csv(RESULTS / "marker_summary.csv")
    raw = pd.read_csv(RESULTS / "cross_mouse_metrics.csv")
    targets = list(LABELS)
    models = ["totalVI", "FusionVI"]
    x = np.arange(len(targets))
    width = 0.34

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for index, model in enumerate(models):
        frame = summary[summary["model"] == model].set_index("target")
        means = [frame.loc[target, "mean_fold_spearman"] for target in targets]
        positions = x + (index - 0.5) * width
        bars = ax.bar(positions, means, width, color=COLORS[model], label=model, zorder=2)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=4, fontsize=9)
        for target_index, target in enumerate(targets):
            values = raw[(raw["target"] == target) & (raw["model"] == model)]["spearman"]
            ax.scatter(
                np.repeat(positions[target_index], len(values)), values,
                s=26, color="white", edgecolor="#172554", linewidth=0.8, zorder=3,
            )

    total_mean = summary[summary["model"] == "totalVI"]["mean_fold_spearman"].mean()
    fusion_mean = summary[summary["model"] == "FusionVI"]["mean_fold_spearman"].mean()
    ax.text(
        0.99, 0.05,
        f"Mean across markers\ntotalVI  {total_mean:.3f}\nFusionVI  {fusion_mean:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#172554",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#FFF7ED", "edgecolor": "#FDBA74"},
    )
    ax.set_xticks(x, [LABELS[target] for target in targets])
    ax.set_ylim(0.40, 0.82)
    ax.set_ylabel("Spearman correlation")
    fig.suptitle(
        "Hidden surface-marker recovery in an unseen mouse",
        x=0.10, y=0.96, ha="left", fontsize=17, weight="bold", color="#172554",
    )
    fig.text(
        0.10, 0.895,
        "Bars show the mean of two mouse-held-out folds; dots show each fold",
        color="#475569", fontsize=10,
    )
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.82)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "totalvi_vs_fusionvi.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "totalvi_vs_fusionvi.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
