"""Plot the scale-matched complete-panel re-evaluation from saved CSV results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "calibration_benchmark.png"
COLORS = {"totalvi": "#243B6B", "fusionvi": "#0EA5A4", "rna_ridge": "#F59E0B"}
LABELS = {"totalvi": "totalVI", "fusionvi": "FusionVI", "rna_ridge": "RNA ridge"}


def main() -> None:
    summary = pd.read_csv(RESULTS / "calibration_summary.csv")
    neural = summary[summary["arm"].isin(["totalvi", "fusionvi"])].copy()
    order = ["log_mean", "pred_log", "calibrated"]
    readout_labels = ["log1p(E[y])\noriginal", "E[log1p y]\nposterior", "D1 affine\ncalibrated"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    # A: the metric-scale question.
    x = np.arange(len(order))
    for arm in ("totalvi", "fusionvi"):
        frame = neural[neural["arm"] == arm].set_index("readout").loc[order]
        axes[0].plot(x, frame["rmsle"], marker="o", linewidth=2.4, markersize=7,
                     color=COLORS[arm], label=LABELS[arm])
    ridge = float(summary.loc[summary["arm"] == "rna_ridge", "rmsle"].iloc[0])
    axes[0].axhline(ridge, color=COLORS["rna_ridge"], linestyle="--", linewidth=2,
                    label=f"RNA ridge ({ridge:.3f})")
    axes[0].set_xticks(x, readout_labels)
    axes[0].set_ylabel("Mean protein RMSLE")
    axes[0].set_ylim(0.50, 1.11)
    axes[0].set_title("A  Error depends on readout scale", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    # B: show that calibration removes offset while leaving residual spread.
    groups = [("log_mean", "abs_bias"), ("log_mean", "residual_sd"),
              ("calibrated", "abs_bias"), ("calibrated", "residual_sd")]
    group_labels = ["Original\n|bias|", "Original\nresidual", "Calibrated\n|bias|", "Calibrated\nresidual"]
    width = 0.35
    gx = np.arange(len(groups))
    for offset, arm in ((-width / 2, "totalvi"), (width / 2, "fusionvi")):
        arm_frame = neural[neural["arm"] == arm].set_index("readout")
        vals = [float(arm_frame.loc[r, metric]) for r, metric in groups]
        axes[1].bar(gx + offset, vals, width, color=COLORS[arm], label=LABELS[arm])
    axes[1].set_xticks(gx, group_labels)
    axes[1].set_ylabel("Mean log1p error component")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("B  Calibration removes mean offset", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    # C: scale-free and phenotype-separation readouts after calibration.
    metrics = ["spearman", "within_celltype_spearman", "marker_auroc"]
    metric_labels = ["Overall\nSpearman", "Within-cell-type\nSpearman", "CD4/CD8/CD19\nAUROC"]
    cy = np.arange(len(metrics))[::-1]
    calibrated = neural[neural["readout"] == "calibrated"].set_index("arm")
    for arm, marker, dy in (("totalvi", "o", 0.08), ("fusionvi", "s", -0.08)):
        vals = calibrated.loc[arm, metrics].to_numpy(dtype=float)
        axes[2].scatter(vals, cy + dy, s=62, marker=marker, color=COLORS[arm], label=LABELS[arm], zorder=3)
        for value, ypos in zip(vals, cy + dy):
            axes[2].text(value + 0.018, ypos, f"{value:.3f}", va="center", fontsize=9, color=COLORS[arm])
    axes[2].set_yticks(cy, metric_labels)
    axes[2].set_xlim(0, 1.08)
    axes[2].set_xlabel("Metric value")
    axes[2].set_title("C  Information recovery is similar", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.68, 0.48))

    for ax in axes:
        ax.grid(axis="y", color="#DCE3EC", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Scale-matched re-evaluation of the complete-panel benchmark", x=0.02,
                 ha="left", fontsize=17, weight="bold", color="#172554")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
