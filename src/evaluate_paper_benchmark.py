"""Aggregate the paper-aligned missing-protein benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "paper_benchmark_runs"
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"totalVI": "#64748B", "FusionVI": "#F97316"}


def make_figure(pivot: pd.DataFrame, seed_pivot: pd.DataFrame) -> None:
    """Create the main paper-benchmark figure from completed paired runs."""
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    # A. Paired initialization means.
    ax = axes[0]
    for seed, row in seed_pivot.iterrows():
        ax.plot([0, 1], [row["totalVI"], row["FusionVI"]], color="#CBD5E1", lw=1.4, zorder=1)
        ax.scatter(0, row["totalVI"], color=COLORS["totalVI"], s=42, zorder=2)
        ax.scatter(1, row["FusionVI"], color=COLORS["FusionVI"], s=42, zorder=2)
    ax.set_xticks([0, 1], ["totalVI", "FusionVI"])
    ax.set_ylabel("Mean protein RMSLE")
    ax.set_title("A  Paired initializations", loc="left", weight="bold")
    ax.text(0.02, 0.02, "Lower is better", transform=ax.transAxes, color="#475569")

    # B. Average error for every protein.
    ax = axes[1]
    limit_low = float(min(pivot["totalVI"].min(), pivot["FusionVI"].min()))
    limit_high = float(max(pivot["totalVI"].max(), pivot["FusionVI"].max()))
    pad = (limit_high - limit_low) * 0.07
    ax.scatter(pivot["totalVI"], pivot["FusionVI"], s=22, color="#0EA5E9", alpha=0.78)
    ax.plot([limit_low - pad, limit_high + pad], [limit_low - pad, limit_high + pad], ls="--", color="#94A3B8")
    ax.set_xlim(limit_low - pad, limit_high + pad)
    ax.set_ylim(limit_low - pad, limit_high + pad)
    ax.set_xlabel("totalVI RMSLE")
    ax.set_ylabel("FusionVI RMSLE")
    ax.set_title("B  Error for each protein", loc="left", weight="bold")
    wins = int(pivot["fusionvi_better"].sum())
    ax.text(0.03, 0.95, f"FusionVI lower for {wins}/{len(pivot)} proteins", transform=ax.transAxes, va="top")

    # C. Distribution of paired per-protein differences.
    ax = axes[2]
    differences = pivot["difference_fusion_minus_total"].to_numpy()
    bins = min(24, max(10, int(np.sqrt(len(differences)) * 2)))
    ax.hist(differences, bins=bins, color="#FDBA74", edgecolor="white")
    ax.axvline(0, color="#475569", ls="--", lw=1.3)
    ax.axvline(float(np.mean(differences)), color="#C2410C", lw=2)
    ax.set_xlabel("FusionVI minus totalVI RMSLE")
    ax.set_ylabel("Proteins")
    ax.set_title("C  Paired error difference", loc="left", weight="bold")
    ax.text(0.03, 0.95, "Negative values favor FusionVI", transform=ax.transAxes, va="top")

    for ax in axes:
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paper-aligned missing-protein benchmark", x=0.06, y=1.02, ha="left", fontsize=17, weight="bold", color="#172554")
    fig.tight_layout()
    fig.savefig(figures / "paper_benchmark_totalvi_vs_fusionvi.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "paper_benchmark_totalvi_vs_fusionvi.svg", bbox_inches="tight")
    plt.close(fig)

    # Wider-label, two-panel version sized for the presentation layout.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    ax = axes[0]
    for seed, row in seed_pivot.iterrows():
        ax.plot([0, 1], [row["totalVI"], row["FusionVI"]], color="#CBD5E1", lw=1.7, zorder=1)
        ax.scatter(0, row["totalVI"], color=COLORS["totalVI"], s=58, zorder=2)
        ax.scatter(1, row["FusionVI"], color=COLORS["FusionVI"], s=58, zorder=2)
    ax.set_xticks([0, 1], ["totalVI", "FusionVI"])
    ax.set_ylabel("Mean protein RMSLE")
    ax.set_title("Paired initializations", loc="left", weight="bold")
    ax.text(0.02, 0.02, "Lower is better", transform=ax.transAxes, color="#475569")

    ax = axes[1]
    ax.scatter(pivot["totalVI"], pivot["FusionVI"], s=30, color="#0EA5E9", alpha=0.8)
    ax.plot([limit_low - pad, limit_high + pad], [limit_low - pad, limit_high + pad], ls="--", color="#94A3B8")
    ax.set_xlim(limit_low - pad, limit_high + pad)
    ax.set_ylim(limit_low - pad, limit_high + pad)
    ax.set_xlabel("totalVI RMSLE")
    ax.set_ylabel("FusionVI RMSLE")
    ax.set_title("Mean error for each protein", loc="left", weight="bold")
    ax.text(0.03, 0.96, f"FusionVI lower for {wins}/{len(pivot)} proteins", transform=ax.transAxes, va="top")
    for ax in axes:
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original totalVI missing-protein benchmark", x=0.07, y=1.01, ha="left", fontsize=17, weight="bold", color="#172554")
    fig.tight_layout()
    fig.savefig(figures / "paper_benchmark_slide.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with (ROOT / "config" / "paper_benchmark.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    frames = []
    completed = []
    for model in ("totalvi", "fusionvi"):
        for seed in cfg["executed_seeds"]:
            run_dir = RUNS / f"{model}__seed{seed}"
            metrics = run_dir / "protein_metrics.csv"
            if metrics.exists():
                frames.append(pd.read_csv(metrics))
                completed.append(json.loads((run_dir / "complete.json").read_text()))
    if not frames:
        raise RuntimeError("No completed paper benchmark runs were found")
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(OUT / "paper_benchmark_all_metrics.csv", index=False)

    protein_summary = (
        all_metrics.groupby(["protein", "model"], as_index=False)
        .agg(
            mean_rmsle=("rmsle", "mean"),
            sd_rmsle=("rmsle", "std"),
            mean_spearman=("spearman", "mean"),
            mean_pearson_log1p=("pearson_log1p", "mean"),
            n_initializations=("seed", "nunique"),
        )
    )
    protein_summary.to_csv(OUT / "paper_benchmark_protein_summary.csv", index=False)
    pivot = protein_summary.pivot(index="protein", columns="model", values="mean_rmsle").dropna()
    pivot["difference_fusion_minus_total"] = pivot["FusionVI"] - pivot["totalVI"]
    pivot["fusionvi_better"] = pivot["difference_fusion_minus_total"] < 0
    pivot.reset_index().to_csv(OUT / "paper_benchmark_paired_proteins.csv", index=False)

    seed_summary = (
        all_metrics.groupby(["seed", "model"], as_index=False)
        .agg(mean_rmsle=("rmsle", "mean"), median_rmsle=("rmsle", "median"))
    )
    seed_summary.to_csv(OUT / "paper_benchmark_seed_summary.csv", index=False)
    seed_pivot = seed_summary.pivot(index="seed", columns="model", values="mean_rmsle").dropna()

    make_figure(pivot, seed_pivot)

    statistic, p_value = wilcoxon(pivot["FusionVI"], pivot["totalVI"], alternative="two-sided")
    model_means = all_metrics.groupby("model")[["rmsle", "mae_log1p", "mae_raw", "spearman", "pearson_log1p"]].mean()
    secondary = {}
    for metric in ("mae_log1p", "mae_raw", "spearman", "pearson_log1p"):
        secondary[f"totalvi_mean_{metric}"] = float(model_means.loc["totalVI", metric])
        secondary[f"fusionvi_mean_{metric}"] = float(model_means.loc["FusionVI", metric])
        secondary[f"fusionvi_minus_totalvi_{metric}"] = float(
            model_means.loc["FusionVI", metric] - model_means.loc["totalVI", metric]
        )
    headline = {
        "paper_experiment": cfg["benchmark"],
        "metric": cfg["primary_metric"],
        "paper_protocol_initializations": int(cfg["paper_initializations"]),
        "completed_totalvi_initializations": int(all_metrics.loc[all_metrics["model"] == "totalVI", "seed"].nunique()),
        "completed_fusionvi_initializations": int(all_metrics.loc[all_metrics["model"] == "FusionVI", "seed"].nunique()),
        "totalvi_mean_rmsle": float(all_metrics.loc[all_metrics["model"] == "totalVI", "rmsle"].mean()),
        "fusionvi_mean_rmsle": float(all_metrics.loc[all_metrics["model"] == "FusionVI", "rmsle"].mean()),
        "fusionvi_minus_totalvi_rmsle": float(pivot["difference_fusion_minus_total"].mean()),
        "proteins_fusionvi_better": int(pivot["fusionvi_better"].sum()),
        "proteins_compared": int(len(pivot)),
        "protein_level_paired_wilcoxon_p": float(p_value),
        "all_completed_seeds_favor_fusionvi": bool((seed_pivot["FusionVI"] < seed_pivot["totalVI"]).all()) if len(seed_pivot) else False,
        **secondary,
        "interpretation_limit": "Proteins and random initializations are algorithmic benchmark units, not independent biological cohorts.",
        "completed_runs": completed,
    }
    (OUT / "paper_benchmark_headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
