"""Aggregate the paper-aligned missing-protein benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "paper_benchmark_runs"
OUT = ROOT / "results"


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

    statistic, p_value = wilcoxon(pivot["FusionVI"], pivot["totalVI"], alternative="two-sided")
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
        "interpretation_limit": "Proteins and random initializations are algorithmic benchmark units, not independent biological cohorts.",
        "completed_runs": completed,
    }
    (OUT / "paper_benchmark_headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
