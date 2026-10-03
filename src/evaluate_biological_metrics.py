"""Biological secondary metrics for Experiment 4 model predictions.

The primary endpoint remains per-protein RMSLE. This script adds foreground
error, within-cell-type rank recovery, and positive-marker AUROC. When an old
run predates ``target_predictions.npz``, pass ``--generate-missing`` to rebuild
predictions from its saved module weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from scvi.model import TOTALVI

from benchmark_arms import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
RUNS = ROOT / "results" / "paper_benchmark_runs"
MODELS = ROOT / "models" / "paper_benchmark"
OUT = ROOT / "results"


def rho(y, pred) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def generate_predictions(adata, cfg: dict, arm_name: str, seed: int, out: Path) -> None:
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    model = build_model(adata, cfg, cfg["arms"][arm_name])
    state = torch.load(MODELS / f"{arm_name}__seed{seed}.pt", map_location="cpu", weights_only=True)
    model.module.load_state_dict(state)
    model.module.eval()
    target = adata[adata.obs["batch"].astype(str) == cfg["target_batch"]].copy()
    predictions = {}
    for key, background in (("prediction", True), ("foreground_prediction", False)):
        _, frame = model.get_normalized_expression(
            target,
            transform_batch=cfg["source_batch"],
            n_samples=int(cfg["posterior_samples"]),
            return_mean=True,
            include_protein_background=background,
            scale_protein=False,
            return_numpy=False,
            batch_size=512,
        )
        predictions[key] = frame.loc[target.obs_names].to_numpy(dtype=np.float32)
    truth = target.obsm["protein_truth"].astype(float)
    np.savez_compressed(
        out,
        truth=truth.to_numpy(dtype=np.float32),
        protein_names=np.asarray(truth.columns, dtype=str),
        cell_type=np.asarray(target.obs["cell_type"].astype(str), dtype=str),
        **predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["totalvi", "fusionvi"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--generate-missing", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "paper_benchmark.yaml").read_text())
    seeds = args.seeds or cfg["executed_seeds"]
    adata = sc.read_h5ad(DATA)
    source = adata.obs["batch"].astype(str).to_numpy() == cfg["source_batch"]
    source_truth = np.log1p(adata.obsm["protein_truth"].to_numpy(dtype=np.float32)[source])
    protein_names = np.asarray(adata.obsm["protein_truth"].columns, dtype=str)
    marker_tokens = {"CD4": "ADT_CD4_", "CD8": "ADT_CD8a_", "CD19": "ADT_CD19_"}
    thresholds = {}
    for marker, token in marker_tokens.items():
        j = next(i for i, name in enumerate(protein_names) if token in name)
        means = np.sort(GaussianMixture(n_components=2, random_state=2026).fit(source_truth[:, [j]]).means_.reshape(-1))
        thresholds[marker] = (j, float(means.mean()))

    summary_rows, within_rows, auc_rows = [], [], []
    for arm in args.arms:
        for seed in seeds:
            run = RUNS / f"{arm}__seed{seed}"
            archive = run / "target_predictions.npz"
            if not archive.exists():
                if not args.generate_missing or not (MODELS / f"{arm}__seed{seed}.pt").exists():
                    continue
                generate_predictions(adata, cfg, arm, seed, archive)
            arrays = np.load(archive)
            truth_log = np.log1p(arrays["truth"])
            pred_log = np.log1p(np.maximum(arrays["prediction"], 0))
            foreground_log = np.log1p(np.maximum(arrays["foreground_prediction"], 0))
            cell_types = arrays["cell_type"].astype(str)
            label = "totalVI" if arm == "totalvi" else "FusionVI" if arm == "fusionvi" else arm
            summary_rows.append(
                {
                    "model": label,
                    "arm": arm,
                    "seed": seed,
                    "raw_rmsle": float(np.sqrt(np.mean((pred_log - truth_log) ** 2))),
                    "foreground_rmsle": float(np.sqrt(np.mean((foreground_log - truth_log) ** 2))),
                }
            )
            for cell_type in np.unique(cell_types):
                mask = cell_types == cell_type
                if mask.sum() < 30:
                    continue
                for j, protein in enumerate(arrays["protein_names"].astype(str)):
                    within_rows.append(
                        {
                            "model": label,
                            "arm": arm,
                            "seed": seed,
                            "cell_type": cell_type,
                            "protein": protein,
                            "n_cells": int(mask.sum()),
                            "spearman": rho(truth_log[mask, j], foreground_log[mask, j]),
                        }
                    )
            for marker, (j, threshold) in thresholds.items():
                labels = truth_log[:, j] > threshold
                auc_rows.append(
                    {
                        "model": label,
                        "arm": arm,
                        "seed": seed,
                        "marker": marker,
                        "source_log1p_threshold": threshold,
                        "positive_fraction": float(labels.mean()),
                        "auroc": float(roc_auc_score(labels, foreground_log[:, j])),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    within = pd.DataFrame(within_rows)
    auc = pd.DataFrame(auc_rows)
    summary.to_csv(OUT / "paper_benchmark_biological_summary.csv", index=False)
    within.to_csv(OUT / "paper_benchmark_within_celltype.csv", index=False)
    auc.to_csv(OUT / "paper_benchmark_marker_auroc.csv", index=False)
    marker_auc = {}
    if len(auc):
        for model, frame in auc.groupby("model"):
            marker_auc[model] = frame.groupby("marker")["auroc"].mean().round(4).to_dict()
    report = {
        "definitions": {
            "foreground_rmsle": "RMSLE using totalVI's denoised foreground protein estimate",
            "within_celltype": "Spearman calculated inside cell types with at least 30 target cells",
            "marker_auroc": "D2 AUROC using a two-component source-D1 threshold; CD4, CD8a and CD19 were prespecified",
        },
        "completed_runs": int(len(summary)),
        "run_means": summary.groupby("model").mean(numeric_only=True).round(4).to_dict(orient="index") if len(summary) else {},
        "mean_within_celltype_spearman": within.groupby("model")["spearman"].mean().round(4).to_dict() if len(within) else {},
        "mean_marker_auroc": marker_auc,
    }
    (OUT / "paper_benchmark_biological_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
