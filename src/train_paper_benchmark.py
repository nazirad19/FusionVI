"""Train totalVI or FusionVI on the paper's all-proteins-missing benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from scvi.model import TOTALVI

from fusionvi import FusionVIEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
RUNS = ROOT / "results" / "paper_benchmark_runs"
MODELS = ROOT / "models" / "paper_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["totalvi", "fusionvi"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    with (ROOT / "config" / "paper_benchmark.yaml").open() as handle:
        cfg = yaml.safe_load(handle)

    run_name = f"{args.model}__seed{args.seed}"
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "complete.json"
    if complete.exists():
        print(f"Skipping completed paper benchmark {run_name}")
        return

    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision("high")
    adata = sc.read_h5ad(DATA)
    TOTALVI.setup_anndata(
        adata,
        batch_key="batch",
        layer="counts",
        protein_expression_obsm_key="protein_counts",
    )
    model = TOTALVI(
        adata,
        n_latent=int(cfg["n_latent"]),
        n_hidden=int(cfg["totalvi_hidden"]),
        n_layers_encoder=2,
        n_layers_decoder=1,
        gene_likelihood="nb",
        latent_distribution="normal",
        empirical_protein_background_prior=False,
    )
    if args.model == "fusionvi":
        model.module.encoder = FusionVIEncoder(
            n_genes=adata.n_vars,
            n_proteins=adata.obsm["protein_counts"].shape[1],
            n_latent=int(cfg["n_latent"]),
            masked_protein_indices=[],
            n_cat_list=[model.module.n_batch],
            n_layers=2,
            n_hidden=int(cfg["fusion_branch_hidden"]),
            dropout_rate=0.2,
            distribution="normal",
        )

    effective_max_epochs = int(args.max_epochs or cfg["max_epochs"])
    model.train(
        max_epochs=effective_max_epochs,
        lr=float(cfg["learning_rate"]),
        accelerator="gpu",
        devices=1,
        train_size=0.9,
        validation_size=0.1,
        batch_size=int(cfg["batch_size"]),
        early_stopping=True,
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        check_val_every_n_epoch=1,
        reduce_lr_on_plateau=True,
        adversarial_classifier=True,
        datasplitter_kwargs={"num_workers": 4, "persistent_workers": True, "pin_memory": True},
        enable_progress_bar=True,
    )

    target = adata[adata.obs["batch"].astype(str) == cfg["target_batch"]].copy()
    _, predicted = model.get_normalized_expression(
        target,
        transform_batch=cfg["source_batch"],
        n_samples=int(cfg["posterior_samples"]),
        return_mean=True,
        include_protein_background=True,
        scale_protein=False,
        return_numpy=False,
    )
    truth = target.obsm["protein_truth"].astype(float)
    predicted = predicted.loc[truth.index, truth.columns].clip(lower=0.0)
    rows = []
    for protein in truth.columns:
        observed = truth[protein].to_numpy()
        estimate = predicted[protein].to_numpy()
        observed_log = np.log1p(observed)
        estimate_log = np.log1p(estimate)
        rows.append(
            {
                "model": "totalVI" if args.model == "totalvi" else "FusionVI",
                "seed": args.seed,
                "protein": protein,
                "rmsle": float(np.sqrt(np.mean(np.square(estimate_log - observed_log)))),
                "mae_log1p": float(np.mean(np.abs(estimate_log - observed_log))),
                "mae_raw": float(np.mean(np.abs(estimate - observed))),
                "spearman": float(spearmanr(observed, estimate).statistic),
                "pearson_log1p": float(pearsonr(observed_log, estimate_log).statistic),
                "n_target_cells": int(target.n_obs),
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "protein_metrics.csv", index=False)
    history = []
    for metric, frame in model.history.items():
        for epoch, value in enumerate(np.asarray(frame).reshape(-1)):
            history.append({"metric": metric, "epoch": epoch, "value": float(value)})
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.module.state_dict(), MODELS / f"{run_name}.pt")
    result = {
        "model": "totalVI" if args.model == "totalvi" else "FusionVI",
        "seed": args.seed,
        "configured_max_epochs": effective_max_epochs,
        "epochs_completed": int(model.history["elbo_train"].shape[0]),
        "trainable_parameters": int(sum(p.numel() for p in model.module.parameters() if p.requires_grad)),
        "mean_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].mean()),
        "median_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].median()),
    }
    complete.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
