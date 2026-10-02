"""Train one cross-mouse fold on the original totalVI SLN111 data."""

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
from scvi.model import TOTALVI

from fusion_encoder import DualBranchEncoderTOTALVI, MaskedJointEncoderTOTALVI


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "totalvi_original_sln111.h5ad"
RESULTS = ROOT / "results" / "experiment3_totalvi_original" / "folds"
MODELS = ROOT / "models" / "experiment3_totalvi_original"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["totalvi", "fusionvi"], required=True)
    parser.add_argument("--fold", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with (ROOT / "config" / "totalvi_original.yaml").open() as handle:
        cfg = yaml.safe_load(handle)

    fold_name = f"{args.model}__mouse{args.fold}__seed{args.seed}"
    fold_dir = RESULTS / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    complete = fold_dir / "complete.json"
    if complete.exists():
        print(f"Skipping completed fold {fold_name}")
        return

    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision("high")
    adata = sc.read_h5ad(DATA)
    test_mouse = f"mouse{args.fold}"
    train = adata[adata.obs["mouse"].astype(str) != test_mouse].copy()
    test = adata[adata.obs["mouse"].astype(str) == test_mouse].copy()
    protein_names = list(map(str, train.obsm["protein_counts"].columns))
    masked = [protein_names.index(name) for name in cfg["targets"]]

    TOTALVI.setup_anndata(train, layer="counts", protein_expression_obsm_key="protein_counts")
    model = TOTALVI(
        train,
        n_latent=int(cfg["n_latent"]),
        n_hidden=int(cfg["n_hidden"]),
        n_layers_encoder=2,
        n_layers_decoder=1,
        gene_likelihood="nb",
        latent_distribution="normal",
        empirical_protein_background_prior=False,
    )
    kwargs = dict(
        n_genes=train.n_vars,
        n_proteins=len(protein_names),
        n_latent=int(cfg["n_latent"]),
        masked_protein_indices=masked,
        n_cat_list=[model.module.n_batch],
        n_layers=2,
        n_hidden=int(cfg["n_hidden"]),
        dropout_rate=0.2,
        distribution="normal",
    )
    model.module.encoder = (
        MaskedJointEncoderTOTALVI(**kwargs)
        if args.model == "totalvi"
        else DualBranchEncoderTOTALVI(**kwargs)
    )
    model.train(
        max_epochs=int(cfg["max_epochs"]),
        accelerator="gpu",
        devices=1,
        batch_size=int(cfg["batch_size"]),
        early_stopping=True,
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    history = []
    for metric, frame in model.history.items():
        for epoch, value in enumerate(np.asarray(frame).reshape(-1)):
            history.append({"metric": metric, "epoch": epoch, "value": float(value)})
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    np.save(fold_dir / "latent_train.npy", model.get_latent_representation(train).astype("float32"))
    np.save(fold_dir / "latent_test.npy", model.get_latent_representation(test).astype("float32"))
    _, predicted = model.get_normalized_expression(test, n_samples=10, return_mean=True, return_numpy=False)
    metadata = ["mouse", "tissue", "cell_type"]
    native = test.obs[metadata].copy()
    for target in cfg["targets"]:
        native[f"true__{target}"] = np.asarray(test.obsm["protein_counts"][target])
        native[f"pred__{target}"] = np.asarray(predicted[target])
    native.to_csv(fold_dir / "native_predictions.csv")
    train.obs[metadata].to_csv(fold_dir / "train_metadata.csv")
    test.obs[metadata].to_csv(fold_dir / "test_metadata.csv")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.module.state_dict(), MODELS / f"{fold_name}.pt")
    complete.write_text(json.dumps({
        "model": args.model,
        "heldout_mouse": test_mouse,
        "train_cells": int(train.n_obs),
        "test_cells": int(test.n_obs),
        "epochs_completed": int(model.history["elbo_train"].shape[0]),
        "masked_proteins": list(cfg["targets"]),
    }, indent=2))
    print(f"Completed {fold_name}", flush=True)


if __name__ == "__main__":
    main()

