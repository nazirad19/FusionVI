"""Train one perturbation-held-out Papalexi ECCITE-seq neural fold."""

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
DATA = ROOT / "data" / "processed" / "papalexi_eccite.h5ad"
RESULTS = ROOT / "results" / "experiment2_papalexi" / "folds"
MODELS = ROOT / "models" / "experiment2_papalexi"


def load_config() -> dict:
    with (ROOT / "config" / "papalexi.yaml").open() as handle:
        return yaml.safe_load(handle)


def history_frame(model: TOTALVI) -> pd.DataFrame:
    rows = []
    for metric, frame in model.history.items():
        values = np.asarray(frame).reshape(-1)
        rows.extend({"metric": metric, "epoch": i, "value": float(v)} for i, v in enumerate(values))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["totalvi", "fusionvi"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()

    cfg = load_config()
    if not 0 <= args.fold < int(cfg["n_outer_folds"]):
        raise ValueError(f"Fold must be between 0 and {int(cfg['n_outer_folds']) - 1}")
    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision("high")
    adata = sc.read_h5ad(DATA)
    train = adata[adata.obs["outer_fold"].astype(int) != args.fold].copy()
    test = adata[adata.obs["outer_fold"].astype(int) == args.fold].copy()

    target = str(cfg["heldout_protein"])
    protein_names = list(map(str, train.obsm["protein_counts"].columns))
    if target not in protein_names:
        raise RuntimeError(f"Held-out protein is absent: {target}")
    masked_indices = [protein_names.index(target)]

    fold_name = f"{args.model}__fold{args.fold}__seed{args.seed}"
    fold_dir = RESULTS / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    complete = fold_dir / "complete.json"
    if complete.exists():
        print(f"Skipping completed fold {fold_name}")
        return

    TOTALVI.setup_anndata(
        train,
        layer="counts",
        batch_key="lane",
        protein_expression_obsm_key="protein_counts",
    )
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
    encoder_kwargs = dict(
        n_genes=train.n_vars,
        n_proteins=len(protein_names),
        n_latent=int(cfg["n_latent"]),
        masked_protein_indices=masked_indices,
        n_cat_list=[model.module.n_batch],
        n_layers=2,
        n_hidden=int(cfg["n_hidden"]),
        dropout_rate=0.2,
        distribution="normal",
    )
    if args.model == "totalvi":
        model.module.encoder = MaskedJointEncoderTOTALVI(**encoder_kwargs)
    else:
        model.module.encoder = DualBranchEncoderTOTALVI(**encoder_kwargs)

    model.train(
        max_epochs=int(args.max_epochs or cfg["max_epochs"]),
        accelerator="gpu",
        devices=1,
        batch_size=int(cfg["batch_size"]),
        early_stopping=True,
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    history_frame(model).to_csv(fold_dir / "training_history.csv", index=False)

    z_train = model.get_latent_representation(train, batch_size=int(cfg["batch_size"]))
    z_test = model.get_latent_representation(test, batch_size=int(cfg["batch_size"]))
    np.save(fold_dir / "latent_train.npy", z_train.astype(np.float32))
    np.save(fold_dir / "latent_test.npy", z_test.astype(np.float32))

    _, predicted_protein = model.get_normalized_expression(
        test,
        n_samples=10,
        return_mean=True,
        return_numpy=False,
        batch_size=int(cfg["batch_size"]),
    )
    metadata_columns = ["gene", "guide", "replicate", "lane", "crispr", "outer_fold"]
    prediction = test.obs[metadata_columns].copy()
    prediction["true_PDL1"] = np.asarray(test.obsm["protein_counts"][target])
    prediction["pred_PDL1"] = np.asarray(predicted_protein[target])
    prediction.to_csv(fold_dir / "native_predictions.csv")
    train.obs[metadata_columns].to_csv(fold_dir / "train_metadata.csv")
    test.obs[metadata_columns].to_csv(fold_dir / "test_metadata.csv")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.module.state_dict(), MODELS / f"{fold_name}.pt")
    heldout_targets = sorted(set(test.obs.loc[test.obs["gene"] != "NT", "gene"].astype(str)))
    complete.write_text(
        json.dumps(
            {
                "model": args.model,
                "fold": args.fold,
                "seed": args.seed,
                "epochs_completed": int(model.history["elbo_train"].shape[0]),
                "train_cells": int(train.n_obs),
                "test_cells": int(test.n_obs),
                "heldout_targets": heldout_targets,
                "masked_protein": target,
            },
            indent=2,
        )
    )
    print(f"Completed {fold_name}: {heldout_targets}", flush=True)


if __name__ == "__main__":
    main()

