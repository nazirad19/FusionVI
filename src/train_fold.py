"""Train one held-out-donor fold of masked totalVI or FusionVI."""

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
DATA = ROOT / "data" / "processed" / "lawlor_pbmc_citeseq.h5ad"
RESULTS = ROOT / "results" / "folds"
MODELS = ROOT / "models"


def load_config() -> dict:
    with (ROOT / "config" / "default.yaml").open() as handle:
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
    parser.add_argument("--heldout-donor", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()

    cfg = load_config()
    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision("high")
    adata = sc.read_h5ad(DATA)
    donor = args.heldout_donor
    if donor not in set(adata.obs["donor"].astype(str)):
        raise ValueError(f"Unknown donor: {donor}")
    train = adata[adata.obs["donor"].astype(str) != donor].copy()
    test = adata[adata.obs["donor"].astype(str) == donor].copy()
    heldout = list(cfg["heldout_proteins"])
    protein_names = list(map(str, train.obsm["protein_counts"].columns))
    missing = sorted(set(heldout).difference(protein_names))
    if missing:
        raise RuntimeError(f"Held-out proteins absent: {missing}")
    masked_indices = [protein_names.index(name) for name in heldout]

    fold_name = f"{args.model}__{donor}__seed{args.seed}"
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

    # scvi-tools transfers the training registry to this AnnData. All lanes are
    # represented in the training donors, so no unseen batch category is added.
    z_train = model.get_latent_representation(train)
    gate_chunks: list[np.ndarray] = []
    hook = None
    if args.model == "fusionvi":
        # Capture the scalar fusion gate during deterministic test-set encoding.
        # A value near 1 means the encoder relied more on its RNA branch; a
        # value near 0 means it relied more on its protein branch.
        def capture_gate(module, _inputs, _output) -> None:
            gate_chunks.append(module.last_gate.detach().cpu().numpy().reshape(-1))

        hook = model.module.encoder.register_forward_hook(capture_gate)
    z_test = model.get_latent_representation(test, batch_size=int(cfg["batch_size"]))
    if hook is not None:
        hook.remove()
        gates = np.concatenate(gate_chunks)
        if len(gates) != test.n_obs:
            raise RuntimeError(f"Captured {len(gates)} gates for {test.n_obs} test cells")
        gate_frame = test.obs[["donor", "lane", "condition", "cell_type"]].copy()
        gate_frame["rna_gate"] = gates
        gate_frame["protein_gate"] = 1.0 - gates
        gate_frame.to_csv(fold_dir / "test_fusion_gates.csv")
    np.save(fold_dir / "latent_train.npy", z_train.astype(np.float32))
    np.save(fold_dir / "latent_test.npy", z_test.astype(np.float32))

    _, predicted_protein = model.get_normalized_expression(
        test,
        n_samples=25,
        return_mean=True,
        return_numpy=False,
        batch_size=int(cfg["batch_size"]),
    )
    prediction = test.obs[["donor", "lane", "condition", "cell_type"]].copy()
    for marker in heldout:
        prediction[f"true_{marker}"] = np.asarray(test.obsm["protein_counts"][marker])
        prediction[f"pred_{marker}"] = np.asarray(predicted_protein[marker])
    prediction.to_csv(fold_dir / "masked_protein_predictions.csv")
    train.obs[["donor", "lane", "condition", "cell_type"]].to_csv(fold_dir / "train_metadata.csv")
    test.obs[["donor", "lane", "condition", "cell_type"]].to_csv(fold_dir / "test_metadata.csv")
    torch.save(model.module.state_dict(), MODELS / f"{fold_name}.pt")
    with complete.open("w") as handle:
        json.dump(
            {
                "model": args.model,
                "heldout_donor": donor,
                "seed": args.seed,
                "epochs_completed": int(model.history["elbo_train"].shape[0]),
                "train_cells": int(train.n_obs),
                "test_cells": int(test.n_obs),
                "masked_proteins": heldout,
            },
            handle,
            indent=2,
        )
    print(f"Completed {fold_name}")


if __name__ == "__main__":
    MODELS.mkdir(parents=True, exist_ok=True)
    main()
