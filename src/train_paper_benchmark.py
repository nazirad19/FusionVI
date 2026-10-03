"""Train totalVI or FusionVI on the paper's all-proteins-missing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from scvi.model import TOTALVI

from benchmark_arms import build_model, n_trainable

LABELS = {"totalvi": "totalVI", "fusionvi": "FusionVI"}


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
RUNS = ROOT / "results" / "paper_benchmark_runs"
MODELS = ROOT / "models" / "paper_benchmark"


@torch.no_grad()
def save_gates(model: TOTALVI, adata, run_dir: Path) -> None:
    """Per-cell RNA-branch weight from the FusionVI gate (1 = RNA only, 0 = protein only)."""
    module = model.module
    module.eval()
    out = []
    for tensors in model._make_data_loader(adata=adata, batch_size=1024, shuffle=False):
        inference_inputs = module._get_inference_input(tensors)
        module.inference(**inference_inputs)
        out.append(module.encoder.last_gate.cpu().numpy().reshape(-1))
    frame = adata.obs[[c for c in ("batch", "tissue", "cell_type") if c in adata.obs]].copy()
    frame["rna_gate"] = np.concatenate(out)
    frame.to_csv(run_dir / "gates.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--arm", dest="model", required=True,
                        help="arm name from `arms:` in config/paper_benchmark.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--smoke-cells", type=int, default=None,
                        help="subsample cells for a quick CPU smoke test; results go to a separate folder")
    args = parser.parse_args()
    with (ROOT / "config" / "paper_benchmark.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    arm = cfg["arms"][args.model]
    label = LABELS.get(args.model, args.model)
    global RUNS, MODELS
    if args.smoke_cells:
        RUNS = ROOT / "results" / "smoke_runs"
        MODELS = ROOT / "models" / "smoke"

    run_name = f"{args.model}__seed{args.seed}"
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    complete = run_dir / "complete.json"
    if complete.exists():
        print(f"Skipping completed paper benchmark {run_name}")
        return

    scvi.settings.seed = args.seed
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    adata = sc.read_h5ad(DATA)
    if args.smoke_cells:
        adata = sc.pp.subsample(adata, n_obs=args.smoke_cells, random_state=args.seed, copy=True)
    TOTALVI.setup_anndata(
        adata,
        batch_key="batch",
        layer="counts",
        protein_expression_obsm_key="protein_counts",
    )
    model = build_model(adata, cfg, arm)

    effective_max_epochs = int(args.max_epochs or cfg["max_epochs"])
    model.train(
        max_epochs=effective_max_epochs,
        lr=float(cfg["learning_rate"]),
        accelerator=args.accelerator,
        devices=1,
        train_size=0.9,
        validation_size=0.1,
        batch_size=int(cfg["batch_size"]),
        early_stopping=True,
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        check_val_every_n_epoch=1,
        reduce_lr_on_plateau=True,
        adversarial_classifier=True,
        # Worker spawning is unreliable in long Windows runs; zero workers is
        # slower but leaves the model, batches and optimization unchanged.
        datasplitter_kwargs={"num_workers": 0, "pin_memory": True},
        enable_progress_bar=True,
    )

    if hasattr(model.module.encoder, "last_gate"):
        save_gates(model, adata, run_dir)

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
    _, predicted_foreground = model.get_normalized_expression(
        target,
        transform_batch=cfg["source_batch"],
        n_samples=int(cfg["posterior_samples"]),
        return_mean=True,
        include_protein_background=False,
        scale_protein=False,
        return_numpy=False,
    )
    predicted_foreground = predicted_foreground.loc[truth.index, truth.columns].clip(lower=0.0)
    np.savez_compressed(
        run_dir / "target_predictions.npz",
        truth=truth.to_numpy(dtype=np.float32),
        prediction=predicted.to_numpy(dtype=np.float32),
        foreground_prediction=predicted_foreground.to_numpy(dtype=np.float32),
        protein_names=np.asarray(truth.columns, dtype=str),
        cell_type=np.asarray(target.obs["cell_type"].astype(str), dtype=str),
    )
    rows = []
    for protein in truth.columns:
        observed = truth[protein].to_numpy()
        estimate = predicted[protein].to_numpy()
        observed_log = np.log1p(observed)
        estimate_log = np.log1p(estimate)
        rows.append(
            {
                "model": label,
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
    config_bytes = (ROOT / "config" / "paper_benchmark.yaml").read_bytes()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None
    result = {
        "model": label,
        "arm": args.model,
        "encoder_variant": arm["encoder"],
        "encoder_hidden": int(arm["hidden"]),
        "gate_mode": arm.get("gate"),
        "modality_dropout": float(arm.get("modality_dropout", 0.0)),
        "seed": args.seed,
        "configured_max_epochs": effective_max_epochs,
        "epochs_completed": int(model.history["elbo_train"].shape[0]),
        "trainable_parameters": n_trainable(model),
        "mean_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].mean()),
        "median_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].median()),
        "environment": {
            "scvi_tools": scvi.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git_commit": git_commit,
    }
    complete.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
