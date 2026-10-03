"""Prepare the missing-protein benchmark used in totalVI Figure 3f-h."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "totalvi_original" / "spleen_lymph_111.h5ad"
OUT = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
SUMMARY = ROOT / "results" / "paper_benchmark_data_summary.json"


def main() -> None:
    source = ad.read_h5ad(RAW)
    keep_genes = source.var["hvg_encode"].astype(bool).to_numpy()
    adata = source[:, keep_genes].copy()

    protein_names = np.asarray(source.uns["protein_names"], dtype=str)
    keep_proteins = ~np.char.startswith(protein_names, "HTO")
    protein_names = protein_names[keep_proteins]
    truth = pd.DataFrame(
        np.asarray(source.obsm["protein_expression"][:, keep_proteins], dtype=np.float32),
        index=adata.obs_names,
        columns=protein_names,
    )

    original_batch = source.obs.loc[adata.obs_names, "batch_indices"].astype(int)
    batch = pd.Categorical(
        np.where(original_batch.to_numpy() == 0, "SLN111-D1", "SLN111-D2"),
        categories=["SLN111-D1", "SLN111-D2"],
    )
    masked = truth.copy()
    masked.loc[np.asarray(batch) == "SLN111-D2", :] = 0.0

    adata.layers["counts"] = adata.X.copy()
    adata.obsm["protein_truth"] = truth
    adata.obsm["protein_counts"] = masked
    adata.obs = pd.DataFrame(
        {
            "batch": batch,
            "mouse": pd.Categorical(np.where(original_batch.to_numpy() == 0, "mouse0", "mouse1")),
            "tissue": source.obs.loc[adata.obs_names, "hash_id"].astype(str),
            "cell_type": source.obs.loc[adata.obs_names, "cell_types"].astype(str),
        },
        index=adata.obs_names,
    )
    adata.uns["benchmark"] = {
        "paper": "Gayoso et al., Nature Methods 2021",
        "figure": "Figure 3f-h missing-protein imputation",
        "source_batch": "SLN111-D1 proteins observed",
        "target_batch": "SLN111-D2 all 110 proteins hidden during training",
        "metric": "root mean squared log error",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT, compression="gzip")
    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "proteins": int(truth.shape[1]),
        "source_cells": int((adata.obs["batch"] == "SLN111-D1").sum()),
        "target_cells": int((adata.obs["batch"] == "SLN111-D2").sum()),
        "target_observed_protein_entries_used_for_training": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
