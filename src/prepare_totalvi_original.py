"""Prepare the original totalVI SLN111 dataset for cross-mouse transfer."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "totalvi_original" / "spleen_lymph_111.h5ad"
OUT = ROOT / "data" / "processed" / "totalvi_original_sln111.h5ad"
RESULTS = ROOT / "results" / "experiment3_totalvi_original"


def main() -> None:
    with (ROOT / "config" / "totalvi_original.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    source = ad.read_h5ad(RAW)
    keep_cells = source.obs["hash_id"].astype(str) != "Negative"
    forced_genes = set(cfg["targets"].values())
    keep_genes = source.var["highly_variable"].astype(bool) | source.var_names.isin(forced_genes)
    adata = source[keep_cells, keep_genes].copy()

    protein_names = list(map(str, source.uns["protein_names"]))
    proteins = pd.DataFrame(
        source.obsm["protein_expression"][keep_cells.to_numpy()],
        index=adata.obs_names,
        columns=protein_names,
    )
    proteins = proteins.loc[:, ~proteins.columns.str.startswith("HTO_")].astype("float32")
    adata.obsm["protein_counts"] = proteins
    adata.layers["counts"] = adata.X.copy()
    adata.obs = pd.DataFrame(
        {
            "mouse": pd.Categorical("mouse" + source.obs.loc[adata.obs_names, "batch_indices"].astype(str)),
            "tissue": source.obs.loc[adata.obs_names, "hash_id"].astype(str),
            "cell_type": source.obs.loc[adata.obs_names, "cell_types"].astype(str),
        },
        index=adata.obs_names,
    )
    adata.uns["heldout_proteins"] = list(cfg["targets"])
    adata.uns["source"] = {
        "study": "Gayoso et al., Nature Methods 2021",
        "accession": "GSE150599",
        "file": "spleen_lymph_111.h5ad",
        "design": "two biological replicate mice, spleen and lymph node",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT, compression="gzip")
    mouse_counts = adata.obs["mouse"].value_counts()
    tissue_counts = adata.obs.groupby(["mouse", "tissue"], observed=True).size()
    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "proteins": int(proteins.shape[1]),
        "mice": {str(mouse): int(count) for mouse, count in mouse_counts.items()},
        "tissues": {
            f"{mouse}__{tissue}": int(count)
            for (mouse, tissue), count in tissue_counts.items()
        },
        "masked_targets": cfg["targets"],
    }
    (RESULTS / "prepared_data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
