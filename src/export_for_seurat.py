"""Export a benchmark object for the Seurat v3 transfer baseline (R/seurat_transfer.R).

Writes data/seurat/<benchmark>/:
  rna_counts.mtx, genes.txt, cells.txt   RNA counts, genes x cells (all cells)
  adt_counts.mtx, proteins.txt           source-batch protein counts, proteins x source cells
  source_cells.txt, target_cells.txt     cell barcodes per role
  source_folds.csv                       5-fold assignment of source cells (cell, fold),
                                         stratified by cell type, for cross-fitted D1 predictions

Only source proteins are exported; target protein truth never leaves Python.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from benchmarks import data_path, load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="paper")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.benchmark)
    adata = sc.read_h5ad(data_path(cfg))
    out = ROOT / "data" / "seurat" / args.benchmark
    out.mkdir(parents=True, exist_ok=True)

    batch = adata.obs["batch"].astype(str).to_numpy()
    source, target = batch == cfg["source_batch"], batch == cfg["target_batch"]
    cells = np.asarray(adata.obs_names, dtype=str)

    scipy.io.mmwrite(out / "rna_counts.mtx", sp.csr_matrix(adata.layers["counts"]).T.tocoo().astype(np.float64))
    pd.Series(adata.var_names).to_csv(out / "genes.txt", index=False, header=False)
    pd.Series(cells).to_csv(out / "cells.txt", index=False, header=False)

    adt = adata.obsm["protein_counts"].loc[source]            # source proteins only (== truth for source cells)
    scipy.io.mmwrite(out / "adt_counts.mtx", sp.csr_matrix(adt.to_numpy(dtype=np.float64)).T.tocoo())
    pd.Series(adt.columns.astype(str)).to_csv(out / "proteins.txt", index=False, header=False)
    pd.Series(cells[source]).to_csv(out / "source_cells.txt", index=False, header=False)
    pd.Series(cells[target]).to_csv(out / "target_cells.txt", index=False, header=False)

    strata = pd.Series(adata.obs.loc[source, "cell_type"].astype(str).to_numpy())
    strata = np.where(strata.map(strata.value_counts()) >= args.folds, strata, "rare")
    if 0 < (strata == "rare").sum() < args.folds:
        nonrare = pd.Series(strata[strata != "rare"])
        strata[strata == "rare"] = nonrare.mode().iloc[0] if len(nonrare) else "all"
    folds = np.empty(source.sum(), dtype=int)
    for k, (_, test) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=2026).split(strata, strata)):
        folds[test] = k + 1
    pd.DataFrame({"cell": cells[source], "fold": folds}).to_csv(out / "source_folds.csv", index=False)
    print(f"Wrote {out} ({source.sum()} source, {target.sum()} target cells, {adt.shape[1]} proteins)")


if __name__ == "__main__":
    main()
