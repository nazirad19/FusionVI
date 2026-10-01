"""Convert HCA RDS matrices into a compact AnnData object for FusionVI."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rdata
import scanpy as sc
import yaml
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_config() -> dict:
    with (ROOT / "config" / "default.yaml").open() as handle:
        return yaml.safe_load(handle)


def read_r_sparse(path: Path) -> tuple[sparse.csc_matrix, list[str], list[str]]:
    parsed = rdata.parser.parse_file(path)
    obj = rdata.conversion.convert(parsed)
    slots = vars(obj)
    matrix = sparse.csc_matrix(
        (slots["x"], slots["i"], slots["p"]),
        shape=tuple(int(v) for v in slots["Dim"]),
    )
    rows = [str(v) for v in slots["Dimnames"][0]]
    cols = [str(v) for v in slots["Dimnames"][1]]
    return matrix, rows, cols


def main() -> None:
    cfg = load_config()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "lawlor_pbmc_citeseq.h5ad"

    annotation = pd.read_csv(RAW / "CZI.PBMC.cell.annotations.csv", low_memory=False)
    annotation = annotation.rename(columns={"Unnamed: 0": "cell_id"})
    annotation = annotation.dropna(subset=["Celltype_Annotation"]).copy()
    annotation = annotation[
        annotation["HTO_Classification"].isin(["Baseline", "LPS", "CD3_CD28"])
    ].copy()
    annotation = annotation.drop_duplicates("cell_id").set_index("cell_id")

    print("Reading RNA sparse matrix", flush=True)
    rna, genes, all_cells = read_r_sparse(RAW / "CZI.PBMC.RNA.matrix.Rds")
    positions = {cell: idx for idx, cell in enumerate(all_cells)}
    missing = annotation.index.difference(positions)
    if len(missing):
        raise RuntimeError(f"{len(missing)} annotated cells missing from RNA matrix")
    selected_columns = np.asarray([positions[cell] for cell in annotation.index], dtype=int)
    counts = rna[:, selected_columns].T.tocsr().astype(np.float32)
    del rna, positions, all_cells

    print("Reading ADT matrix", flush=True)
    adt = rdata.conversion.convert(
        rdata.parser.parse_file(RAW / "CZI.PBMC.ADT.matrix.Rds")
    )
    protein_rows = [
        name
        for name in adt.index.astype(str)
        if not name.startswith("control_")
        and name not in {"bad_struct", "no_match", "total_reads"}
    ]
    if len(protein_rows) != 39:
        raise RuntimeError(f"Expected 39 biological proteins, found {len(protein_rows)}")
    # HDF5 keys cannot contain forward slashes, so retain underscores in
    # compound antibody labels such as CD127_IL7Ra.
    protein_names = [name.rsplit("-", 1)[0] for name in protein_rows]
    protein_counts = adt.loc[protein_rows, annotation.index].T
    protein_counts.index = annotation.index
    protein_counts.columns = protein_names
    protein_counts = protein_counts.astype(np.float32)
    del adt

    obs = annotation[
        ["HTO_Classification", "Run_Identifier", "Donor_of_Origin", "Celltype_Annotation"]
    ].rename(
        columns={
            "HTO_Classification": "condition",
            "Run_Identifier": "lane",
            "Donor_of_Origin": "donor",
            "Celltype_Annotation": "cell_type",
        }
    )
    obs = obs.astype(str)
    adata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["protein_counts"] = protein_counts

    # Extremely sparse genes make the Seurat-v3 LOESS fit singular in this
    # relatively small dataset. Removing genes seen in fewer than ten cells is
    # a count-preserving QC step and leaves all prespecified activation genes
    # that contain usable information.
    sc.pp.filter_genes(adata, min_cells=10)
    adata.layers["counts"] = adata.X.copy()
    # The Cell Ranger dispersion procedure is numerically stable for this
    # shallow 10x-v2 dataset. Compute it on a temporary log-normalized view,
    # while leaving the model input as untouched integer counts.
    hvg_view = adata.copy()
    sc.pp.normalize_total(hvg_view, target_sum=1e4)
    sc.pp.log1p(hvg_view)
    sc.pp.highly_variable_genes(
        hvg_view,
        n_top_genes=int(cfg["n_hvg"]),
        flavor="cell_ranger",
        batch_key="lane",
        subset=False,
    )
    requested = set(cfg["activation_genes"])
    force = np.asarray([gene in requested for gene in adata.var_names])
    keep = np.asarray(hvg_view.var["highly_variable"], dtype=bool) | force
    del hvg_view
    adata = adata[:, keep].copy()
    adata.layers["counts"] = adata.X.copy()
    adata.uns["heldout_proteins"] = list(cfg["heldout_proteins"])
    adata.uns["source"] = {
        "study": "Lawlor et al., Frontiers in Immunology 2021",
        "hca_project": "efea6426-510a-4b60-9a19-277e52bfa815",
    }
    adata.write_h5ad(output, compression="gzip")

    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "proteins": int(protein_counts.shape[1]),
        "donors": int(obs["donor"].nunique()),
        "lanes": int(obs["lane"].nunique()),
        "conditions": obs["condition"].value_counts().to_dict(),
        "cell_types": obs["cell_type"].value_counts().to_dict(),
        "heldout_proteins": list(cfg["heldout_proteins"]),
    }
    with (RESULTS / "prepared_data_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
