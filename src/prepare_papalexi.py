"""Create a compact Papalexi ECCITE-seq AnnData from GEO count tables.

The released RNA matrix is a dense TSV with genes in rows. To keep memory
bounded, it is scanned twice: the first pass computes gene-level dispersion,
and the second stores only the selected highly variable genes plus CD274.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "papalexi"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "experiment2_papalexi"
RNA_FILE = RAW / "GSM4633614_ECCITE_cDNA_counts.tsv.gz"
ADT_FILE = RAW / "GSM4633615_ECCITE_ADT_counts.tsv.gz"
META_FILE = RAW / "GSE153056_ECCITE_metadata.tsv.gz"


def load_config() -> dict:
    with (ROOT / "config" / "papalexi.yaml").open() as handle:
        return yaml.safe_load(handle)


def parse_count_line(line: str) -> tuple[str, np.ndarray]:
    label, values = line.rstrip("\r\n").split("\t", 1)
    return label.strip('"'), np.fromstring(values, sep="\t", dtype=np.int32)


def select_hvgs(n_top: int, forced: set[str]) -> tuple[list[str], list[str]]:
    with gzip.open(RNA_FILE, "rt", newline="") as handle:
        header = next(csv.reader([handle.readline()], delimiter="\t"))[1:]
        stats = []
        for number, line in enumerate(handle, 1):
            gene, values = parse_count_line(line)
            if len(values) != len(header):
                raise RuntimeError(f"Malformed RNA row {number}: {gene}")
            nonzero = int(np.count_nonzero(values))
            if nonzero >= 10 or gene in forced:
                mean = float(values.mean())
                variance = float(values.var())
                stats.append((gene, nonzero, mean, variance))
            if number % 5000 == 0:
                print(f"HVG scan: {number:,} genes", flush=True)

    frame = pd.DataFrame(stats, columns=["gene", "nonzero", "mean", "variance"])
    frame["log_mean"] = np.log1p(frame["mean"])
    frame["log_dispersion"] = np.log1p(frame["variance"] / frame["mean"].clip(lower=1e-8))
    frame["mean_bin"] = pd.qcut(frame["log_mean"], q=20, duplicates="drop")
    grouped = frame.groupby("mean_bin", observed=True)["log_dispersion"]
    frame["dispersion_z"] = grouped.transform(
        lambda values: (values - values.mean()) / max(float(values.std(ddof=0)), 1e-8)
    )
    selected = set(frame.nlargest(n_top, "dispersion_z")["gene"])
    selected.update(forced)
    ordered = frame.loc[frame["gene"].isin(selected), "gene"].tolist()
    return header, ordered


def load_selected_rna(cell_names: list[str], selected: list[str]) -> sparse.csr_matrix:
    selected_index = {gene: row for row, gene in enumerate(selected)}
    dense = np.zeros((len(selected), len(cell_names)), dtype=np.float32)
    found: set[str] = set()
    with gzip.open(RNA_FILE, "rt", newline="") as handle:
        header = next(csv.reader([handle.readline()], delimiter="\t"))[1:]
        if header != cell_names:
            raise RuntimeError("RNA cell order changed between HVG scan and matrix loading")
        for number, line in enumerate(handle, 1):
            gene, values = parse_count_line(line)
            if gene in selected_index:
                dense[selected_index[gene], :] = values
                found.add(gene)
            if number % 5000 == 0:
                print(f"RNA load: {number:,} genes", flush=True)
    missing = set(selected).difference(found)
    if missing:
        raise RuntimeError(f"Selected genes absent from RNA matrix: {sorted(missing)}")
    matrix = sparse.csr_matrix(dense.T)
    del dense
    return matrix


def assign_outer_folds(obs: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    targets = sorted(set(obs["gene"]).difference({"NT"}))
    rng = np.random.default_rng(seed)
    rng.shuffle(targets)
    mapping = {target: index % n_folds for index, target in enumerate(targets)}
    folds = obs["gene"].map(mapping).astype("Int64")
    # Partition negative controls within replicate. Each outer fold therefore
    # has an untouched NT reference while the other four fifths remain usable
    # for fitting the readout and the latent representation.
    for replicate, indices in obs.index[obs["gene"] == "NT"].to_series().groupby(
        obs.loc[obs["gene"] == "NT", "replicate"]
    ):
        cells = indices.to_numpy().copy()
        rng.shuffle(cells)
        folds.loc[cells] = np.arange(len(cells)) % n_folds
    return folds.astype(int)


def main() -> None:
    cfg = load_config()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(META_FILE, sep="\t", index_col=0)
    metadata.index = metadata.index.astype(str)
    proteins = pd.read_csv(ADT_FILE, sep="\t", index_col=0).T
    proteins.index = proteins.index.astype(str)
    proteins.columns = proteins.columns.astype(str)

    header, selected = select_hvgs(int(cfg["n_hvg"]), {str(cfg["rna_proxy"])})
    if set(header) != set(metadata.index) or set(header) != set(proteins.index):
        raise RuntimeError("RNA, protein and metadata releases do not contain identical cells")
    metadata = metadata.loc[header].copy()
    proteins = proteins.loc[header].copy()
    counts = load_selected_rna(header, selected)

    obs = metadata[
        ["orig.ident", "gene", "guide_ID", "replicate", "crispr", "Phase", "nCount_RNA"]
    ].rename(columns={"orig.ident": "lane", "guide_ID": "guide"})
    obs = obs.astype({"lane": str, "gene": str, "guide": str, "replicate": str, "crispr": str, "Phase": str})
    obs["outer_fold"] = assign_outer_folds(obs, int(cfg["n_outer_folds"]), int(cfg["project_seed"]))

    adata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(selected, name="gene")),
    )
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["protein_counts"] = proteins.astype(np.float32)
    adata.uns["heldout_proteins"] = [str(cfg["heldout_protein"])]
    adata.uns["source"] = {
        "study": "Papalexi et al., Nature Genetics 2021",
        "geo": "GSE153056",
        "design": "ECCITE-seq CRISPR screen with IFN-gamma stimulation",
    }
    output = PROCESSED / "papalexi_eccite.h5ad"
    adata.write_h5ad(output, compression="gzip")

    assignment = (
        obs.loc[obs["gene"] != "NT", ["gene", "outer_fold"]]
        .drop_duplicates()
        .sort_values(["outer_fold", "gene"])
    )
    assignment.to_csv(RESULTS / "outer_fold_assignment.csv", index=False)
    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "proteins": list(proteins.columns),
        "perturbation_targets": int(obs.loc[obs["gene"] != "NT", "gene"].nunique()),
        "negative_control_cells": int((obs["gene"] == "NT").sum()),
        "replicates": obs["replicate"].value_counts().to_dict(),
        "fold_targets": assignment.groupby("outer_fold")["gene"].apply(list).to_dict(),
    }
    (RESULTS / "prepared_data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
