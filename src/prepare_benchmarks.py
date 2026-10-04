"""Build the development and partial-panel benchmark objects.

Outputs (data/processed/):
  dev_sln111_random.h5ad   SLN111-D1 only; a stratified 20% of D1 cells become
                           DEV-HOLDOUT with the whole panel hidden. The real
                           target batch (SLN111-D2) is not included at all.
  dev_sln111_tissue.h5ad   SLN111-D1 only; D1 lymph-node cells become
                           DEV-HOLDOUT (adds a composition shift).
  sln206_partial_panel.h5ad
                           SLN206 (207 proteins, two mice). SLN206-D1 keeps
                           every protein. SLN206-D2 keeps only the 110 proteins
                           of the SLN111 panel; the other 97 are hidden and are
                           the evaluation targets (uns["eval_proteins"]).
  dev_sln206_partial.h5ad  SLN206-D1 only; a stratified 20% become DEV-HOLDOUT
                           with the same 110-protein panel. SLN206-D2 excluded.

Every object stores the hidden values only in obsm["protein_truth"]; the
training matrix obsm["protein_counts"] has them set to zero, which scvi-tools
treats as unmeasured for that batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "totalvi_original"
OUT = ROOT / "data" / "processed"
SEED = 2026


def load(name: str, batch_labels: tuple[str, str], drop_hash=("Negative", "Doublet")) -> ad.AnnData:
    source = ad.read_h5ad(RAW / name)
    keep_cells = ~source.obs["hash_id"].astype(str).isin(drop_hash).to_numpy() if drop_hash else np.ones(source.n_obs, bool)
    keep_genes = source.var["hvg_encode"].astype(bool).to_numpy()
    adata = source[keep_cells, keep_genes].copy()
    names = np.asarray(source.uns["protein_names"], dtype=str)
    keep_p = ~np.char.startswith(names, "HTO")
    truth = pd.DataFrame(np.asarray(source.obsm["protein_expression"][keep_cells][:, keep_p], dtype=np.float32),
                         index=adata.obs_names, columns=names[keep_p])
    batch = np.where(adata.obs["batch_indices"].astype(int).to_numpy() == 0, *batch_labels)
    adata.obs = pd.DataFrame({
        "batch": batch,
        "mouse": np.where(batch == batch_labels[0], "mouse0", "mouse1"),
        "tissue": adata.obs["hash_id"].astype(str).to_numpy(),
        "cell_type": adata.obs["cell_types"].astype(str).to_numpy(),
    }, index=adata.obs_names)
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["protein_truth"] = truth
    return adata


def finalize(adata: ad.AnnData, target_label: str, hidden: list[str], source_label: str, path: Path,
             description: str) -> dict:
    truth = adata.obsm["protein_truth"]
    counts = truth.copy()
    is_target = (adata.obs["batch"] == target_label).to_numpy()
    counts.loc[is_target, hidden] = 0.0
    adata.obsm["protein_counts"] = counts
    adata.obs["batch"] = pd.Categorical(adata.obs["batch"], categories=[source_label, target_label])
    adata.uns["eval_proteins"] = list(hidden)
    adata.uns["benchmark_description"] = description
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")
    return {
        "file": path.name,
        "cells": int(adata.n_obs),
        "source_cells": int((~is_target).sum()),
        "target_cells": int(is_target.sum()),
        "genes": int(adata.n_vars),
        "proteins_total": int(truth.shape[1]),
        "proteins_hidden_in_target": len(hidden),
        "target_hidden_entries_in_training_matrix": float(counts.loc[is_target, hidden].to_numpy().sum()),
    }


def holdout_random(adata: ad.AnnData, frac: float) -> np.ndarray:
    idx = np.arange(adata.n_obs)
    strata = adata.obs["cell_type"].astype(str).to_numpy()
    counts = pd.Series(strata).value_counts()
    strata = np.where(pd.Series(strata).map(counts).to_numpy() >= 5, strata, "rare")
    _, held = train_test_split(idx, test_size=frac, random_state=SEED, stratify=strata)
    mask = np.zeros(adata.n_obs, bool)
    mask[held] = True
    return mask


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "paper_benchmark.yaml").read_text())
    frac = float(cfg.get("dev_holdout_fraction", 0.2))
    summary = []

    # ---------------- SLN111 development splits (D1 only; real D2 never loaded into the object)
    sln111 = load("spleen_lymph_111.h5ad", ("SLN111-D1", "SLN111-D2"), drop_hash=())
    d1 = sln111[sln111.obs["batch"] == "SLN111-D1"].copy()
    all_111 = list(map(str, d1.obsm["protein_truth"].columns))
    for split, mask in (("random", holdout_random(d1, frac)),
                        ("tissue", (d1.obs["tissue"] == "Lymph Node").to_numpy())):
        obj = d1.copy()
        obj.obs["batch"] = np.where(mask, "DEV-HOLDOUT", "SLN111-D1")
        summary.append(finalize(obj, "DEV-HOLDOUT", all_111, "SLN111-D1", OUT / f"dev_sln111_{split}.h5ad",
                                f"SLN111-D1 only; {split} holdout with the full panel hidden"))

    # ---------------- SLN206 partial panel
    sln206 = load("spleen_lymph_206.h5ad", ("SLN206-D1", "SLN206-D2"))
    names_206 = list(map(str, sln206.obsm["protein_truth"].columns))
    shared = set(all_111)
    hidden = [p for p in names_206 if p not in shared]
    summary.append(finalize(sln206.copy(), "SLN206-D2", hidden, "SLN206-D1", OUT / "sln206_partial_panel.h5ad",
                            "SLN206-D2 observes the 110 SLN111-panel proteins; the other 97 are imputed"))

    d1_206 = sln206[sln206.obs["batch"] == "SLN206-D1"].copy()
    mask = holdout_random(d1_206, frac)
    d1_206.obs["batch"] = np.where(mask, "DEV-HOLDOUT", "SLN206-D1")
    summary.append(finalize(d1_206, "DEV-HOLDOUT", hidden, "SLN206-D1", OUT / "dev_sln206_partial.h5ad",
                            "SLN206-D1 only; random holdout observes the 110-protein panel"))

    (ROOT / "results" / "benchmark_objects_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
