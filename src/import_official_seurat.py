"""Import the totalVI authors' published SLN Seurat predictions.

The official repository stores the paper's Seurat v3 target prediction as a
proteins-by-cells CSV. This script downloads it, verifies complete alignment to
our paper benchmark, and saves a cells-by-proteins file for raw scoring. It
cannot provide source cross-fitted predictions, so it must not be used to fit
the D1-only calibration.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from benchmarks import data_path, load_config

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://raw.githubusercontent.com/YosefLab/totalVI_reproducibility/"
    "master/harmonization/seurat_harmo_results/imputed_sln.csv"
)


def main() -> None:
    cfg = load_config("paper")
    adata = sc.read_h5ad(data_path(cfg))
    target = adata.obs["batch"].astype(str).eq(cfg["target_batch"]).to_numpy()
    cells = np.asarray(adata.obs_names[target], dtype=str)
    proteins = list(map(str, adata.obsm["protein_truth"].columns))

    with urllib.request.urlopen(URL, timeout=120) as response:
        raw = response.read()
    official = pd.read_csv(io.BytesIO(raw), index_col=0)
    seurat_to_original = {p.replace("_", "-"): p for p in proteins}
    official.index = [seurat_to_original.get(str(x), str(x)) for x in official.index]
    official.columns = [str(x).replace(".", "-") for x in official.columns]
    missing_cells = sorted(set(cells) - set(official.columns))
    missing_proteins = sorted(set(proteins) - set(official.index))
    if missing_cells or missing_proteins:
        raise SystemExit(
            f"Official result does not align: {len(missing_cells)} cells and "
            f"{len(missing_proteins)} proteins missing"
        )

    aligned = official.loc[proteins, cells].T
    out = ROOT / "results" / "seurat" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out / "target_imputed_official.csv")
    (out / "official_provenance.json").write_text(json.dumps({
        "url": URL,
        "source_shape_proteins_by_cells": list(official.shape),
        "saved_shape_cells_by_proteins": list(aligned.shape),
        "use": "raw target scoring only; no source cross-fit predictions are available",
    }, indent=2))
    print(f"Imported official Seurat predictions: {aligned.shape} -> {out}")


if __name__ == "__main__":
    main()
