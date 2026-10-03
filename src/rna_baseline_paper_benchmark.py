"""Leakage-safe RNA-only ridge baseline for the complete-panel benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
OUT = ROOT / "results"


def log_normalize(counts) -> sp.csr_matrix:
    x = sp.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(x.sum(axis=1)).reshape(-1)
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    x = sp.diags(scale) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


def safe_rho(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "paper_benchmark.yaml").read_text())
    adata = sc.read_h5ad(DATA)
    source = adata.obs["batch"].astype(str).to_numpy() == cfg["source_batch"]
    target = adata.obs["batch"].astype(str).to_numpy() == cfg["target_batch"]
    x = log_normalize(adata.layers["counts"])
    truth = adata.obsm["protein_truth"].astype(float)
    y_log = np.log1p(truth.to_numpy(dtype=np.float32))

    # A fixed source-fitted SVD makes the multi-output ridge baseline fast and
    # prevents target-batch expression from influencing representation fitting.
    svd = TruncatedSVD(n_components=128, random_state=2026)
    x_source = svd.fit_transform(x[source])
    x_target = svd.transform(x[target])
    source_idx = np.arange(x_source.shape[0])
    train_idx, valid_idx = train_test_split(
        source_idx,
        test_size=0.2,
        random_state=2026,
        stratify=adata.obs.loc[source, "cell_type"].astype(str),
    )
    alphas = [0.1, 1.0, 10.0, 100.0]
    tuning = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="cholesky")
        model.fit(x_source[train_idx], y_log[source][train_idx])
        pred = np.maximum(model.predict(x_source[valid_idx]), 0.0)
        tuning.append({"alpha": alpha, "validation_rmsle": float(np.sqrt(np.mean((pred - y_log[source][valid_idx]) ** 2)))})
    best_alpha = min(tuning, key=lambda row: row["validation_rmsle"])["alpha"]

    model = Ridge(alpha=best_alpha, solver="cholesky")
    model.fit(x_source, y_log[source])
    pred_log = np.maximum(model.predict(x_target), 0.0)
    observed_log = y_log[target]
    protein_names = np.asarray(truth.columns, dtype=str)
    rows = []
    for j, protein in enumerate(protein_names):
        rows.append(
            {
                "model": "RNA ridge",
                "protein": protein,
                "rmsle": float(np.sqrt(np.mean((pred_log[:, j] - observed_log[:, j]) ** 2))),
                "spearman": safe_rho(observed_log[:, j], pred_log[:, j]),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "rna_ridge_protein_metrics.csv", index=False)

    cell_types = adata.obs.loc[target, "cell_type"].astype(str).to_numpy()
    within = []
    for cell_type in np.unique(cell_types):
        mask = cell_types == cell_type
        if mask.sum() < 30:
            continue
        for j, protein in enumerate(protein_names):
            within.append(
                {
                    "model": "RNA ridge",
                    "cell_type": cell_type,
                    "protein": protein,
                    "n_cells": int(mask.sum()),
                    "spearman": safe_rho(observed_log[mask, j], pred_log[mask, j]),
                }
            )
    pd.DataFrame(within).to_csv(OUT / "rna_ridge_within_celltype.csv", index=False)

    marker_tokens = {"CD4": "ADT_CD4_", "CD8": "ADT_CD8a_", "CD19": "ADT_CD19_"}
    auc_rows = []
    for marker, token in marker_tokens.items():
        matches = [i for i, name in enumerate(protein_names) if token in name]
        if len(matches) != 1:
            continue
        j = matches[0]
        source_values = y_log[source, j].reshape(-1, 1)
        mixture = GaussianMixture(n_components=2, random_state=2026).fit(source_values)
        means = np.sort(mixture.means_.reshape(-1))
        threshold = float(means.mean())
        labels = observed_log[:, j] > threshold
        auc_rows.append(
            {
                "model": "RNA ridge",
                "marker": marker,
                "protein": protein_names[j],
                "source_log1p_threshold": threshold,
                "positive_fraction": float(labels.mean()),
                "auroc": float(roc_auc_score(labels, pred_log[:, j])),
            }
        )
    pd.DataFrame(auc_rows).to_csv(OUT / "rna_ridge_marker_auroc.csv", index=False)

    summary = {
        "model": "RNA-only 128-component SVD plus multi-output ridge",
        "input": "source-fitted SVD of log1p library-size-normalized RNA",
        "best_alpha_selected_on_source_only": best_alpha,
        "alpha_tuning": tuning,
        "mean_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].mean()),
        "median_protein_rmsle": float(pd.DataFrame(rows)["rmsle"].median()),
        "mean_within_celltype_spearman": float(pd.DataFrame(within)["spearman"].mean()),
        "marker_auroc": {r["marker"]: r["auroc"] for r in auc_rows},
    }
    (OUT / "rna_ridge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
