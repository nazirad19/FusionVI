"""Evaluate cross-mouse hidden-marker recovery on the original totalVI data."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ADATA = ROOT / "data" / "processed" / "totalvi_original_sln111.h5ad"
RESULTS = ROOT / "results" / "experiment3_totalvi_original"
FOLDS = RESULTS / "folds"


def rho(y: np.ndarray, p: np.ndarray) -> float:
    return float(spearmanr(y, p).statistic) if np.std(y) and np.std(p) else float("nan")


def gene_vector(adata: ad.AnnData, cell_ids: pd.Index, gene: str) -> np.ndarray:
    """Return one log-normalized raw-count gene vector for the requested cells."""
    values = adata[cell_ids, gene].layers["counts"]
    dense = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.log1p(dense.reshape(-1))


def main() -> None:
    with (ROOT / "config" / "totalvi_original.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    adata = ad.read_h5ad(ADATA)
    proteins = np.log1p(adata.obsm["protein_counts"].astype(float))
    targets = list(cfg["targets"])
    context_names = [name for name in proteins.columns if name not in targets]
    context = proteins[context_names]
    rows = []
    cell_rows = []

    for fold in (0, 1):
        reference = FOLDS / f"totalvi__mouse{fold}__seed{cfg['project_seed']}"
        train_meta = pd.read_csv(reference / "train_metadata.csv", index_col=0)
        test_meta = pd.read_csv(reference / "test_metadata.csv", index_col=0)
        train_ids, test_ids = train_meta.index, test_meta.index
        latents = {
            model: (
                np.load(FOLDS / f"{model}__mouse{fold}__seed{cfg['project_seed']}" / "latent_train.npy"),
                np.load(FOLDS / f"{model}__mouse{fold}__seed{cfg['project_seed']}" / "latent_test.npy"),
            )
            for model in ("totalvi", "fusionvi")
        }
        native = {
            model: pd.read_csv(
                FOLDS / f"{model}__mouse{fold}__seed{cfg['project_seed']}" / "native_predictions.csv",
                index_col=0,
            )
            for model in ("totalvi", "fusionvi")
        }
        for target, proxy in cfg["targets"].items():
            train_rna = gene_vector(adata, train_ids, proxy)
            test_rna = gene_vector(adata, test_ids, proxy)
            y_train = proteins.loc[train_ids, target].to_numpy()
            y_test = proteins.loc[test_ids, target].to_numpy()
            blocks = {
                "rna_proxy": (train_rna[:, None], test_rna[:, None]),
                "protein_context": (context.loc[train_ids].to_numpy(), context.loc[test_ids].to_numpy()),
                "totalvi_xmodal": (
                    np.column_stack([latents["totalvi"][0], context.loc[train_ids], train_rna]),
                    np.column_stack([latents["totalvi"][1], context.loc[test_ids], test_rna]),
                ),
                "fusionvi_xmodal": (
                    np.column_stack([latents["fusionvi"][0], context.loc[train_ids], train_rna]),
                    np.column_stack([latents["fusionvi"][1], context.loc[test_ids], test_rna]),
                ),
            }
            for name, (x_train, x_test) in blocks.items():
                head = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                head.fit(x_train, y_train)
                pred = head.predict(x_test)
                rows.append({"heldout_mouse": fold, "target": target, "model": name,
                             "spearman": rho(y_test, pred), "mae": float(np.mean(np.abs(y_test - pred))),
                             "n_cells": len(y_test)})
                cell_rows.append(pd.DataFrame({"cell_id": test_ids, "heldout_mouse": fold,
                                               "target": target, "model": name,
                                               "true": y_test, "prediction": pred}))
            for name in ("totalvi", "fusionvi"):
                pred = native[name].loc[test_ids, f"pred__{target}"].to_numpy()
                rows.append({"heldout_mouse": fold, "target": target, "model": name,
                             "spearman": rho(y_test, pred), "mae": np.nan, "n_cells": len(y_test)})
                cell_rows.append(pd.DataFrame({"cell_id": test_ids, "heldout_mouse": fold,
                                               "target": target, "model": name,
                                               "true": y_test, "prediction": pred}))

    metrics = pd.DataFrame(rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    pooled = []
    for (target, model), frame in cells.groupby(["target", "model"]):
        comparable = model not in {"totalvi", "fusionvi"}
        pooled.append({"target": target, "model": model,
                       "spearman": rho(frame["true"].to_numpy(), frame["prediction"].to_numpy()),
                       "mae": float(np.mean(np.abs(frame["true"] - frame["prediction"]))) if comparable else np.nan,
                       "n_cells": len(frame)})
    summary = pd.DataFrame(pooled)
    fold_summary = (
        metrics.groupby(["target", "model"], as_index=False)
        .agg(
            mean_fold_spearman=("spearman", "mean"),
            min_fold_spearman=("spearman", "min"),
            max_fold_spearman=("spearman", "max"),
            mean_fold_mae=("mae", "mean"),
            n_folds=("heldout_mouse", "nunique"),
        )
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS / "cross_mouse_metrics.csv", index=False)
    summary.to_csv(RESULTS / "summary_metrics.csv", index=False)
    fold_summary.to_csv(RESULTS / "fold_summary_metrics.csv", index=False)
    print(summary.pivot(index="target", columns="model", values="spearman").round(3), flush=True)


if __name__ == "__main__":
    main()
