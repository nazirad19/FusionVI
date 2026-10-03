"""Legacy native-decoder vs FusionVI-X cross-mouse comparison.

This output is retained for reproducibility, but it is not a like-for-like
model contrast. Use ``baselines_cross_mouse.py`` for native, X-readout and
latent-free comparisons.
"""

from __future__ import annotations

import json
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
RESULTS = ROOT / "results"
FOLDS = RESULTS / "final_folds"


def rho(y: np.ndarray, prediction: np.ndarray) -> float:
    """Spearman correlation with a defined result for non-constant vectors."""
    if np.std(y) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(spearmanr(y, prediction).statistic)


def gene_vector(adata: ad.AnnData, cell_ids: pd.Index, gene: str) -> np.ndarray:
    """Return one log1p raw-count gene vector for the requested cells."""
    values = adata[cell_ids, gene].layers["counts"]
    dense = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.log1p(dense.reshape(-1))


def main() -> None:
    with (ROOT / "config" / "experiment.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    adata = ad.read_h5ad(ADATA)
    proteins = np.log1p(adata.obsm["protein_counts"].astype(float))
    targets = list(cfg["targets"])
    context = proteins[[name for name in proteins.columns if name not in targets]]
    rows: list[dict] = []
    pooled_predictions: list[pd.DataFrame] = []

    for fold in (0, 1):
        totalvi_dir = FOLDS / f"totalvi__mouse{fold}__seed{cfg['project_seed']}"
        fusionvi_dir = FOLDS / f"fusionvi__mouse{fold}__seed{cfg['project_seed']}"
        train_meta = pd.read_csv(totalvi_dir / "train_metadata.csv", index_col=0)
        test_meta = pd.read_csv(totalvi_dir / "test_metadata.csv", index_col=0)
        train_ids, test_ids = train_meta.index, test_meta.index
        fusion_latent_train = np.load(fusionvi_dir / "latent_train.npy")
        fusion_latent_test = np.load(fusionvi_dir / "latent_test.npy")
        totalvi_native = pd.read_csv(totalvi_dir / "native_predictions.csv", index_col=0)

        for target, proxy in cfg["targets"].items():
            y_train = proteins.loc[train_ids, target].to_numpy()
            y_test = proteins.loc[test_ids, target].to_numpy()

            totalvi_prediction = totalvi_native.loc[test_ids, f"pred__{target}"].to_numpy()
            rows.append({
                "heldout_mouse": fold,
                "target": target,
                "model": "totalVI",
                "spearman": rho(y_test, totalvi_prediction),
                "n_cells": len(y_test),
            })
            pooled_predictions.append(pd.DataFrame({
                "heldout_mouse": fold,
                "target": target,
                "model": "totalVI",
                "true": y_test,
                "prediction": totalvi_prediction,
            }))

            train_rna = gene_vector(adata, train_ids, proxy)
            test_rna = gene_vector(adata, test_ids, proxy)
            x_train = np.column_stack([
                fusion_latent_train,
                context.loc[train_ids].to_numpy(),
                train_rna,
            ])
            x_test = np.column_stack([
                fusion_latent_test,
                context.loc[test_ids].to_numpy(),
                test_rna,
            ])
            head = make_pipeline(StandardScaler(), Ridge(alpha=float(cfg["ridge_alpha"])))
            head.fit(x_train, y_train)
            fusion_prediction = head.predict(x_test)
            rows.append({
                "heldout_mouse": fold,
                "target": target,
                "model": "FusionVI-X",
                "spearman": rho(y_test, fusion_prediction),
                "n_cells": len(y_test),
            })
            pooled_predictions.append(pd.DataFrame({
                "heldout_mouse": fold,
                "target": target,
                "model": "FusionVI-X",
                "true": y_test,
                "prediction": fusion_prediction,
            }))

    metrics = pd.DataFrame(rows)
    fold_summary = (
        metrics.groupby(["target", "model"], as_index=False)
        .agg(
            mean_fold_spearman=("spearman", "mean"),
            min_fold_spearman=("spearman", "min"),
            max_fold_spearman=("spearman", "max"),
            n_folds=("heldout_mouse", "nunique"),
        )
    )
    predictions = pd.concat(pooled_predictions, ignore_index=True)
    pooled_rows = []
    for (target, model), frame in predictions.groupby(["target", "model"]):
        pooled_rows.append({
            "target": target,
            "model": model,
            "pooled_spearman": rho(frame["true"].to_numpy(), frame["prediction"].to_numpy()),
            "n_cells": len(frame),
        })
    pooled = pd.DataFrame(pooled_rows)
    mean_by_model = fold_summary.groupby("model")["mean_fold_spearman"].mean()
    headline = {
        "evaluation": "two-fold leave-one-mouse-out",
        "targets": len(cfg["targets"]),
        "independent_biological_replicates": 2,
        "total_cells": int(adata.n_obs),
        "totalVI_mean_fold_spearman": float(mean_by_model["totalVI"]),
        "FusionVI_X_mean_fold_spearman": float(mean_by_model["FusionVI-X"]),
        "absolute_difference_not_like_for_like": float(mean_by_model["FusionVI-X"] - mean_by_model["totalVI"]),
        "interpretation": (
            "This is not a like-for-like comparison: it contrasts the native totalVI decoder with the "
            "FusionVI-X ridge readout (latent + remaining proteins + matching transcript). Use "
            "baselines_cross_mouse.py for fair readout controls. Two mice support a technical transfer "
            "result, not population-level inference."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS / "cross_mouse_metrics.csv", index=False)
    fold_summary.to_csv(RESULTS / "marker_summary.csv", index=False)
    pooled.to_csv(RESULTS / "pooled_metrics.csv", index=False)
    (RESULTS / "headline_results.json").write_text(json.dumps(headline, indent=2))
    fold_summary["model"] = fold_summary["model"].astype(str)
    print(fold_summary.pivot(index="target", columns="model", values="mean_fold_spearman").round(3))
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
