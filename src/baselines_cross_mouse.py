"""Latent-free baselines for Experiment 3 (cross-mouse hidden-marker recovery).

These readouts need no trained model, so they isolate how much of the
cross-modal (X) readout is explained by the remaining protein panel alone.

Readouts (ridge head fitted on the training mouse, evaluated on the held-out mouse):
  rna_proxy                 log1p counts of the matching transcript
  protein_context           the 106 non-masked proteins
  protein_context_plus_rna  protein context + matching transcript
                            (= the X readout without the latent state)

If latent arrays from train_fold.py exist in results/final_folds, the script
also refits the full X readout (latent + context + RNA) for both models and
reports the latent increment: X minus protein_context_plus_rna.
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
FOLDS = ROOT / "results" / "final_folds"
OUT = ROOT / "results"


def rho(y: np.ndarray, p: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def ridge(x_tr, y_tr, x_te, alpha):
    head = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    head.fit(x_tr, y_tr)
    return head.predict(x_te)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    alpha = float(cfg["ridge_alpha"])
    adata = ad.read_h5ad(ADATA)
    proteins = np.log1p(adata.obsm["protein_counts"].astype(float))
    targets = list(cfg["targets"])
    context = proteins[[c for c in proteins.columns if c not in targets]]
    mouse = adata.obs["mouse"].astype(str)

    rows = []
    for fold in (0, 1):
        test_mask = (mouse == f"mouse{fold}").to_numpy()
        train_ids, test_ids = adata.obs_names[~test_mask], adata.obs_names[test_mask]
        latents = {}
        for model in ("totalvi", "fusionvi"):
            d = FOLDS / f"{model}__mouse{fold}__seed{cfg['project_seed']}"
            if (d / "latent_train.npy").exists():
                latents[model] = (np.load(d / "latent_train.npy"), np.load(d / "latent_test.npy"))

        for target, gene in cfg["targets"].items():
            y_tr = proteins.loc[train_ids, target].to_numpy()
            y_te = proteins.loc[test_ids, target].to_numpy()
            counts = adata[:, gene].layers["counts"]
            counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
            rna = pd.Series(np.log1p(counts.reshape(-1)), index=adata.obs_names)
            r_tr, r_te = rna.loc[train_ids].to_numpy()[:, None], rna.loc[test_ids].to_numpy()[:, None]
            c_tr, c_te = context.loc[train_ids].to_numpy(), context.loc[test_ids].to_numpy()

            preds = {
                "rna_proxy": r_te[:, 0],
                "protein_context": ridge(c_tr, y_tr, c_te, alpha),
                "protein_context_plus_rna": ridge(np.hstack([c_tr, r_tr]), y_tr, np.hstack([c_te, r_te]), alpha),
            }
            for model, (z_tr, z_te) in latents.items():
                preds[f"{model}_xmodal"] = ridge(
                    np.hstack([z_tr, c_tr, r_tr]), y_tr, np.hstack([z_te, c_te, r_te]), alpha
                )
            for readout, p in preds.items():
                rows.append({"heldout_mouse": fold, "target": target, "readout": readout, "spearman": rho(y_te, p)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "cross_mouse_baselines.csv", index=False)
    table = metrics.groupby(["target", "readout"])["spearman"].mean().unstack("readout")
    table.loc["MEAN"] = table.mean()
    summary = {"mean_fold_spearman": table.loc["MEAN"].round(4).to_dict()}
    for model in ("totalvi", "fusionvi"):
        col = f"{model}_xmodal"
        if col in table:
            summary[f"{model}_latent_increment"] = (table[col] - table["protein_context_plus_rna"]).round(4).to_dict()
    (OUT / "cross_mouse_baselines_summary.json").write_text(json.dumps(summary, indent=2))
    print(table.round(3).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
