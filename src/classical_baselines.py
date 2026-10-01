"""Donor-held-out RNA, protein, and equal late-fusion PCA baselines.

These transparent baselines test whether a deep generative model is needed for
the biological readouts. The three held-out activation markers are excluded
from every protein-derived input, exactly as in totalVI/FusionVI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import CONTRASTS, MARKERS, SEED, build_rna_proxies, safe_auc


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "lawlor_pbmc_citeseq.h5ad"
OUTPUT = ROOT / "results" / "classical_baseline_metrics.csv"
HELDOUT = ["CD25", "CD69", "HLA-DR"]


def log_normalize_rna(x):
    x = sparse.csr_matrix(x, dtype=np.float32, copy=True)
    totals = np.asarray(x.sum(axis=1)).ravel()
    factors = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    x = x.multiply(factors[:, None]).tocsr()
    x.data = np.log1p(x.data)
    return x


def clr_protein(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    denominator = np.exp(np.mean(np.log1p(y), axis=1, keepdims=True))
    return np.log1p(y / np.maximum(denominator, 1e-8))


def fit_embeddings(adata, train_mask: np.ndarray, test_mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rna_train = log_normalize_rna(adata.layers["counts"][train_mask])
    rna_test = log_normalize_rna(adata.layers["counts"][test_mask])
    rna_svd = TruncatedSVD(n_components=15, random_state=SEED)
    z_rna_train = rna_svd.fit_transform(rna_train)
    z_rna_test = rna_svd.transform(rna_test)

    proteins = list(map(str, adata.obsm["protein_counts"].columns))
    protein_indices = [i for i, name in enumerate(proteins) if name not in HELDOUT]
    y = np.asarray(adata.obsm["protein_counts"])
    y_train, y_test = clr_protein(y[train_mask][:, protein_indices]), clr_protein(y[test_mask][:, protein_indices])
    protein_pipe = make_pipeline(StandardScaler(), PCA(n_components=15, random_state=SEED))
    z_protein_train = protein_pipe.fit_transform(y_train)
    z_protein_test = protein_pipe.transform(y_test)

    branch_scaler = StandardScaler()
    joined_train = branch_scaler.fit_transform(np.c_[z_rna_train, z_protein_train])
    joined_test = branch_scaler.transform(np.c_[z_rna_test, z_protein_test])
    joint_pca = PCA(n_components=15, random_state=SEED)
    z_joint_train = joint_pca.fit_transform(joined_train)
    z_joint_test = joint_pca.transform(joined_test)
    return {
        "rna_pca": (z_rna_train, z_rna_test),
        "protein_pca": (z_protein_train, z_protein_test),
        "equal_fusion_pca": (z_joint_train, z_joint_test),
    }


def add_latent_rows(rows: list[dict], model: str, donor: str, z_train, train_meta, z_test, test_meta) -> None:
    for target, spec in CONTRASTS.items():
        tr = train_meta["cell_type"].isin(spec["cell_types"]) & train_meta["condition"].isin(["Baseline", spec["positive"]])
        te = test_meta["cell_type"].isin(spec["cell_types"]) & test_meta["condition"].isin(["Baseline", spec["positive"]])
        y_train = (train_meta.loc[tr, "condition"] == spec["positive"]).astype(int).to_numpy()
        y_test = (test_meta.loc[te, "condition"] == spec["positive"]).astype(int).to_numpy()
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED))
        classifier.fit(z_train[tr.to_numpy()], y_train)
        score = classifier.predict_proba(z_test[te.to_numpy()])[:, 1]
        values = {
            "n_cells": len(y_test),
            "auc": safe_auc(y_test, score),
            "balanced_accuracy": balanced_accuracy_score(y_test, score >= 0.5),
        }
        for metric, value in values.items():
            rows.append({"model": model, "donor": donor, "analysis": "latent_response", "target": target, "metric": metric, "value": value})


def add_marker_rows(rows: list[dict], model: str, donor: str, z_train, train_meta, z_test, test_meta, adata, rna_proxies) -> None:
    proteins = adata.obsm["protein_counts"]
    for target, spec in MARKERS.items():
        marker = spec["marker"]
        true_train = np.asarray(proteins.loc[train_meta.index, marker], dtype=float)
        true_test = np.asarray(proteins.loc[test_meta.index, marker], dtype=float)
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        regressor.fit(z_train, np.log1p(true_train))
        prediction = regressor.predict(z_test)
        keep = test_meta["cell_type"].isin(spec["cell_types"]) & test_meta["condition"].isin(["Baseline", spec["positive"]])
        y = (test_meta.loc[keep, "condition"] == spec["positive"]).astype(int).to_numpy()
        pred = prediction[keep.to_numpy()]
        true = true_test[keep.to_numpy()]
        rho = spearmanr(true, pred).statistic if np.std(true) > 0 and np.std(pred) > 0 else np.nan
        pred_auc = safe_auc(y, pred)
        true_auc = safe_auc(y, true)
        values = {
            "n_cells": int(keep.sum()),
            "pred_auc": pred_auc,
            "pred_ap": average_precision_score(y, pred),
            "true_auc": true_auc,
            "effect_concordance": 1.0 - abs(pred_auc - true_auc),
            "spearman_true_pred": rho,
        }
        proxy = rna_proxies[marker].reindex(test_meta.index).loc[keep]
        rna_rank = proxy.rank(pct=True)
        protein_rank = pd.Series(true, index=proxy.index).rank(pct=True)
        discordant = (rna_rank - protein_rank).abs() >= 0.5
        discordant_pred = pd.Series(pred, index=proxy.index).loc[discordant]
        discordant_true = pd.Series(true, index=proxy.index).loc[discordant]
        values["discordant_n"] = int(discordant.sum())
        values["discordant_spearman"] = (
            spearmanr(discordant_true, discordant_pred).statistic
            if len(discordant_true) >= 10 and discordant_true.std() > 0 and discordant_pred.std() > 0
            else np.nan
        )
        for metric, value in values.items():
            rows.append({"model": model, "donor": donor, "analysis": "heldout_marker", "target": target, "metric": metric, "value": value})


def main() -> None:
    adata = sc.read_h5ad(DATA)
    rna_proxies = build_rna_proxies(adata)
    donor_values = adata.obs["donor"].astype(str)
    rows: list[dict] = []
    for donor in sorted(donor_values.unique()):
        train_mask = (donor_values != donor).to_numpy()
        test_mask = (donor_values == donor).to_numpy()
        train_meta = adata.obs.loc[train_mask, ["donor", "lane", "condition", "cell_type"]].copy()
        test_meta = adata.obs.loc[test_mask, ["donor", "lane", "condition", "cell_type"]].copy()
        for model, (z_train, z_test) in fit_embeddings(adata, train_mask, test_mask).items():
            add_latent_rows(rows, model, donor, z_train, train_meta, z_test, test_meta)
            add_marker_rows(rows, model, donor, z_train, train_meta, z_test, test_meta, adata, rna_proxies)
        print(f"Completed classical baselines for {donor}")
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
