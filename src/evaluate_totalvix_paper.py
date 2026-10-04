"""Evaluate a source-trained totalVI-X readout on the paper's SLN111 benchmark.

The original benchmark hides every SLN111-D2 protein.  Consequently, the
readout is restricted to information available for an scRNA-seq target cell:

* the frozen totalVI RNA-only latent mean;
* the frozen totalVI protein-decoder prediction; and
* a source-fitted low-dimensional RNA representation.

For each saved totalVI initialization, a per-protein affine calibration and a
multi-output ridge residual model are trained only on SLN111-D1.  Hyperparameter
selection also uses D1 only.  D2 protein truth is opened only for final scoring.
This keeps the dataset, split, trained totalVI parameters and paper RMSLE metric
unchanged while testing a reproducible post-hoc extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.model import TOTALVI
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark_arms import build_model
from benchmarks import data_path, eval_proteins, load_config, models_dir
from evaluate_calibration import affine_calibrate, column_correlations


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def log_normalize(counts) -> sp.csr_matrix:
    """Library-size normalize and log1p transform a sparse RNA count matrix."""
    x = sp.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(x.sum(1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    x = sp.diags(scale) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


@torch.no_grad()
def rna_only_latent(model: TOTALVI, adata, indices: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return posterior latent means after setting every protein input to zero."""
    module = model.module
    module.eval()
    rows = []
    for tensors in model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False):
        tensors = dict(tensors)
        tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = torch.zeros_like(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])
        out = module.inference(**module._get_inference_input(tensors))
        qz = out["qz"]
        rows.append(qz.loc.detach().cpu().numpy())
    return np.concatenate(rows)


@torch.no_grad()
def expected_count_readouts(
    model: TOTALVI,
    adata,
    indices: np.ndarray,
    source_code: int,
    n_samples: int,
    hide_protein: bool,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log1p expected counts without and with learned protein efficiency."""
    module = model.module
    module.eval()
    helper_scale, likelihood_scale = [], []
    for tensors in model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False):
        tensors = dict(tensors)
        if hide_protein:
            tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = torch.zeros_like(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])
        _, gen = module.forward(
            tensors,
            inference_kwargs={"n_samples": n_samples},
            generative_kwargs={"transform_batch": source_code},
            compute_loss=False,
        )
        py = gen["py_"]
        py_norm = gen["py_norm_"]
        mixing = torch.sigmoid(py["mixing"])
        raw_mean = (py["rate_fore"] * (1 - mixing) + py["rate_back"] * mixing).mean(0)
        efficient_mean = (
            py_norm["rate_fore"] * (1 - mixing) + py_norm["rate_back"] * mixing
        ).mean(0)
        helper_scale.append(torch.log1p(raw_mean).cpu().numpy())
        likelihood_scale.append(torch.log1p(efficient_mean).cpu().numpy())
    return np.concatenate(helper_scale), np.concatenate(likelihood_scale)


def metric_row(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Paper RMSLE plus scale-free diagnostics, averaged over proteins."""
    err = pred - truth
    return {
        "rmsle": float(np.sqrt(np.square(err).mean(0)).mean()),
        "global_rmsle": float(np.sqrt(np.square(err).mean())),
        "abs_bias": float(np.abs(err.mean(0)).mean()),
        "residual_sd": float(err.std(0).mean()),
        "spearman": float(np.nanmean(column_correlations(truth, pred, rank=True))),
        "pearson": float(np.nanmean(column_correlations(truth, pred, rank=False))),
    }


def mean_within_celltype_spearman(pred: np.ndarray, truth: np.ndarray, cell_types: np.ndarray) -> float:
    values = []
    for label in np.unique(cell_types):
        keep = cell_types == label
        if keep.sum() < 30:
            continue
        values.append(np.nanmean(column_correlations(truth[keep], pred[keep], rank=True)))
    return float(np.nanmean(values))


def fit_totalvix(
    base_source: np.ndarray,
    base_target: np.ndarray,
    latent_source: np.ndarray,
    latent_target: np.ndarray,
    rna_source: sp.csr_matrix,
    rna_target: sp.csr_matrix,
    truth_source: np.ndarray,
    labels_source: np.ndarray,
    seed: int,
    n_svd: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit D1-only affine + residual ridge and return D2 predictions."""
    all_idx = np.arange(len(truth_source))
    train_idx, valid_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=seed,
        stratify=labels_source,
    )

    # Source-only model selection.  The affine term controls output scale; the
    # ridge learns remaining cell-specific residual structure.
    calibrated_valid = affine_calibrate(base_source[train_idx], truth_source[train_idx], base_source[valid_idx])
    svd_tune = TruncatedSVD(n_components=n_svd, random_state=seed)
    rna_train = svd_tune.fit_transform(rna_source[train_idx])
    rna_valid = svd_tune.transform(rna_source[valid_idx])
    x_train = np.hstack([base_source[train_idx], latent_source[train_idx], rna_train])
    x_valid = np.hstack([base_source[valid_idx], latent_source[valid_idx], rna_valid])
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    calibrated_train = affine_calibrate(base_source[train_idx], truth_source[train_idx], base_source[train_idx])
    residual_train = truth_source[train_idx] - calibrated_train

    grid = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    scores = {}
    for alpha in grid:
        residual = Ridge(alpha=alpha, solver="lsqr").fit(x_train, residual_train).predict(x_valid)
        pred = np.maximum(calibrated_valid + residual, 0.0)
        scores[alpha] = metric_row(pred, truth_source[valid_idx])["rmsle"]
    alpha = min(scores, key=scores.get)

    # Final fit uses every source cell after D1-only model selection.
    calibrated_source = affine_calibrate(base_source, truth_source, base_source)
    calibrated_target = affine_calibrate(base_source, truth_source, base_target)
    svd = TruncatedSVD(n_components=n_svd, random_state=seed)
    rna_source_svd = svd.fit_transform(rna_source)
    rna_target_svd = svd.transform(rna_target)
    x_source = np.hstack([base_source, latent_source, rna_source_svd])
    x_target = np.hstack([base_target, latent_target, rna_target_svd])
    scaler = StandardScaler().fit(x_source)
    residual = Ridge(alpha=alpha, solver="lsqr").fit(
        scaler.transform(x_source), truth_source - calibrated_source
    ).predict(scaler.transform(x_target))
    return np.maximum(calibrated_target + residual, 0.0), {
        "alpha": float(alpha),
        "validation_rmsle": float(scores[alpha]),
    }


def paired_summary(frame: pd.DataFrame, candidate: str, reference: str, metric: str) -> dict[str, float]:
    wide = frame.pivot(index="seed", columns="readout", values=metric)
    delta = (wide[candidate] - wide[reference]).dropna().to_numpy()
    half = stats.t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else np.nan
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "n_seeds": int(len(delta)),
        "candidate_minus_reference": float(delta.mean()),
        "ci95_low": float(delta.mean() - half),
        "ci95_high": float(delta.mean() + half),
        "candidate_wins": int((delta < 0).sum() if metric in {"rmsle", "global_rmsle", "abs_bias", "residual_sd"} else (delta > 0).sum()),
        "p_t": float(stats.ttest_1samp(delta, 0).pvalue) if len(delta) > 1 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-svd", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = load_config("paper")
    seeds = args.seeds or cfg["control_seeds"]
    n_samples = int(args.n_samples or cfg["posterior_samples"])
    adata = sc.read_h5ad(data_path(cfg))
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    batch = adata.obs["batch"].astype(str).to_numpy()
    source = batch == cfg["source_batch"]
    target = batch == cfg["target_batch"]
    src_idx, tgt_idx = np.where(source)[0], np.where(target)[0]
    source_code = list(adata.obs["batch"].cat.categories).index(cfg["source_batch"])
    proteins = eval_proteins(adata)
    truth = adata.obsm["protein_truth"].loc[:, proteins].to_numpy(dtype=np.float64)
    truth_log = np.log1p(truth)
    labels_target = adata.obs.loc[target, "cell_type"].astype(str).to_numpy()

    rows = []
    protein_rows = []
    efficiency_rows = []
    for seed in seeds:
        weights = models_dir(cfg) / f"totalvi__seed{seed}.pt"
        if not weights.exists():
            print(f"missing {weights}; skipping", flush=True)
            continue
        model = build_model(adata, cfg, cfg["arms"]["totalvi"])
        model.module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.module.to(args.device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base_source, efficient_source = expected_count_readouts(
            model, adata, src_idx, source_code, n_samples, hide_protein=True
        )
        base_target, efficient_target = expected_count_readouts(
            model, adata, tgt_idx, source_code, n_samples, hide_protein=False
        )
        calibrated = affine_calibrate(base_source, truth_log[src_idx], base_target)
        efficient_calibrated = affine_calibrate(
            efficient_source, truth_log[src_idx], efficient_target
        )
        efficiency = torch.exp(model.module.log_per_batch_efficiency[:, source_code]).detach().cpu().numpy()
        efficiency_rows.extend(
            {"seed": seed, "protein": protein, "source_batch_efficiency": float(value)}
            for protein, value in zip(proteins, efficiency)
        )

        for name, pred in (
            ("totalVI helper-uncorrected", base_target),
            ("totalVI likelihood-consistent", efficient_target),
            ("D1 affine head", calibrated),
            ("totalVI likelihood-consistent + D1 head", efficient_calibrated),
        ):
            record = {"seed": seed, "readout": name, **metric_row(pred, truth_log[tgt_idx])}
            record["within_celltype_spearman"] = mean_within_celltype_spearman(pred, truth_log[tgt_idx], labels_target)
            rows.append(record)
            per_protein = np.sqrt(np.square(pred - truth_log[tgt_idx]).mean(0))
            protein_rows.extend(
                {"seed": seed, "readout": name, "protein": protein, "rmsle": float(value)}
                for protein, value in zip(proteins, per_protein)
            )
        print(f"evaluated efficiency audit seed {seed}: mean D1 efficiency={efficiency.mean():.4f}", flush=True)

    frame = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    frame.to_csv(OUT / f"totalvix_paper_seed_summary{suffix}.csv", index=False)
    pd.DataFrame(protein_rows).to_csv(OUT / f"totalvix_paper_protein_metrics{suffix}.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv(OUT / f"totalvi_efficiency_factors{suffix}.csv", index=False)
    comparisons = []
    for candidate, reference in (
        ("totalVI likelihood-consistent", "totalVI helper-uncorrected"),
        ("D1 affine head", "totalVI likelihood-consistent"),
        ("totalVI likelihood-consistent + D1 head", "totalVI likelihood-consistent"),
    ):
        for metric in ("rmsle", "global_rmsle", "spearman", "pearson", "within_celltype_spearman"):
            comparisons.append(paired_summary(frame, candidate, reference, metric))
    pd.DataFrame(comparisons).to_csv(OUT / f"totalvix_paper_contrasts{suffix}.csv", index=False)
    with pd.option_context("display.width", 200):
        print("\nMean across saved initializations")
        print(frame.groupby("readout").mean(numeric_only=True).round(4).to_string())
        print("\nPaired contrasts")
        print(pd.DataFrame(comparisons).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
