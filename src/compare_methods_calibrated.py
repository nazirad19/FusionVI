"""Compare correctly read totalVI with other missing-protein methods.

Every method is scored on the target batch twice -- raw, and after the SAME
pre-specified calibration -- and the method ranking is compared.

Calibration (fixed in advance, applied identically to every method):
    per-protein affine map  observed log1p = a + b * predicted log1p,
    fitted on SOURCE cells only. No choice between calibrators is made on the
    target batch.

Source-cell predictions used for fitting the calibration are out-of-sample
where the method allows it:
    rna_ridge, knn_*   5-fold cross-fitting within the source batch
    seurat_v3          5-fold cross-fitting (R/seurat_transfer.R)
    neural models      source cells pushed through the target's input route
                       (panel hidden); these cells were seen in training, so
                       their calibration is in-sample -- a stated limitation.

Methods
    totalVI (per seed, from saved checkpoints), four label-free diagnostics:
        likelihood_mean        log1p E[y] from py_norm_ (primary)
        likelihood_median      predictive median from py_norm_
        likelihood_foreground  foreground mean from py_norm_
        helper_mean            log1p E[y] from py_ (historical only)
    rna_ridge        RNA SVD(128) -> log1p protein, multi-output ridge
    knn_log          k=50 neighbours in RNA SVD space, mean of log1p protein
    knn_count        k=50 neighbours, mean of raw counts (Seurat-like scale)
    seurat_v3        if results/seurat/<benchmark>/ exists (R/seurat_transfer.R)

Outputs (results/):
    method_comparison_runs<sfx>.csv        one row per method x seed x scoring
    method_comparison_summary<sfx>.csv     mean (and SD over seeds) per method x scoring
    method_comparison_ranking<sfx>.csv     ranks raw vs calibrated, per metric

Usage (from src/):
    python compare_methods_calibrated.py --benchmark paper
    python compare_methods_calibrated.py --benchmark paper --skip-neural --tag totalvix_baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomialMixture
from scvi.model import TOTALVI
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from benchmark_arms import build_model
from benchmarks import data_path, eval_proteins, load_config, models_dir, output_suffix
from evaluate_calibration import affine_calibrate, column_correlations, protein_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
LABELS = {"totalvi": "totalVI"}
METRICS = ["rmsle", "abs_bias", "residual_sd", "spearman", "pearson", "within_ct_spearman"]
LOWER_BETTER = {"rmsle", "abs_bias", "residual_sd"}


def within_celltype_spearman(pred: np.ndarray, truth: np.ndarray, cell_types: np.ndarray) -> float:
    """Mean protein-wise Spearman within cell types represented by at least 30 cells."""
    values = []
    for label in np.unique(cell_types):
        keep = cell_types == label
        if keep.sum() >= 30:
            values.append(np.nanmean(column_correlations(truth[keep], pred[keep], rank=True)))
    return float(np.nanmean(values))


# ------------------------------------------------------------------ neural readouts
@torch.no_grad()
def neural_readouts(model: TOTALVI, adata, indices: np.ndarray, source_code: int, n_samples: int,
                    hide_protein: bool, hide_columns: np.ndarray | None, batch_size: int = 256) -> dict[str, np.ndarray]:
    """Likelihood-consistent neural readouts decoded in the source batch."""
    module = model.module
    module.eval()
    encoder = module.encoder
    saved = getattr(encoder, "available_batch_indices", "absent")
    if hide_protein and saved != "absent":
        encoder.available_batch_indices = ()
    out = {
        "helper_mean": [],
        "likelihood_mean": [],
        "likelihood_median": [],
        "likelihood_foreground": [],
    }
    try:
        for tensors in model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False):
            tensors = dict(tensors)
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY].clone()
            if hide_protein:
                y.zero_()
            elif hide_columns is not None and len(hide_columns):
                y[:, torch.as_tensor(hide_columns, dtype=torch.long)] = 0.0
            tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = y
            _, gen = module.forward(tensors, inference_kwargs={"n_samples": n_samples},
                                    generative_kwargs={"transform_batch": source_code}, compute_loss=False)
            py, py_norm = gen["py_"], gen["py_norm_"]
            mix = torch.sigmoid(py_norm["mixing"])
            out["helper_mean"].append(torch.log1p(
                (py["rate_fore"] * (1 - mix) + py["rate_back"] * mix).mean(0)
            ).cpu().numpy())
            out["likelihood_mean"].append(torch.log1p(
                (py_norm["rate_fore"] * (1 - mix) + py_norm["rate_back"] * mix).mean(0)
            ).cpu().numpy())
            out["likelihood_foreground"].append(
                torch.log1p((py_norm["rate_fore"] * (1 - mix)).mean(0)).cpu().numpy()
            )
            draws = NegativeBinomialMixture(
                mu1=py_norm["rate_back"], mu2=py_norm["rate_fore"], theta1=py_norm["r"],
                mixture_logits=py_norm["mixing"],
            ).sample()
            out["likelihood_median"].append(torch.log1p(draws.median(0).values).cpu().numpy())
    finally:
        if saved != "absent":
            encoder.available_batch_indices = saved
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ simple baselines
def rna_svd(adata, src_idx, tgt_idx):
    x = sp.csr_matrix(adata.layers["counts"], dtype=np.float32)
    totals = np.asarray(x.sum(1)).ravel()
    x = sp.diags(np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)) @ x
    x.data = np.log1p(x.data)
    svd = TruncatedSVD(n_components=128, random_state=2026)
    xs = svd.fit_transform(x[src_idx])
    scaler = StandardScaler().fit(xs)
    return scaler.transform(xs), scaler.transform(svd.transform(x[tgt_idx]))


def safe_strata(strata: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Merge strata too small to split and absorb an undersized rare pool."""
    s = pd.Series(np.asarray(strata, dtype=str))
    s = s.where(s.map(s.value_counts()) >= min_count, "rare")
    if 0 < (s == "rare").sum() < min_count:
        nonrare = s[s != "rare"]
        s[s == "rare"] = nonrare.mode().iloc[0] if len(nonrare) else "all"
    return s.to_numpy()


def ridge_fit_predict(xs, ys, xt, strata):
    tr, va = train_test_split(np.arange(len(xs)), test_size=0.2, random_state=2026, stratify=safe_strata(strata))
    scores = {a: np.sqrt(np.mean((np.maximum(Ridge(alpha=a).fit(xs[tr], ys[tr]).predict(xs[va]), 0) - ys[va]) ** 2))
              for a in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)}
    return np.maximum(Ridge(alpha=min(scores, key=scores.get)).fit(xs, ys).predict(xt), 0.0)


def knn_predict(xs, ys_log, ys_raw, xt, k=50):
    nn = NearestNeighbors(n_neighbors=k).fit(xs)
    dist, idx = nn.kneighbors(xt)
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(1, keepdims=True)
    log_pred = np.einsum("nk,nkp->np", w, ys_log[idx])
    count_pred = np.log1p(np.einsum("nk,nkp->np", w, ys_raw[idx]))
    return log_pred, count_pred


def crossfit(fn, xs, strata, folds=5):
    """Out-of-fold source predictions from fn(train_idx, test_idx) -> predictions for test_idx."""
    out = None
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=2026).split(xs, strata):
        pred = fn(tr, te)
        out = np.zeros((len(xs), pred.shape[1])) if out is None else out
        out[te] = pred
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="paper")
    parser.add_argument("--skip-neural", action="store_true", help="score only ridge, kNN and available Seurat outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--calib-cells", type=int, default=0, help="source cells used for calibration; 0 means all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["target_batch"] in cfg["panel_available_batches"]:
        raise SystemExit("This comparison is for a complete-missing-panel benchmark.")
    seeds = args.seeds or cfg["control_seeds"]
    n_samples = args.n_samples or int(cfg["posterior_samples"])
    weights_dir = Path(args.models_dir) if args.models_dir else models_dir(cfg)
    sfx = output_suffix(cfg) + (f"_{args.tag}" if args.tag else "")
    rng = np.random.default_rng(2026)

    adata = sc.read_h5ad(data_path(cfg))
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    batch = adata.obs["batch"].astype(str).to_numpy()
    src_idx = np.where(batch == cfg["source_batch"])[0]
    tgt_idx = np.where(batch == cfg["target_batch"])[0]
    n_calib = src_idx.size if args.calib_cells <= 0 else min(args.calib_cells, src_idx.size)
    calib_idx = np.sort(rng.choice(src_idx, size=n_calib, replace=False))
    calib_rows = np.searchsorted(src_idx, calib_idx)
    source_code = list(adata.obs["batch"].cat.categories).index(cfg["source_batch"])
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    strata = safe_strata(cell_type[src_idx], min_count=5)

    proteins = eval_proteins(adata)
    truth_raw = adata.obsm["protein_truth"][proteins].to_numpy(dtype=np.float64)
    truth_log = np.log1p(truth_raw)
    y_src, y_tgt = truth_log[src_idx], truth_log[tgt_idx]
    ct_tgt = cell_type[tgt_idx]

    rows = []

    def score(method: str, readout: str, seed, pred_tgt: np.ndarray, pred_src: np.ndarray, y_src_fit: np.ndarray) -> None:
        for scoring, pred in (("raw", pred_tgt), ("calibrated", affine_calibrate(pred_src, y_src_fit, pred_tgt))):
            m = protein_metrics(pred, y_tgt)
            rows.append({"method": method, "readout": readout, "seed": seed, "scoring": scoring,
                         "rmsle": float(m["rmsle"].mean()), "abs_bias": float(m["bias"].abs().mean()),
                         "residual_sd": float(m["residual_sd"].mean()), "spearman": float(np.nanmean(m["spearman"])),
                         "pearson": float(np.nanmean(m["pearson"])),
                         "within_ct_spearman": within_celltype_spearman(pred, y_tgt, ct_tgt)})

    def score_raw_only(method: str, readout: str, pred_tgt: np.ndarray) -> None:
        m = protein_metrics(pred_tgt, y_tgt)
        rows.append({"method": method, "readout": readout, "seed": None, "scoring": "raw",
                     "rmsle": float(m["rmsle"].mean()), "abs_bias": float(m["bias"].abs().mean()),
                     "residual_sd": float(m["residual_sd"].mean()), "spearman": float(np.nanmean(m["spearman"])),
                     "pearson": float(np.nanmean(m["pearson"])),
                     "within_ct_spearman": within_celltype_spearman(pred_tgt, y_tgt, ct_tgt)})

    # ---- label-free simple baselines with cross-fitted source predictions
    xs, xt = rna_svd(adata, src_idx, tgt_idx)
    ridge_src = crossfit(lambda tr, te: ridge_fit_predict(xs[tr], y_src[tr], xs[te], strata[tr]), xs, strata)
    score("rna_ridge", "log", None, ridge_fit_predict(xs, y_src, xt, strata),
          ridge_src[calib_rows], y_src[calib_rows])
    raw_src = truth_raw[src_idx]
    knn_t = knn_predict(xs, y_src, raw_src, xt)
    knn_s = [crossfit(lambda tr, te, i=i: knn_predict(xs[tr], y_src[tr], raw_src[tr], xs[te])[i], xs, strata) for i in (0, 1)]
    score("knn", "log", None, knn_t[0], knn_s[0][calib_rows], y_src[calib_rows])
    score("knn", "count", None, knn_t[1], knn_s[1][calib_rows], y_src[calib_rows])
    print("baselines done", flush=True)

    # ---- Seurat v3 (from R/seurat_transfer.R)
    sdir = ROOT / "results" / "seurat" / args.benchmark
    if (sdir / "target_imputed.csv").exists() and (sdir / "source_crossfit_imputed.csv").exists():
        cells = np.asarray(adata.obs_names, dtype=str)
        tgt = pd.read_csv(sdir / "target_imputed.csv", index_col=0).reindex(index=cells[tgt_idx], columns=proteins)
        src = pd.read_csv(sdir / "source_crossfit_imputed.csv", index_col=0).reindex(index=cells[src_idx], columns=proteins)
        if tgt.isna().any().any() or src.isna().any().any():
            raise SystemExit(f"Seurat outputs in {sdir} do not cover every cell/protein; re-export and rerun.")
        score("seurat_v3", "count", None, np.log1p(np.clip(tgt.to_numpy(), 0, None)),
              np.log1p(np.clip(src.to_numpy(), 0, None))[calib_rows], y_src[calib_rows])
        print("seurat_v3 included", flush=True)
    else:
        official = sdir / "target_imputed_official.csv"
        if official.exists():
            tgt = pd.read_csv(official, index_col=0).reindex(index=np.asarray(adata.obs_names, dtype=str)[tgt_idx], columns=proteins)
            if tgt.isna().any().any():
                raise SystemExit(f"Official Seurat output in {official} is incomplete.")
            score_raw_only("seurat_v3_official", "count", np.log1p(np.clip(tgt.to_numpy(), 0, None)))
            print("official seurat_v3 target prediction included for raw scoring only", flush=True)
        else:
            print(f"seurat_v3 skipped (no outputs in {sdir}; run R/seurat_transfer.R)", flush=True)

    # ---- neural models from saved checkpoints
    for arm in ([] if args.skip_neural else ["totalvi"]):
        for seed in seeds:
            weights = weights_dir / f"{arm}__seed{seed}.pt"
            if not weights.exists():
                continue
            model = build_model(adata, cfg, cfg["arms"][arm])
            model.module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            model.module.to(args.device)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            t = neural_readouts(model, adata, tgt_idx, source_code, n_samples, hide_protein=False, hide_columns=None)
            c = neural_readouts(model, adata, calib_idx, source_code, n_samples, hide_protein=True, hide_columns=None)
            for readout in ("likelihood_mean", "likelihood_median", "likelihood_foreground", "helper_mean"):
                score(LABELS.get(arm, arm), readout, seed, t[readout], c[readout], y_src[calib_rows])
            print(f"evaluated {arm} seed {seed}", flush=True)

    runs = pd.DataFrame(rows)
    runs["label"] = runs["method"] + " (" + runs["readout"] + ")"
    is_totalvi = runs["method"] == "totalVI"
    is_primary = is_totalvi & (runs["readout"] == "likelihood_mean")
    is_helper = is_totalvi & (runs["readout"] == "helper_mean")
    runs.loc[is_primary & (runs["scoring"] == "raw"), "label"] = "totalVI (likelihood-consistent)"
    runs.loc[is_primary & (runs["scoring"] == "calibrated"), "label"] = "totalVI + D1 affine head"
    runs.loc[is_helper & (runs["scoring"] == "raw"), "label"] = "totalVI helper output (historical)"
    runs.loc[is_helper & (runs["scoring"] == "calibrated"), "label"] = "D1 affine head on helper output (historical)"
    runs.to_csv(OUT / f"method_comparison_runs{sfx}.csv", index=False)

    mean = runs.groupby(["label", "scoring"])[METRICS].mean()
    sd = runs.groupby(["label", "scoring"])[METRICS].std().add_suffix("_seed_sd")
    summary = mean.join(sd)
    summary["n_seeds"] = runs.groupby(["label", "scoring"]).size()
    summary.round(4).to_csv(OUT / f"method_comparison_summary{sfx}.csv")

    rank_key = runs["method"] + " (" + runs["readout"] + ")"
    rank_mean = runs.assign(rank_key=rank_key).groupby(["rank_key", "scoring"])[METRICS].mean()
    ranks = []
    for metric in METRICS:
        table = rank_mean[metric].unstack("scoring")
        for scoring in ("raw", "calibrated"):
            table[f"rank_{scoring}"] = table[scoring].rank(ascending=metric in LOWER_BETTER, method="min").astype("Int64")
        table["rank_change"] = table["rank_raw"] - table["rank_calibrated"]
        table.insert(0, "metric", metric)
        ranks.append(table.reset_index())
    ranks = pd.concat(ranks, ignore_index=True)
    ranks.to_csv(OUT / f"method_comparison_ranking{sfx}.csv", index=False)

    # Separate headline tables avoid comparing a raw method with a calibrated one.
    headline = runs[runs["readout"] != "helper_mean"].copy()
    headline_key = headline["method"] + " (" + headline["readout"] + ")"
    headline = headline.assign(method_key=headline_key)
    for scoring, reference_key in (
        ("raw", "totalVI (likelihood_mean)"),
        ("calibrated", "totalVI (likelihood_mean)"),
    ):
        table = headline[headline["scoring"] == scoring].groupby("method_key")[METRICS].mean()
        table["rmsle_rank"] = table["rmsle"].rank(ascending=True, method="min").astype("Int64")
        if reference_key in table.index:
            for metric in METRICS:
                table[f"totalvi_minus_method_{metric}"] = table.loc[reference_key, metric] - table[metric]
        table = table.sort_values(["rmsle_rank", "rmsle"])
        table.round(4).to_csv(OUT / f"totalvi_vs_methods_{scoring}{sfx}.csv")

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        for metric in ("rmsle", "within_ct_spearman"):
            print(f"\n{metric}: raw vs calibrated (rank 1 = best)")
            print(ranks[ranks["metric"] == metric].drop(columns="metric").round(4).sort_values("rank_calibrated").to_string(index=False))
    (OUT / f"method_comparison_meta{sfx}.json").write_text(json.dumps({
        "benchmark": args.benchmark, "scored_proteins": len(proteins), "posterior_samples": n_samples,
        "calibration": "per-protein affine on source cells, identical for every method, fixed before scoring",
        "source_predictions": {"rna_ridge/knn/seurat_v3": "5-fold cross-fitted", "neural": "route-matched, in-sample"},
        "neural_calibration_cells": int(calib_idx.size),
    }, indent=2))


if __name__ == "__main__":
    main()
