"""Scale-matched and calibrated re-evaluation of the complete-panel benchmark.

Why: RMSLE is minimized by predicting E[log1p y]. The neural models were scored
on log1p(E[y]) -- the log of the expected count including background -- which is
biased upward for over-dispersed counts (Jensen). The RNA ridge baseline is fit
directly on log1p(y). Its RMSLE advantage can therefore come from output scale
rather than from information about which cells express which protein.

For every saved model (no retraining) this script scores D2 three ways:

  log_mean     log1p(E[y | z])          the original metric (sanity check)
  pred_log     E[log1p y]               mean of log1p over posterior predictive
                                        draws from the protein NB mixture
  calibrated   a + b * log_mean         per-protein affine map fitted on D1
                                        cells pushed through the SAME RNA-only
                                        encoder route used for D2 (protein
                                        input hidden); no D2 truth is used

Each row reports RMSLE together with its decomposition
    RMSLE^2 = bias^2 + residual_sd^2
(bias = mean signed error in log1p units, residual_sd = SD of the error),
plus Spearman and Pearson, which are scale-free. A source-only RNA ridge is
scored on the same cells with the same decomposition.

Seed-paired contrasts (FusionVI minus each control) are computed for every
readout and metric, with the seed as the unit and Holm across the primary
contrasts within each readout x metric.

Usage (from src/):
    python evaluate_calibration.py                       # tier 0+1 arms, control_seeds
    python evaluate_calibration.py --arms totalvi fusionvi --seeds 2026 2027
    python evaluate_calibration.py --models-dir ../models/smoke --tag smoke   # smoke test
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
import yaml
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomialMixture
from scvi.model import TOTALVI
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from benchmark_arms import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
OUT = ROOT / "results"
LABELS = {"totalvi": "totalVI", "fusionvi": "FusionVI"}
READOUTS = ("log_mean", "pred_log", "calibrated")
CONTROLS = ("totalvi", "totalvi_w128", "totalvi_pmatch", "totalvi_avail_w128")


# ----------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model: TOTALVI, adata, indices: np.ndarray, source_code: int, n_samples: int,
            hide_protein: bool, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (log1p of expected counts, mean of log1p predictive draws), cells x proteins.

    Decoding uses the source batch (as in the paper and the original evaluation).
    With hide_protein=True the encoder sees no protein panel, so source cells
    take the same RNA-only route that target cells take.
    """
    module = model.module
    module.eval()
    encoder = module.encoder
    saved = getattr(encoder, "available_batch_indices", "absent")
    if hide_protein and saved != "absent":
        encoder.available_batch_indices = ()          # every cell treated as panel-missing
    log_mean, pred_log = [], []
    try:
        loader = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
        for tensors in loader:
            if hide_protein:
                tensors = dict(tensors)
                tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = torch.zeros_like(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])
            _, gen = module.forward(
                tensors,
                inference_kwargs={"n_samples": n_samples},
                generative_kwargs={"transform_batch": source_code},
                compute_loss=False,
            )
            py = gen["py_"]
            mix = torch.sigmoid(py["mixing"])
            mean = (py["rate_fore"] * (1 - mix) + py["rate_back"] * mix).mean(0)      # E[y], averaged over z draws
            dist = NegativeBinomialMixture(mu1=py["rate_back"], mu2=py["rate_fore"],
                                           theta1=py["r"], mixture_logits=py["mixing"])
            draws = dist.sample()                                                    # n_samples x cells x proteins
            log_mean.append(torch.log1p(mean).cpu().numpy())
            pred_log.append(torch.log1p(draws).mean(0).cpu().numpy())
    finally:
        if saved != "absent":
            encoder.available_batch_indices = saved
    return np.concatenate(log_mean), np.concatenate(pred_log)


def affine_calibrate(x_fit: np.ndarray, y_fit: np.ndarray, x_apply: np.ndarray) -> np.ndarray:
    """Per-protein least-squares y = a + b x, fitted on source cells only."""
    x_mean, y_mean = x_fit.mean(0), y_fit.mean(0)
    var = ((x_fit - x_mean) ** 2).mean(0)
    slope = np.divide(((x_fit - x_mean) * (y_fit - y_mean)).mean(0), var, out=np.zeros_like(var), where=var > 0)
    return np.maximum(y_mean + slope * (x_apply - x_mean), 0.0)


# ----------------------------------------------------------------- metrics
def protein_metrics(pred: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    err = pred - truth
    rows = {
        "rmsle": np.sqrt((err ** 2).mean(0)),
        "bias": err.mean(0),
        "residual_sd": err.std(0),
        "spearman": [stats.spearmanr(truth[:, j], pred[:, j]).statistic if pred[:, j].std() > 0 else np.nan
                     for j in range(truth.shape[1])],
        "pearson": [stats.pearsonr(truth[:, j], pred[:, j]).statistic if pred[:, j].std() > 0 else np.nan
                    for j in range(truth.shape[1])],
    }
    return pd.DataFrame(rows)


def ridge_predictions(adata, source: np.ndarray, target: np.ndarray, truth_log: np.ndarray) -> np.ndarray:
    """Source-only RNA ridge (same recipe as rna_baseline_paper_benchmark.py, wider alpha grid)."""
    x = sp.csr_matrix(adata.layers["counts"], dtype=np.float32)
    totals = np.asarray(x.sum(1)).ravel()
    x = sp.diags(np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)) @ x
    x.data = np.log1p(x.data)
    svd = TruncatedSVD(n_components=128, random_state=2026)
    xs, xt = svd.fit_transform(x[source]), svd.transform(x[target])
    ys = truth_log[source]
    tr, va = train_test_split(np.arange(len(xs)), test_size=0.2, random_state=2026,
                              stratify=adata.obs.loc[source, "cell_type"].astype(str))
    grid = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    scores = {a: np.sqrt(np.mean((np.maximum(Ridge(alpha=a).fit(xs[tr], ys[tr]).predict(xs[va]), 0) - ys[va]) ** 2))
              for a in grid}
    alpha = min(scores, key=scores.get)
    print(f"ridge alpha={alpha} (validation RMSLE {scores[alpha]:.4f})", flush=True)
    return np.maximum(Ridge(alpha=alpha).fit(xs, ys).predict(xt), 0.0)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out.tolist()


def contrasts(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (readout, metric), frame in per_seed.groupby(["readout", "metric"]):
        wide = frame.pivot(index="seed", columns="arm", values="value")
        if "fusionvi" not in wide:
            continue
        block = []
        for ref in CONTROLS:
            if ref not in wide:
                continue
            d = (wide["fusionvi"] - wide[ref]).dropna().to_numpy()
            if d.size < 2:
                continue
            lower_better = metric in ("rmsle", "abs_bias", "residual_sd")
            half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
            block.append({
                "readout": readout, "metric": metric, "reference": ref, "n_seeds": int(d.size),
                "fusionvi_minus_ref": float(d.mean()),
                "ci95_low": float(d.mean() - half), "ci95_high": float(d.mean() + half),
                "seeds_fusionvi_better": int(((d < 0) if lower_better else (d > 0)).sum()),
                "p_t": float(stats.ttest_1samp(d, 0).pvalue),
            })
        for row, p in zip(block, holm([r["p_t"] for r in block])):
            row["p_holm_within_readout_metric"] = p
        rows.extend(block)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=None, help="default: tier 0 and 1 arms")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: control_seeds")
    parser.add_argument("--models-dir", default=str(ROOT / "models" / "paper_benchmark"))
    parser.add_argument("--n-samples", type=int, default=None, help="default: posterior_samples from config")
    parser.add_argument("--calib-cells", type=int, default=4000, help="D1 cells used for calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="", help="suffix for output files (e.g. smoke)")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "paper_benchmark.yaml").read_text())
    arms = args.arms or [k for k, v in cfg["arms"].items() if v.get("tier", 0) <= 1]
    seeds = args.seeds or cfg["control_seeds"]
    n_samples = args.n_samples or int(cfg["posterior_samples"])
    models_dir = Path(args.models_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    rng = np.random.default_rng(2026)

    adata = sc.read_h5ad(DATA)
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    batch = adata.obs["batch"].astype(str).to_numpy()
    source, target = batch == cfg["source_batch"], batch == cfg["target_batch"]
    source_code = list(adata.obs["batch"].cat.categories).index(cfg["source_batch"])
    src_idx, tgt_idx = np.where(source)[0], np.where(target)[0]
    calib_idx = np.sort(rng.choice(src_idx, size=min(args.calib_cells, src_idx.size), replace=False))
    truth = adata.obsm["protein_truth"]
    proteins = np.asarray(truth.columns, dtype=str)
    truth_log = np.log1p(truth.to_numpy(dtype=np.float64))

    frames = []

    def add(arm: str, seed: int | None, readout: str, pred: np.ndarray) -> None:
        m = protein_metrics(pred, truth_log[tgt_idx])
        m.insert(0, "protein", proteins)
        m.insert(0, "readout", readout)
        m.insert(0, "seed", seed)
        m.insert(0, "arm", arm)
        frames.append(m)

    add("rna_ridge", None, "ridge", ridge_predictions(adata, source, target, truth_log))

    for arm in arms:
        for seed in seeds:
            weights = models_dir / f"{arm}__seed{seed}.pt"
            if not weights.exists():
                continue
            model = build_model(adata, cfg, cfg["arms"][arm])
            model.module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            model.module.to(args.device)
            log_mean_t, pred_log_t = predict(model, adata, tgt_idx, source_code, n_samples, hide_protein=False)
            log_mean_c, _ = predict(model, adata, calib_idx, source_code, n_samples, hide_protein=True)
            add(arm, seed, "log_mean", log_mean_t)
            add(arm, seed, "pred_log", pred_log_t)
            add(arm, seed, "calibrated", affine_calibrate(log_mean_c, truth_log[calib_idx], log_mean_t))
            print(f"evaluated {arm} seed {seed}", flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    metrics["model"] = metrics["arm"].map(lambda a: LABELS.get(a, a))
    metrics.to_csv(OUT / f"calibration_protein_metrics{suffix}.csv", index=False)

    # Per run: mean over proteins; |bias| averaged so opposite-signed proteins do not cancel.
    metrics["abs_bias"] = metrics["bias"].abs()
    value_cols = ["rmsle", "abs_bias", "residual_sd", "spearman", "pearson"]
    runs = metrics.groupby(["arm", "seed", "readout"], dropna=False)[value_cols].mean().reset_index()
    summary = runs.groupby(["arm", "readout"])[value_cols].mean()
    summary["n_seeds"] = runs.groupby(["arm", "readout"]).size()
    summary = summary.round(4)
    summary.to_csv(OUT / f"calibration_summary{suffix}.csv")

    per_seed = runs.dropna(subset=["seed"]).melt(id_vars=["arm", "seed", "readout"], value_vars=value_cols,
                                                 var_name="metric", value_name="value")
    table = contrasts(per_seed)
    table.to_csv(OUT / f"calibration_contrasts{suffix}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("\nMean over proteins, then over seeds:")
        print(summary.to_string())
        if len(table):
            print("\nFusionVI minus control (seed-paired):")
            print(table.round(4).to_string(index=False))
    (OUT / f"calibration_summary{suffix}.json").write_text(json.dumps({
        "readouts": {
            "log_mean": "log1p(E[y]) with background, decoded in the source batch (original metric)",
            "pred_log": "E[log1p y] over posterior predictive draws from the protein NB mixture",
            "calibrated": "per-protein affine map of log_mean fitted on D1 cells with protein input hidden",
            "ridge": "source-only RNA SVD(128) + ridge, alpha chosen on a D1 validation split",
        },
        "decomposition": "RMSLE^2 = bias^2 + residual_sd^2 per protein (log1p units)",
        "calibration_cells": int(calib_idx.size),
        "posterior_samples": n_samples,
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
    }, indent=2))


if __name__ == "__main__":
    main()
