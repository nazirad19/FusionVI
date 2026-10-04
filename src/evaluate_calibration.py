"""Count-scale re-evaluation of saved checkpoints without retraining.

scvi-tools 1.x TOTALVI learns a per-protein, per-batch efficiency that
multiplies protein rates inside the likelihood (``py_norm_``). The public
normalized-expression helper returns ``py_`` rates without that factor. Scores
against observed protein counts therefore use ``py_norm_`` as the primary
readout.

For every saved model this script scores the target batch four ways:

  likelihood_mean      log1p E[y] from py_norm_ (primary)
  likelihood_pred_log  E[log1p y] from the py_norm_ predictive mixture
  calibrated           D1 affine head fitted to likelihood_mean using the
                       target's input-availability route; no target truth
  helper_mean          log1p E[y] from py_ (historical audit only)

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
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from benchmark_arms import build_model
from benchmarks import data_path, eval_proteins, load_config, models_dir, output_suffix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "paper_figure3_missing_protein.h5ad"
OUT = ROOT / "results"
LABELS = {
    "totalvi": "totalVI",
    "fusionvi": "FusionVI",
    "totalvi_moddrop_p25": "totalVI dropout 0.25",
    "totalvi_moddrop_p50": "totalVI dropout 0.50",
    "totalvi_moddrop_p75": "totalVI dropout 0.75",
}
READOUTS = ("likelihood_mean", "likelihood_pred_log", "calibrated", "helper_mean")
# (candidate, reference) pairs; Holm is applied across the pairs within each readout x metric.
PAIRS = (
    ("fusionvi", "totalvi"),
    ("fusionvi", "totalvi_w128"),
    ("fusionvi", "totalvi_pmatch"),
    ("fusionvi", "totalvi_avail_w128"),
    ("totalvi_avail_w128", "totalvi_w128"),
    ("fusionvi_moddrop", "totalvi_w128_moddrop"),   # fusion vs joint, both with modality dropout
    ("fusionvi_moddrop", "totalvi_moddrop"),
    ("fusionvi_moddrop", "fusionvi"),               # does dropout help FusionVI?
    ("totalvi_w128_moddrop", "totalvi_w128"),       # does dropout help a joint encoder?
    ("totalvi_moddrop_p25", "totalvi"),             # frozen development sweep
    ("totalvi_moddrop_p50", "totalvi"),
    ("totalvi_moddrop_p75", "totalvi"),
)


# ----------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model: TOTALVI, adata, indices: np.ndarray, source_code: int, n_samples: int,
            hide_protein: bool = False, batch_size: int = 512,
            hide_columns: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return likelihood-consistent and historical protein readouts.

    Decoding uses the source batch (as in the paper and the original evaluation).
    hide_protein=True: the encoder sees no protein panel and every cell is
    treated as panel-missing, i.e. the RNA-only route that complete-missing
    target cells take. hide_columns: zero only these protein inputs (the
    proteins absent from a partial target panel); availability is unchanged.
    """
    module = model.module
    module.eval()
    encoder = module.encoder
    saved = getattr(encoder, "available_batch_indices", "absent")
    if hide_protein and saved != "absent":
        encoder.available_batch_indices = ()          # every cell treated as panel-missing
    out = {"helper_mean": [], "likelihood_mean": [], "likelihood_pred_log": []}
    try:
        loader = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
        for tensors in loader:
            if hide_protein:
                tensors = dict(tensors)
                tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = torch.zeros_like(tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY])
            elif hide_columns is not None and len(hide_columns):
                tensors = dict(tensors)
                y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY].clone()
                y[:, torch.as_tensor(hide_columns, dtype=torch.long)] = 0.0
                tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY] = y
            _, gen = module.forward(
                tensors,
                inference_kwargs={"n_samples": n_samples},
                generative_kwargs={"transform_batch": source_code},
                compute_loss=False,
            )
            py = gen["py_"]
            pn = gen["py_norm_"]
            out["helper_mean"].append(torch.log1p(_mixture_mean(py)).cpu().numpy())
            out["likelihood_mean"].append(torch.log1p(_mixture_mean(pn)).cpu().numpy())
            draws = NegativeBinomialMixture(
                mu1=pn["rate_back"], mu2=pn["rate_fore"], theta1=pn["r"],
                mixture_logits=pn["mixing"],
            ).sample()
            out["likelihood_pred_log"].append(torch.log1p(draws).mean(0).cpu().numpy())
    finally:
        if saved != "absent":
            encoder.available_batch_indices = saved
    return {name: np.concatenate(parts) for name, parts in out.items()}


def _mixture_mean(rates: dict[str, torch.Tensor]) -> torch.Tensor:
    """Expected count of the background/foreground mixture, averaged over z."""
    mix = torch.sigmoid(rates["mixing"])
    return (rates["rate_fore"] * (1 - mix) + rates["rate_back"] * mix).mean(0)


@torch.no_grad()
def source_efficiency(model: TOTALVI, source_code: int) -> np.ndarray:
    """Return the model-owned per-protein efficiency for the decode batch."""
    log_efficiency = model.module.log_per_batch_efficiency
    if log_efficiency.ndim != 2 or not 0 <= source_code < log_efficiency.shape[1]:
        raise ValueError("Unexpected totalVI protein-efficiency parameter shape")
    return torch.exp(log_efficiency[:, source_code]).detach().cpu().numpy()


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
        "spearman": column_correlations(truth, pred, rank=True),
        "pearson": column_correlations(truth, pred, rank=False),
    }
    return pd.DataFrame(rows)


def column_correlations(y: np.ndarray, pred: np.ndarray, rank: bool) -> np.ndarray:
    """Vectorized column-wise Pearson or Spearman correlations."""
    if rank:
        y = stats.rankdata(y, axis=0)
        pred = stats.rankdata(pred, axis=0)
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    y = y - y.mean(0)
    pred = pred - pred.mean(0)
    numerator = np.sum(y * pred, axis=0)
    denominator = np.sqrt(np.sum(y * y, axis=0) * np.sum(pred * pred, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def ridge_predictions(
    adata,
    source: np.ndarray,
    target: np.ndarray,
    truth_log: np.ndarray,
    panel_log: np.ndarray | None = None,
) -> np.ndarray:
    """Source-only ridge, optionally augmented with proteins observed in the target."""
    x = sp.csr_matrix(adata.layers["counts"], dtype=np.float32)
    totals = np.asarray(x.sum(1)).ravel()
    x = sp.diags(np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)) @ x
    x.data = np.log1p(x.data)
    svd = TruncatedSVD(n_components=128, random_state=2026)
    xs, xt = svd.fit_transform(x[source]), svd.transform(x[target])
    if panel_log is not None:
        xs = np.hstack([xs, panel_log[source]])
        xt = np.hstack([xt, panel_log[target]])
    ys = truth_log[source]
    tr, va = train_test_split(np.arange(len(xs)), test_size=0.2, random_state=2026,
                              stratify=adata.obs.loc[source, "cell_type"].astype(str))
    grid = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    scores = {a: np.sqrt(np.mean((np.maximum(Ridge(alpha=a).fit(xs[tr], ys[tr]).predict(xs[va]), 0) - ys[va]) ** 2))
              for a in grid}
    alpha = min(scores, key=scores.get)
    print(f"ridge{'+panel' if panel_log is not None else ''} alpha={alpha} (validation RMSLE {scores[alpha]:.4f})", flush=True)
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
        block = []
        for cand, ref in PAIRS:
            if cand not in wide or ref not in wide:
                continue
            d = (wide[cand] - wide[ref]).dropna().to_numpy()
            if d.size < 2:
                continue
            lower_better = metric in ("rmsle", "abs_bias", "residual_sd")
            half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
            block.append({
                "readout": readout, "metric": metric, "candidate": cand, "reference": ref, "n_seeds": int(d.size),
                "candidate_minus_ref": float(d.mean()),
                "ci95_low": float(d.mean() - half), "ci95_high": float(d.mean() + half),
                "seeds_candidate_better": int(((d < 0) if lower_better else (d > 0)).sum()),
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
    parser.add_argument("--models-dir", default=None, help="default: the benchmark's models folder")
    parser.add_argument("--n-samples", type=int, default=None, help="default: posterior_samples from config")
    parser.add_argument("--calib-cells", type=int, default=4000, help="D1 cells used for calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="", help="suffix for output files (e.g. smoke)")
    parser.add_argument("--benchmark", default="paper")
    args = parser.parse_args()

    cfg = load_config(args.benchmark)
    arms = args.arms or [k for k, v in cfg["arms"].items() if v.get("tier", 0) <= 1]
    seeds = args.seeds or cfg["control_seeds"]
    n_samples = args.n_samples or int(cfg["posterior_samples"])
    weights_dir = Path(args.models_dir) if args.models_dir else models_dir(cfg)
    suffix = output_suffix(cfg) + (f"_{args.tag}" if args.tag else "")
    rng = np.random.default_rng(2026)

    adata = sc.read_h5ad(data_path(cfg))
    TOTALVI.setup_anndata(adata, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    batch = adata.obs["batch"].astype(str).to_numpy()
    source, target = batch == cfg["source_batch"], batch == cfg["target_batch"]
    source_code = list(adata.obs["batch"].cat.categories).index(cfg["source_batch"])
    src_idx, tgt_idx = np.where(source)[0], np.where(target)[0]
    calib_idx = np.sort(rng.choice(src_idx, size=min(args.calib_cells, src_idx.size), replace=False))
    truth = adata.obsm["protein_truth"]
    all_proteins = list(map(str, truth.columns))
    scored = eval_proteins(adata)
    scored_columns = np.asarray([all_proteins.index(name) for name in scored], dtype=int)
    proteins = np.asarray(scored, dtype=str)
    truth_log_all = np.log1p(truth.to_numpy(dtype=np.float64))
    truth_log = truth_log_all[:, scored_columns]
    partial_panel = len(scored_columns) < len(all_proteins)
    target_has_panel = cfg["target_batch"] in cfg["panel_available_batches"]
    calibration_route = {
        "hide_protein": not target_has_panel,
        "hide_columns": scored_columns if target_has_panel else None,
    }

    cell_types = adata.obs.loc[target, "cell_type"].astype(str).to_numpy()
    marker_tokens = {"CD4": "ADT_CD4_", "CD8": "ADT_CD8a_", "CD19": "ADT_CD19_"}
    marker_thresholds = {}
    for marker, token in marker_tokens.items():
        matches = [i for i, name in enumerate(proteins) if token in name]
        if not matches:
            continue
        j = matches[0]
        means = np.sort(
            GaussianMixture(n_components=2, random_state=2026)
            .fit(truth_log[source, j, None])
            .means_.reshape(-1)
        )
        marker_thresholds[marker] = (j, float(means.mean()))

    frames, within_rows, auc_rows, efficiency_rows = [], [], [], []

    def add(arm: str, seed: int | None, readout: str, pred: np.ndarray) -> None:
        m = protein_metrics(pred, truth_log[tgt_idx])
        m.insert(0, "protein", proteins)
        m.insert(0, "readout", readout)
        m.insert(0, "seed", seed)
        m.insert(0, "arm", arm)
        frames.append(m)
        for cell_type in np.unique(cell_types):
            mask = cell_types == cell_type
            if mask.sum() < 30:
                continue
            correlations = column_correlations(truth_log[tgt_idx][mask], pred[mask], rank=True)
            within_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "readout": readout,
                    "cell_type": cell_type,
                    "n_cells": int(mask.sum()),
                    "n_valid_proteins": int(np.isfinite(correlations).sum()),
                    "mean_spearman": float(np.nanmean(correlations)),
                    "median_spearman": float(np.nanmedian(correlations)),
                }
            )
        for marker, (j, threshold) in marker_thresholds.items():
            labels = truth_log[tgt_idx, j] > threshold
            auc_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "readout": readout,
                    "marker": marker,
                    "protein": proteins[j],
                    "source_log1p_threshold": threshold,
                    "positive_fraction": float(labels.mean()),
                    "auroc": float(roc_auc_score(labels, pred[:, j])),
                }
            )

    add("rna_ridge", None, "ridge", ridge_predictions(adata, source, target, truth_log))
    if partial_panel:
        observed_columns = np.setdiff1d(np.arange(len(all_proteins)), scored_columns)
        add(
            "rna_panel_ridge",
            None,
            "ridge",
            ridge_predictions(
                adata,
                source,
                target,
                truth_log,
                panel_log=truth_log_all[:, observed_columns],
            ),
        )

    for arm in arms:
        for seed in seeds:
            weights = weights_dir / f"{arm}__seed{seed}.pt"
            if not weights.exists():
                continue
            model = build_model(adata, cfg, cfg["arms"][arm])
            model.module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            model.module.to(args.device)
            efficiency = source_efficiency(model, source_code)
            efficiency_rows.extend(
                {
                    "arm": arm,
                    "seed": seed,
                    "protein": protein,
                    "source_batch": cfg["source_batch"],
                    "source_batch_efficiency": float(value),
                }
                for protein, value in zip(all_proteins, efficiency)
            )
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            target_readouts = {
                name: values[:, scored_columns]
                for name, values in predict(model, adata, tgt_idx, source_code, n_samples).items()
            }
            likelihood_mean_c = predict(
                model,
                adata,
                calib_idx,
                source_code,
                n_samples,
                **calibration_route,
            )["likelihood_mean"][:, scored_columns]
            add(arm, seed, "likelihood_mean", target_readouts["likelihood_mean"])
            add(arm, seed, "likelihood_pred_log", target_readouts["likelihood_pred_log"])
            add(arm, seed, "helper_mean", target_readouts["helper_mean"])
            add(
                arm,
                seed,
                "calibrated",
                affine_calibrate(
                    likelihood_mean_c,
                    truth_log[calib_idx],
                    target_readouts["likelihood_mean"],
                ),
            )
            print(f"evaluated {arm} seed {seed}", flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    metrics["model"] = metrics["arm"].map(lambda a: LABELS.get(a, a))
    metrics.to_csv(OUT / f"calibration_protein_metrics{suffix}.csv", index=False)
    within = pd.DataFrame(within_rows)
    auc = pd.DataFrame(auc_rows)
    within["model"] = within["arm"].map(lambda a: LABELS.get(a, a))
    if len(auc):
        auc["model"] = auc["arm"].map(lambda a: LABELS.get(a, a))
    within.to_csv(OUT / f"calibration_within_celltype{suffix}.csv", index=False)
    auc.to_csv(OUT / f"calibration_marker_auroc{suffix}.csv", index=False)
    efficiency = pd.DataFrame(efficiency_rows)
    if len(efficiency):
        efficiency["model"] = efficiency["arm"].map(lambda a: LABELS.get(a, a))
        efficiency.to_csv(OUT / f"calibration_efficiency{suffix}.csv", index=False)
        efficiency_summary = (
            efficiency.groupby(["arm", "seed"])["source_batch_efficiency"]
            .agg(["mean", "median", "min", "max"])
            .reset_index()
        )
    else:
        efficiency_summary = pd.DataFrame(columns=["arm", "seed", "mean", "median", "min", "max"])
    efficiency_summary.to_csv(OUT / f"calibration_efficiency_summary{suffix}.csv", index=False)

    # Per run: mean over proteins; |bias| averaged so opposite-signed proteins do not cancel.
    metrics["abs_bias"] = metrics["bias"].abs()
    value_cols = ["rmsle", "abs_bias", "residual_sd", "spearman", "pearson"]
    runs = metrics.groupby(["arm", "seed", "readout"], dropna=False)[value_cols].mean().reset_index()
    within_runs = (
        within.groupby(["arm", "seed", "readout"], dropna=False)["mean_spearman"]
        .mean()
        .rename("within_celltype_spearman")
        .reset_index()
    )
    if len(auc):
        auc_runs = (
            auc.groupby(["arm", "seed", "readout"], dropna=False)["auroc"]
            .mean()
            .rename("marker_auroc")
            .reset_index()
        )
    else:
        auc_runs = runs[["arm", "seed", "readout"]].copy()
        auc_runs["marker_auroc"] = np.nan
    runs = runs.merge(within_runs, on=["arm", "seed", "readout"], how="left")
    runs = runs.merge(auc_runs, on=["arm", "seed", "readout"], how="left")
    value_cols += ["within_celltype_spearman", "marker_auroc"]
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
            print("\nCandidate minus reference (seed-paired):")
            print(table.round(4).to_string(index=False))
    (OUT / f"calibration_summary{suffix}.json").write_text(json.dumps({
        "readouts": {
            "likelihood_mean": "PRIMARY. log1p of expected count from py_norm_, including learned protein efficiency",
            "likelihood_pred_log": "E[log1p y] over posterior predictive draws from the py_norm_ mixture",
            "calibrated": "D1 affine head applied to likelihood_mean using the target's input-availability route",
            "helper_mean": "HISTORICAL ONLY. log1p expected count from py_; learned efficiency omitted",
            "ridge": "source-only RNA SVD(128) + ridge, alpha chosen on a D1 validation split",
        },
        "decomposition": "RMSLE^2 = bias^2 + residual_sd^2 per protein (log1p units)",
        "benchmark": args.benchmark,
        "scored_proteins": int(len(scored_columns)),
        "calibration_route": "RNA only" if calibration_route["hide_protein"] else "observed target panel retained",
        "biological_metrics": {
            "within_celltype_spearman": "Spearman within each D2 cell type with at least 30 cells, averaged over cell-type/protein pairs",
            "marker_auroc": "Mean D2 AUROC for prespecified CD4, CD8a and CD19 targets using source-D1 mixture thresholds",
        },
        "calibration_cells": int(calib_idx.size),
        "posterior_samples": n_samples,
        "efficiency_output": f"calibration_efficiency_summary{suffix}.csv",
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
    }, indent=2))


if __name__ == "__main__":
    main()
