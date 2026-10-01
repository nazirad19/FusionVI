"""Donor-safe targeted cross-modal readout for pharmacodynamic markers.

The generative models keep CD25, CD69 and HLA-DR hidden from their encoder
inputs. This script adds a supervised readout trained only on the nine training
donors in each outer fold. Inputs are the saved latent state, all non-held-out
ADTs and a prespecified marker-matched RNA proxy. The readout family is chosen
by grouped inner cross-validation on training donors, so the outer donor never
influences model selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "folds"
RESULTS = ROOT / "results"
ADATA = ROOT / "data" / "processed" / "lawlor_pbmc_citeseq.h5ad"
SEED = 2026

MARKERS = {
    "CD25_T": {
        "marker": "CD25",
        "cell_types": ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem"],
        "positive": "CD3_CD28",
        "rna": ["ENSG00000134460"],
    },
    "CD69_T": {
        "marker": "CD69",
        "cell_types": ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem"],
        "positive": "CD3_CD28",
        "rna": ["ENSG00000110848"],
    },
    "HLA_DR_Monocyte": {
        "marker": "HLA-DR",
        "cell_types": ["CD14_Mono"],
        "positive": "LPS",
        "rna": ["ENSG00000204287", "ENSG00000196126"],
    },
}

HEADS = ("ridge", "histgb", "extra_trees")


def make_head(name: str):
    """Return one fixed candidate; only the family is selected in inner CV."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "histgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=SEED,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=250,
            max_features=0.8,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=SEED,
        )
    raise KeyError(name)


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 10 or np.std(y) == 0 or np.std(pred) == 0:
        return np.nan
    return float(spearmanr(y, pred).statistic)


def select_head(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[str, dict[str, float]]:
    """Choose a head using only grouped training-donor validation folds."""
    splitter = GroupKFold(n_splits=3)
    scores: dict[str, float] = {}
    for name in HEADS:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x, y, groups):
            model = make_head(name)
            model.fit(x[train_idx], y[train_idx])
            fold_scores.append(safe_spearman(y[valid_idx], model.predict(x[valid_idx])))
        scores[name] = float(np.nanmean(fold_scores))
    selected = max(scores, key=scores.get)
    return selected, scores


def feature_block(
    variant: str,
    latent: np.ndarray | None,
    observed_protein: np.ndarray,
    rna_proxy: np.ndarray,
) -> np.ndarray:
    """Construct full model and single-modality ablations."""
    if variant == "rna_proxy_only":
        return rna_proxy[:, None]
    if variant == "protein_context_only":
        return observed_protein
    if variant == "context_no_latent":
        return np.column_stack([observed_protein, rna_proxy])
    if latent is None:
        raise ValueError(f"{variant} requires a latent representation")
    return np.column_stack([latent, observed_protein, rna_proxy])


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def main() -> None:
    adata = ad.read_h5ad(ADATA)
    protein = adata.obsm["protein_counts"].astype(float)
    hidden = [spec["marker"] for spec in MARKERS.values()]
    observed_names = [name for name in protein.columns if name not in hidden]
    observed = np.log1p(protein[observed_names])
    rna_proxy = {}
    for target, spec in MARKERS.items():
        values = np.asarray(adata[:, spec["rna"]].layers["counts"].sum(axis=1)).reshape(-1)
        rna_proxy[target] = pd.Series(np.log1p(values), index=adata.obs_names)

    donors = sorted(path.name.split("__")[1] for path in FOLDS.glob("fusionvi__*__seed2026"))
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    variants = {
        "rna_proxy_only": None,
        "protein_context_only": None,
        "context_no_latent": None,
        "totalvi_xmodal": "totalvi",
        "fusionvi_xmodal": "fusionvi",
    }

    for fold_number, donor in enumerate(donors, 1):
        fusion_dir = FOLDS / f"fusionvi__{donor}__seed2026"
        train_meta = pd.read_csv(fusion_dir / "train_metadata.csv", index_col=0)
        test_meta = pd.read_csv(fusion_dir / "test_metadata.csv", index_col=0)
        latent_cache = {
            name: (
                np.load(FOLDS / f"{name}__{donor}__seed2026" / "latent_train.npy"),
                np.load(FOLDS / f"{name}__{donor}__seed2026" / "latent_test.npy"),
            )
            for name in ("totalvi", "fusionvi")
        }
        train_observed = observed.loc[train_meta.index].to_numpy()
        test_observed = observed.loc[test_meta.index].to_numpy()

        for target, spec in MARKERS.items():
            train_keep = train_meta["cell_type"].isin(spec["cell_types"]) & train_meta["condition"].isin(["Baseline", spec["positive"]])
            test_keep = test_meta["cell_type"].isin(spec["cell_types"]) & test_meta["condition"].isin(["Baseline", spec["positive"]])
            train_ids = train_meta.index[train_keep]
            test_ids = test_meta.index[test_keep]
            y_train = np.log1p(protein.loc[train_ids, spec["marker"]].to_numpy())
            y_test = protein.loc[test_ids, spec["marker"]].to_numpy()
            train_proxy = rna_proxy[target].loc[train_ids].to_numpy()
            test_proxy = rna_proxy[target].loc[test_ids].to_numpy()
            train_groups = train_meta.loc[train_ids, "donor"].to_numpy()

            rna_rank = pd.Series(test_proxy, index=test_ids).rank(pct=True)
            protein_rank = pd.Series(y_test, index=test_ids).rank(pct=True)
            discordant = ((rna_rank - protein_rank).abs() >= 0.50).to_numpy()

            for variant, latent_name in variants.items():
                latent_train = latent_test = None
                if latent_name is not None:
                    latent_train, latent_test = latent_cache[latent_name]
                    latent_train = latent_train[train_keep.to_numpy()]
                    latent_test = latent_test[test_keep.to_numpy()]
                x_train = feature_block(
                    variant,
                    latent_train,
                    train_observed[train_keep.to_numpy()],
                    train_proxy,
                )
                x_test = feature_block(
                    variant,
                    latent_test,
                    test_observed[test_keep.to_numpy()],
                    test_proxy,
                )
                selected, inner_scores = select_head(x_train, y_train, train_groups)
                head = make_head(selected)
                head.fit(x_train, y_train)
                pred = head.predict(x_test)

                global_rho = safe_spearman(y_test, pred)
                discordant_rho = safe_spearman(y_test[discordant], pred[discordant])
                metric_rows.extend(
                    [
                        {"model": variant, "donor": donor, "target": target, "metric": "spearman_true_pred", "value": global_rho, "n_cells": len(y_test)},
                        {"model": variant, "donor": donor, "target": target, "metric": "discordant_spearman", "value": discordant_rho, "n_cells": int(discordant.sum())},
                    ]
                )
                selection_rows.append(
                    {
                        "model": variant,
                        "donor": donor,
                        "target": target,
                        "selected_head": selected,
                        **{f"inner_{name}_rho": value for name, value in inner_scores.items()},
                    }
                )
                if variant in {"totalvi_xmodal", "fusionvi_xmodal"}:
                    frame = pd.DataFrame(
                        {
                            "cell_id": test_ids,
                            "donor": donor,
                            "target": target,
                            "model": variant,
                            "true": y_test,
                            "prediction": pred,
                            "discordant": discordant,
                        }
                    )
                    prediction_rows.append(frame)
        print(f"[{fold_number}/{len(donors)}] completed {donor}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    selection = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(RESULTS / "cross_modal_head_metrics.csv", index=False)
    selection.to_csv(RESULTS / "cross_modal_head_selection.csv", index=False)
    predictions.to_csv(RESULTS / "cross_modal_head_predictions.csv", index=False)

    baseline = pd.read_csv(RESULTS / "cv_metrics.csv")
    baseline = baseline[
        (baseline["analysis"] == "heldout_marker")
        & baseline["metric"].isin(["spearman_true_pred", "discordant_spearman"])
        & baseline["model"].isin(["totalvi", "fusionvi"])
    ][["model", "donor", "target", "metric", "value"]]
    comparison = pd.concat([baseline, metrics], ignore_index=True)
    mean_table = comparison.groupby(["model", "target", "metric"], as_index=False)["value"].mean()
    mean_table.to_csv(RESULTS / "cross_modal_head_means.csv", index=False)

    paired_rows = []
    for model in ["totalvi_xmodal", "fusionvi_xmodal"]:
        for (target, metric), group in comparison[comparison["model"].isin(["totalvi", model])].groupby(["target", "metric"]):
            wide = group.pivot(index="donor", columns="model", values="value").dropna()
            diff = wide[model] - wide["totalvi"]
            p_value = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            rng = np.random.default_rng(SEED)
            boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10_000)])
            paired_rows.append(
                {
                    "model": model,
                    "target": target,
                    "metric": metric,
                    "n_donors": len(diff),
                    "totalvi_mean": wide["totalvi"].mean(),
                    "xmodal_mean": wide[model].mean(),
                    "mean_paired_difference": diff.mean(),
                    "difference_ci_low": np.quantile(boot, 0.025),
                    "difference_ci_high": np.quantile(boot, 0.975),
                    "wilcoxon_p": p_value,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired["bh_q"] = bh_adjust(paired["wilcoxon_p"].to_numpy())
    paired.to_csv(RESULTS / "cross_modal_head_paired.csv", index=False)
    (RESULTS / "cross_modal_head_summary.json").write_text(
        json.dumps(
            {
                "outer_folds": 10,
                "inner_group_folds": 3,
                "heldout_markers": hidden,
                "candidate_heads": list(HEADS),
                "selection_unit": "training donor",
            },
            indent=2,
        )
    )
    print(mean_table.to_string(index=False))
    print("\nPaired against standard totalVI\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
