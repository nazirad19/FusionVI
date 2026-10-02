"""Nested cross-modal PD-L1 prediction for unseen CRISPR targets.

Readout families are selected inside each outer training fold using CRISPR
target-grouped validation. Final performance is summarized at the
perturbation-by-replicate level, with held-out non-targeting cells providing
the reference in every fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ADATA = ROOT / "data" / "processed" / "papalexi_eccite.h5ad"
RESULTS = ROOT / "results" / "experiment2_papalexi"
FOLDS = RESULTS / "folds"


def load_config() -> dict:
    with (ROOT / "config" / "papalexi.yaml").open() as handle:
        return yaml.safe_load(handle)


def make_head(name: str, seed: int):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_features=0.8,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=seed,
        )
    raise KeyError(name)


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def select_head(x: np.ndarray, y: np.ndarray, groups: np.ndarray, candidates: list[str], seed: int):
    splitter = GroupKFold(n_splits=3)
    scores: dict[str, float] = {}
    for name in candidates:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x, y, groups):
            model = make_head(name, seed)
            model.fit(x[train_idx], y[train_idx])
            fold_scores.append(safe_spearman(y[valid_idx], model.predict(x[valid_idx])))
        scores[name] = float(np.nanmean(fold_scores))
    return max(scores, key=scores.get), scores


def feature_block(variant: str, latent: np.ndarray | None, protein: np.ndarray, rna: np.ndarray) -> np.ndarray:
    if variant == "rna_proxy_only":
        return rna[:, None]
    if variant == "protein_context_only":
        return protein
    if variant == "context_no_latent":
        return np.column_stack([protein, rna])
    if latent is None:
        raise ValueError(f"{variant} requires a latent representation")
    return np.column_stack([latent, protein, rna])


def add_effects(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["model", "fold", "gene", "replicate"], as_index=False)
        .agg(true=("true", "mean"), prediction=("prediction", "mean"), n_cells=("true", "size"))
    )
    controls = grouped[grouped["gene"] == "NT"].set_index(["model", "fold", "replicate"])
    treated = grouped[grouped["gene"] != "NT"].copy()
    index = pd.MultiIndex.from_frame(treated[["model", "fold", "replicate"]])
    treated["true_control"] = controls.loc[index, "true"].to_numpy()
    treated["prediction_control"] = controls.loc[index, "prediction"].to_numpy()
    treated["true_effect"] = treated["true"] - treated["true_control"]
    treated["predicted_effect"] = treated["prediction"] - treated["prediction_control"]
    treated["correct_direction"] = np.sign(treated["true_effect"]) == np.sign(treated["predicted_effect"])
    return treated


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project_seed"])
    target = str(cfg["heldout_protein"])
    proxy_gene = str(cfg["rna_proxy"])
    candidates = list(cfg["head_candidates"])
    adata = ad.read_h5ad(ADATA)
    protein = adata.obsm["protein_counts"].astype(float)
    observed_names = [name for name in protein.columns if name != target]
    observed = np.log1p(protein[observed_names])
    rna = pd.Series(
        np.log1p(np.asarray(adata[:, proxy_gene].layers["counts"].todense()).reshape(-1)),
        index=adata.obs_names,
    )
    y = pd.Series(np.log1p(protein[target].to_numpy()), index=adata.obs_names)

    variants = {
        "rna_proxy_only": None,
        "protein_context_only": None,
        "context_no_latent": None,
        "totalvi_xmodal": "totalvi",
        "fusionvi_xmodal": "fusionvi",
    }
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    rna_effect_rows: list[pd.DataFrame] = []

    for fold in range(int(cfg["n_outer_folds"])):
        reference_dir = FOLDS / f"totalvi__fold{fold}__seed{seed}"
        train_meta = pd.read_csv(reference_dir / "train_metadata.csv", index_col=0)
        test_meta = pd.read_csv(reference_dir / "test_metadata.csv", index_col=0)
        train_ids, test_ids = train_meta.index, test_meta.index
        train_protein = observed.loc[train_ids].to_numpy()
        test_protein = observed.loc[test_ids].to_numpy()
        train_rna = rna.loc[train_ids].to_numpy()
        test_rna = rna.loc[test_ids].to_numpy()
        y_train = y.loc[train_ids].to_numpy()
        y_test = y.loc[test_ids].to_numpy()
        groups = train_meta["gene"].astype(str).to_numpy()

        rna_frame = test_meta[["gene", "replicate"]].copy()
        rna_frame["cd274_rna"] = test_rna
        rna_grouped = rna_frame.groupby(["gene", "replicate"], as_index=False)["cd274_rna"].mean()
        rna_control = rna_grouped[rna_grouped["gene"] == "NT"].set_index("replicate")["cd274_rna"]
        rna_treated = rna_grouped[rna_grouped["gene"] != "NT"].copy()
        rna_treated["cd274_rna_effect"] = rna_treated["cd274_rna"] - rna_control.loc[rna_treated["replicate"]].to_numpy()
        rna_treated["fold"] = fold
        rna_effect_rows.append(rna_treated[["fold", "gene", "replicate", "cd274_rna_effect"]])
        latent_cache = {
            name: (
                np.load(FOLDS / f"{name}__fold{fold}__seed{seed}" / "latent_train.npy"),
                np.load(FOLDS / f"{name}__fold{fold}__seed{seed}" / "latent_test.npy"),
            )
            for name in ("totalvi", "fusionvi")
        }

        for variant, latent_name in variants.items():
            latent_train = latent_test = None
            if latent_name is not None:
                latent_train, latent_test = latent_cache[latent_name]
            x_train = feature_block(variant, latent_train, train_protein, train_rna)
            x_test = feature_block(variant, latent_test, test_protein, test_rna)
            selected, inner_scores = select_head(x_train, y_train, groups, candidates, seed)
            head = make_head(selected, seed)
            head.fit(x_train, y_train)
            pred = head.predict(x_test)
            selection_rows.append(
                {
                    "fold": fold,
                    "model": variant,
                    "selected_head": selected,
                    **{f"inner_{name}_rho": value for name, value in inner_scores.items()},
                }
            )
            output = test_meta[["gene", "guide", "replicate"]].copy()
            output["model"] = variant
            output["fold"] = fold
            output["true"] = y_test
            output["prediction"] = pred
            prediction_rows.append(output.reset_index(names="cell_id"))

        for name in ("totalvi", "fusionvi"):
            native = pd.read_csv(FOLDS / f"{name}__fold{fold}__seed{seed}" / "native_predictions.csv", index_col=0)
            output = native[["gene", "guide", "replicate"]].copy()
            output["model"] = name
            output["fold"] = fold
            output["true"] = np.log1p(native["true_PDL1"].to_numpy())
            output["prediction"] = native["pred_PDL1"].to_numpy()
            prediction_rows.append(output.reset_index(names="cell_id"))
        print(f"Evaluated fold {fold}", flush=True)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    effects = add_effects(predictions)
    rna_effects = pd.concat(rna_effect_rows, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    selections.to_csv(RESULTS / "head_selection.csv", index=False)
    effects.to_csv(RESULTS / "perturbation_replicate_predictions.csv", index=False)
    rna_effects.to_csv(RESULTS / "cd274_rna_effects.csv", index=False)

    metric_rows = []
    for model, frame in effects.groupby("model"):
        # Native totalVI decoder output is a normalized abundance on a
        # different scale from log1p ADT counts. Rank and direction metrics are
        # valid, whereas an uncalibrated absolute error is not comparable to
        # the supervised readouts.
        comparable_scale = model not in {"totalvi", "fusionvi"}
        metric_rows.extend(
            [
                {"model": model, "metric": "effect_spearman", "value": safe_spearman(frame["true_effect"].to_numpy(), frame["predicted_effect"].to_numpy()), "n": len(frame)},
                {"model": model, "metric": "effect_mae", "value": float(np.mean(np.abs(frame["true_effect"] - frame["predicted_effect"]))) if comparable_scale else np.nan, "n": len(frame)},
                {"model": model, "metric": "direction_accuracy", "value": float(frame["correct_direction"].mean()), "n": len(frame)},
            ]
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "summary_metrics.csv", index=False)

    gene_effects = effects.groupby(["model", "gene"], as_index=False)[["true_effect", "predicted_effect"]].mean()
    gene_effects["absolute_error"] = np.abs(gene_effects["true_effect"] - gene_effects["predicted_effect"])
    gene_effects.loc[gene_effects["model"].isin(["totalvi", "fusionvi"]), "absolute_error"] = np.nan
    gene_effects.to_csv(RESULTS / "gene_level_effects.csv", index=False)

    observed_gene = (
        gene_effects[gene_effects["model"] == "fusionvi_xmodal"]
        .set_index("gene")["true_effect"]
        .rename("pdl1_protein_effect")
    )
    prediction_wide = gene_effects.pivot(index="gene", columns="model", values="predicted_effect")
    biological = pd.concat(
        [
            observed_gene,
            rna_effects.groupby("gene")["cd274_rna_effect"].mean(),
            prediction_wide[["totalvi_xmodal", "fusionvi_xmodal"]].rename(
                columns={
                    "totalvi_xmodal": "totalvi_x_predicted_effect",
                    "fusionvi_xmodal": "fusionvi_x_predicted_effect",
                }
            ),
        ],
        axis=1,
    ).reset_index()
    biological["rna_protein_opposite_direction"] = (
        np.sign(biological["cd274_rna_effect"]) != np.sign(biological["pdl1_protein_effect"])
    ) & (biological["cd274_rna_effect"].abs() >= 0.02) & (biological["pdl1_protein_effect"].abs() >= 0.10)
    biological.to_csv(RESULTS / "biological_effects.csv", index=False)
    baseline_name = "totalvi_xmodal"
    baseline = gene_effects[gene_effects["model"] == baseline_name].set_index("gene")["absolute_error"]
    paired_rows = []
    for model in [name for name in gene_effects["model"].unique() if name not in {baseline_name, "totalvi", "fusionvi"}]:
        candidate = gene_effects[gene_effects["model"] == model].set_index("gene")["absolute_error"]
        common = baseline.index.intersection(candidate.index)
        difference = baseline.loc[common] - candidate.loc[common]
        statistic, p_value = wilcoxon(difference) if np.any(difference != 0) else (0.0, 1.0)
        paired_rows.append(
            {
                "comparison": f"{model} vs {baseline_name}",
                "median_absolute_error_reduction": float(np.median(difference)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
                "n_targets": int(len(common)),
            }
        )
    pd.DataFrame(paired_rows).to_csv(RESULTS / "paired_tests.csv", index=False)

    summary = {
        "design": "five-fold leave-CRISPR-targets-out with held-out NT controls",
        "targets": int(effects["gene"].nunique()),
        "replicate_target_effects": int(len(effects) // effects["model"].nunique()),
        "best_effect_spearman": metrics.loc[metrics["metric"] == "effect_spearman"].sort_values("value", ascending=False).iloc[0].to_dict(),
    }
    (RESULTS / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print(metrics.pivot(index="model", columns="metric", values="value").round(4), flush=True)


if __name__ == "__main__":
    main()
