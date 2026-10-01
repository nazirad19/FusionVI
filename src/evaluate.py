"""Evaluate donor-held-out totalVI and FusionVI experiments.

The primary analyses avoid cell-level significance testing. Metrics are first
computed independently in each held-out donor, then models are compared with
paired donor-level statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "folds"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 2026

CONTRASTS = {
    "T_cell_direct": {"cell_types": ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem"], "positive": "CD3_CD28"},
    "Monocyte_direct": {"cell_types": ["CD14_Mono"], "positive": "LPS"},
    "NK_indirect": {"cell_types": ["NK"], "positive": "CD3_CD28"},
    "B_cell_indirect": {"cell_types": ["B"], "positive": "CD3_CD28"},
}

MARKERS = {
    "CD25_T": {"marker": "CD25", "cell_types": ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem"], "positive": "CD3_CD28"},
    "CD69_T": {"marker": "CD69", "cell_types": ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem"], "positive": "CD3_CD28"},
    "HLA_DR_Monocyte": {"marker": "HLA-DR", "cell_types": ["CD14_Mono"], "positive": "LPS"},
}

RNA_PROXY_IDS = {
    "CD25": ["ENSG00000134460"],  # IL2RA
    "CD69": ["ENSG00000110848"],  # CD69
    "HLA-DR": ["ENSG00000204287", "ENSG00000196126"],  # HLA-DRA, HLA-DRB1
}


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def marker_metrics(frame: pd.DataFrame, spec: dict, rna_proxy: pd.Series | None = None) -> dict[str, float]:
    keep = frame["cell_type"].isin(spec["cell_types"]) & frame["condition"].isin(["Baseline", spec["positive"]])
    sub = frame.loc[keep]
    marker = spec["marker"]
    y = (sub["condition"] == spec["positive"]).astype(int).to_numpy()
    pred = sub[f"pred_{marker}"].to_numpy()
    true = sub[f"true_{marker}"].to_numpy()
    rho = spearmanr(true, pred).statistic if np.std(true) > 0 and np.std(pred) > 0 else np.nan
    pred_auc = safe_auc(y, pred)
    true_auc = safe_auc(y, true)
    output = {
        "n_cells": len(sub),
        "pred_auc": pred_auc,
        "pred_ap": float(average_precision_score(y, pred)),
        "true_auc": true_auc,
        "effect_concordance": 1.0 - abs(pred_auc - true_auc),
        "spearman_true_pred": float(rho),
    }
    if rna_proxy is not None:
        proxy = rna_proxy.reindex(sub.index)
        rna_rank = proxy.rank(pct=True)
        protein_rank = pd.Series(true, index=sub.index).rank(pct=True)
        discordant = (rna_rank - protein_rank).abs() >= 0.5
        discordant_pred = pd.Series(pred, index=sub.index).loc[discordant]
        discordant_true = pd.Series(true, index=sub.index).loc[discordant]
        discordant_rho = (
            spearmanr(discordant_true, discordant_pred).statistic
            if len(discordant_true) >= 10 and discordant_true.std() > 0 and discordant_pred.std() > 0
            else np.nan
        )
        output["discordant_n"] = int(discordant.sum())
        output["discordant_spearman"] = float(discordant_rho)
    return output


def build_rna_proxies(adata) -> dict[str, pd.Series]:
    proxies = {}
    for marker, ids in RNA_PROXY_IDS.items():
        missing = set(ids).difference(adata.var_names)
        if missing:
            raise RuntimeError(f"RNA proxy genes absent for {marker}: {sorted(missing)}")
        values = np.asarray(adata[:, ids].layers["counts"].sum(axis=1)).reshape(-1)
        proxies[marker] = pd.Series(np.log1p(values), index=adata.obs_names)
    return proxies


def latent_metrics(z_train: np.ndarray, train_meta: pd.DataFrame, z_test: np.ndarray, test_meta: pd.DataFrame, spec: dict) -> dict[str, float]:
    train_keep = train_meta["cell_type"].isin(spec["cell_types"]) & train_meta["condition"].isin(["Baseline", spec["positive"]])
    test_keep = test_meta["cell_type"].isin(spec["cell_types"]) & test_meta["condition"].isin(["Baseline", spec["positive"]])
    x_train, x_test = z_train[train_keep.to_numpy()], z_test[test_keep.to_numpy()]
    y_train = (train_meta.loc[train_keep, "condition"] == spec["positive"]).astype(int).to_numpy()
    y_test = (test_meta.loc[test_keep, "condition"] == spec["positive"]).astype(int).to_numpy()
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    classifier.fit(x_train, y_train)
    score = classifier.predict_proba(x_test)[:, 1]
    prediction = (score >= 0.5).astype(int)
    return {
        "n_cells": len(y_test),
        "auc": safe_auc(y_test, score),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
    }


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return pd.Series(output, index=p_values.index)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (analysis, target, metric), group in metrics.groupby(["analysis", "target", "metric"]):
        if metric in {"n_cells", "discordant_n", "true_auc"}:
            continue
        wide = group.pivot(index="donor", columns="model", values="value").dropna()
        if not {"totalvi", "fusionvi"}.issubset(wide.columns):
            continue
        diff = wide["fusionvi"] - wide["totalvi"]
        try:
            p_value = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            p_value = np.nan
        rng = np.random.default_rng(SEED)
        boot = np.array([rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean() for _ in range(10000)])
        rows.append(
            {
                "analysis": analysis,
                "target": target,
                "metric": metric,
                "n_donors": len(diff),
                "totalvi_mean": wide["totalvi"].mean(),
                "fusionvi_mean": wide["fusionvi"].mean(),
                "mean_paired_difference": diff.mean(),
                "difference_ci_low": np.quantile(boot, 0.025),
                "difference_ci_high": np.quantile(boot, 0.975),
                "wilcoxon_p": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q"] = bh_adjust(summary["wilcoxon_p"].fillna(1.0))
    return summary


def make_figures(metrics: pd.DataFrame, gates: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"totalvi": "#5267D9", "fusionvi": "#F15A59"}

    selected = metrics[
        ((metrics["analysis"] == "latent_response") & (metrics["metric"] == "auc"))
        | ((metrics["analysis"] == "heldout_marker") & (metrics["metric"] == "pred_auc"))
    ].copy()
    selected["label"] = np.where(
        selected["analysis"] == "latent_response",
        "Latent: " + selected["target"].str.replace("_", " "),
        "Held-out: " + selected["target"].str.replace("_", " "),
    )
    labels = list(dict.fromkeys(selected["label"]))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 5.4), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = selected[selected["label"] == label]
        wide = sub.pivot(index="donor", columns="model", values="value").dropna()
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row["totalvi"], row["fusionvi"]], color="#C7CAD5", linewidth=1.2, zorder=1)
        for x, model in enumerate(["totalvi", "fusionvi"]):
            ax.scatter(np.repeat(x, len(wide)), wide[model], s=45, color=palette[model], edgecolor="white", zorder=2)
        ax.set_xticks([0, 1], ["totalVI", "FusionVI"], rotation=20)
        ax.set_title(label, fontsize=12)
        ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Held-out donor ROC AUC")
    fig.suptitle("Biological response recovery in unseen donors", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_auc_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not gates.empty:
        order = ["CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem", "B", "NK", "CD14_Mono"]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=gates, x="cell_type", y="protein_gate", hue="condition", order=order, showfliers=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Protein branch weight (1 − RNA gate)")
        ax.set_title("FusionVI adapts modality use by cell and stimulation context", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Condition", frameon=True, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        fig.tight_layout()
        fig.savefig(FIGURES / "fusion_gate_by_cell_condition.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    import scanpy as sc

    adata = sc.read_h5ad(ROOT / "data" / "processed" / "lawlor_pbmc_citeseq.h5ad")
    rna_proxies = build_rna_proxies(adata)
    metric_rows = []
    gate_frames = []
    gate_discordance_rows = []
    for model_name in ["totalvi", "fusionvi"]:
        for fold in sorted(FOLDS.glob(f"{model_name}__*__seed{SEED}")):
            complete_path = fold / "complete.json"
            if not complete_path.exists():
                continue
            info = json.loads(complete_path.read_text())
            donor = info["heldout_donor"]
            predictions = pd.read_csv(fold / "masked_protein_predictions.csv", index_col=0)
            for target, spec in MARKERS.items():
                for metric, value in marker_metrics(predictions, spec, rna_proxies[spec["marker"]]).items():
                    metric_rows.append({"model": model_name, "donor": donor, "analysis": "heldout_marker", "target": target, "metric": metric, "value": value})

            train_meta = pd.read_csv(fold / "train_metadata.csv", index_col=0)
            test_meta = pd.read_csv(fold / "test_metadata.csv", index_col=0)
            z_train = np.load(fold / "latent_train.npy")
            z_test = np.load(fold / "latent_test.npy")
            for target, spec in CONTRASTS.items():
                for metric, value in latent_metrics(z_train, train_meta, z_test, test_meta, spec).items():
                    metric_rows.append({"model": model_name, "donor": donor, "analysis": "latent_response", "target": target, "metric": metric, "value": value})

            gate_path = fold / "test_fusion_gates.csv"
            if model_name == "fusionvi" and gate_path.exists():
                gate_frame = pd.read_csv(gate_path, index_col=0)
                gate_frames.append(gate_frame)
                for target, spec in MARKERS.items():
                    keep = gate_frame["cell_type"].isin(spec["cell_types"]) & gate_frame["condition"].isin(["Baseline", spec["positive"]])
                    indices = gate_frame.index[keep]
                    rna_rank = rna_proxies[spec["marker"]].reindex(indices).rank(pct=True)
                    protein_rank = predictions.loc[indices, f"true_{spec['marker']}"].rank(pct=True)
                    discordant = (rna_rank - protein_rank).abs() >= 0.5
                    for label, selector in [("discordant", discordant), ("concordant", ~discordant)]:
                        values = gate_frame.loc[indices, "protein_gate"].loc[selector]
                        gate_discordance_rows.append(
                            {"donor": donor, "target": target, "stratum": label, "n_cells": len(values), "mean_protein_gate": values.mean()}
                        )

    metrics = pd.DataFrame(metric_rows)
    expected = 2 * 10
    observed = metrics[["model", "donor"]].drop_duplicates().shape[0] if not metrics.empty else 0
    if observed != expected:
        raise RuntimeError(f"Expected {expected} completed model-folds, found {observed}")
    baseline_path = RESULTS / "classical_baseline_metrics.csv"
    if baseline_path.exists():
        metrics = pd.concat([metrics, pd.read_csv(baseline_path)], ignore_index=True)
    metrics.to_csv(RESULTS / "cv_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(RESULTS / "cv_summary.csv", index=False)

    gates = pd.concat(gate_frames) if gate_frames else pd.DataFrame()
    if not gates.empty:
        gates.to_csv(RESULTS / "fusion_gates_all_cells.csv")
        gate_summary = gates.groupby(["cell_type", "condition"])[["rna_gate", "protein_gate"]].agg(["count", "mean", "median", "std"])
        gate_summary.to_csv(RESULTS / "fusion_gate_summary.csv")
    gate_discordance = pd.DataFrame(gate_discordance_rows)
    gate_headline = []
    if not gate_discordance.empty:
        gate_discordance.to_csv(RESULTS / "gate_discordance_by_donor.csv", index=False)
        for target, group in gate_discordance.groupby("target"):
            wide = group.pivot(index="donor", columns="stratum", values="mean_protein_gate").dropna()
            diff = wide["discordant"] - wide["concordant"]
            p_value = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            gate_headline.append(
                {"target": target, "n_donors": len(diff), "discordant_minus_concordant_protein_gate": diff.mean(), "wilcoxon_p": p_value}
            )
        pd.DataFrame(gate_headline).to_csv(RESULTS / "gate_discordance_summary.csv", index=False)
    make_figures(metrics, gates)

    primary = summary[
        ((summary["analysis"] == "heldout_marker") & summary["metric"].isin(["effect_concordance", "spearman_true_pred", "discordant_spearman"]))
        | ((summary["analysis"] == "latent_response") & (summary["metric"] == "auc"))
    ]
    headline = {
        "seed": SEED,
        "folds_per_model": 10,
        "unit_of_replication": "held-out donor",
        "primary_results": primary.to_dict(orient="records"),
        "gate_discordance": gate_headline,
    }
    (RESULTS / "headline_results.json").write_text(json.dumps(headline, indent=2))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
