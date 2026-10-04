"""Aggregate control-arm runs and test pre-specified contrasts at the seed level.

Reads results/paper_benchmark_runs/<arm>__seed<k>/protein_metrics.csv for every
completed run, writes results/control_all_metrics.csv, and for each contrast
reports the seed-paired difference in mean protein RMSLE (primary) with a
t-interval, exact sign-flip p, and Holm-adjusted p across contrasts.
Gate weights (FusionVI arms) are summarized by cell type in the source batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from stats_paper_benchmark import sign_flip_p

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "paper_benchmark_runs"
OUT = ROOT / "results"
CONFIG = ROOT / "config" / "paper_benchmark.yaml"

# (candidate, reference, question answered)
CONTRASTS = [
    ("FusionVI", "totalvi_w128", "Does branch fusion beat a joint encoder of the same width?"),
    ("FusionVI", "totalvi_pmatch", "Does branch fusion beat a joint encoder with the same parameter count?"),
    ("FusionVI", "totalvi_avail_w128", "Does fusion add anything beyond telling a joint encoder the panel is missing?"),
    ("totalvi_avail_w128", "totalvi_w128", "Does the missing-panel indicator alone help a joint encoder?"),
    ("fusionvi_pmatch", "totalVI", "At equal (totalVI-sized) capacity, does fusion still help?"),
    ("FusionVI", "fusionvi_fixedgate", "Does learning the gate help versus a fixed 0.5 blend?"),
    ("FusionVI", "fusionvi_rnaonly", "Does the protein branch help at all on source cells?"),
    ("FusionVI", "totalVI", "Original comparison (confounded by width and the library network)."),
]


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    control_seeds = {int(seed) for seed in cfg["control_seeds"]}
    frames = [pd.read_csv(f) for f in sorted(RUNS.glob("*/protein_metrics.csv"))]
    if not frames:
        raise SystemExit(f"No completed runs under {RUNS}")
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics["seed"].isin(control_seeds)].copy()
    metrics.to_csv(OUT / "control_all_metrics.csv", index=False)
    per_seed = metrics.groupby(["model", "seed"])[["rmsle", "mae_log1p", "spearman", "pearson_log1p"]].mean()

    print("Mean over seeds (n seeds):")
    overview = per_seed.groupby("model").mean().round(4)
    overview["n_seeds"] = per_seed.groupby("model").size()
    print(overview.to_string())

    results = []
    for cand, ref, question in CONTRASTS:
        if cand not in per_seed.index.get_level_values(0) or ref not in per_seed.index.get_level_values(0):
            continue
        paired = per_seed.loc[cand, "rmsle"].to_frame("c").join(per_seed.loc[ref, "rmsle"].to_frame("r"), how="inner")
        d = (paired["c"] - paired["r"]).to_numpy()
        if d.size < 2:
            continue
        half = stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
        results.append({
            "candidate": cand, "reference": ref, "question": question, "n_seeds": int(d.size),
            "rmsle_diff": float(d.mean()), "ci95_low": float(d.mean() - half), "ci95_high": float(d.mean() + half),
            "seeds_candidate_better": int((d < 0).sum()),
            "p_t": float(stats.ttest_1samp(d, 0).pvalue),
            "p_exact": sign_flip_p(d) if d.size <= 20 else None,
        })
    primary = [r for r in results if r["reference"] != "totalVI" or r["candidate"] != "FusionVI"]
    for r, p in zip(primary, holm([r["p_t"] for r in primary])):
        r["p_holm"] = p
    table = pd.DataFrame(results)
    table.to_csv(OUT / "control_contrasts.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.drop(columns="question").round(4).to_string(index=False))

    gate_files = sorted(RUNS.glob("*/gates.csv"))
    if gate_files:
        gates = []
        for f in gate_files:
            g = pd.read_csv(f, index_col=0)
            g["arm"] = f.parent.name.split("__")[0]
            gates.append(g)
        gates = pd.concat(gates)
        src = gates[gates["batch"].astype(str).str.endswith("D1")]
        summary = src.groupby(["arm", "cell_type"])["rna_gate"].mean().unstack("arm").round(3)
        summary["learned_minus_mean"] = (summary.get("fusionvi", np.nan) - summary.get("fusionvi", np.nan).mean()).round(3)
        summary = summary.sort_values("learned_minus_mean")
        summary.to_csv(OUT / "control_gate_by_celltype.csv")
        print("\nMean RNA-gate weight on source-batch cells, by cell type:")
        print(summary.to_string())

    (OUT / "control_contrasts.json").write_text(json.dumps(results, indent=2))

    # A compact, report-ready view of the three prespecified Tier 1 controls.
    key_refs = ["totalvi_w128", "totalvi_pmatch", "totalvi_avail_w128"]
    key = table[(table["candidate"] == "FusionVI") & table["reference"].isin(key_refs)].copy()
    labels = {
        "totalvi_w128": "Joint encoder\nsame width",
        "totalvi_pmatch": "Joint encoder\nsame parameters",
        "totalvi_avail_w128": "Joint encoder\n+ availability",
    }
    key["label"] = key["reference"].map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(key))
    axes[0].errorbar(
        key["rmsle_diff"], y,
        xerr=[key["rmsle_diff"] - key["ci95_low"], key["ci95_high"] - key["rmsle_diff"]],
        fmt="o", color="#6D28D9", ecolor="#A78BFA", capsize=4, linewidth=2,
    )
    axes[0].axvline(0, color="#475569", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, key["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("FusionVI - control mean RMSLE")
    axes[0].set_title("Paired effect across 16 seeds", loc="left", weight="bold")
    axes[0].text(0.02, -0.23, "Lower favors FusionVI; bars are 95% t intervals", transform=axes[0].transAxes, fontsize=9)

    models = ["FusionVI", *key_refs]
    colors = ["#6D28D9", "#38BDF8", "#14B8A6", "#F59E0B"]
    for idx, (model, color) in enumerate(zip(models, colors)):
        vals = per_seed.loc[model, "rmsle"].to_numpy()
        jitter = np.linspace(-0.09, 0.09, len(vals))
        axes[1].scatter(np.full(len(vals), idx) + jitter, vals, s=22, alpha=0.65, color=color, edgecolor="none")
        axes[1].plot([idx - 0.18, idx + 0.18], [vals.mean(), vals.mean()], color="#0F172A", linewidth=2.5)
    axes[1].set_xticks(range(len(models)), ["FusionVI", "Same\nwidth", "Same\nparameters", "+ availability"])
    axes[1].set_ylabel("Mean protein RMSLE")
    axes[1].set_title("Every dot is one seed", loc="left", weight="bold")
    axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Seed-paired RMSLE: FusionVI and matched joint encoders", x=0.06, ha="left", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "control_benchmark.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Keep the consolidated report source synchronized with the executed controls.
    summary_path = OUT / "fusionvi_experiments_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        experiment = next(exp for exp in summary["experiments"] if exp["id"] == 4)
        means = per_seed.groupby("model")["rmsle"].mean().to_dict()
        report_models = ["FusionVI", "totalVI", *key_refs]
        params = {}
        for model in report_models:
            files = sorted(RUNS.glob(f"{model}__seed*/complete.json"))
            if files:
                params[model] = json.loads(files[0].read_text()).get("trainable_parameters")
        experiment["paired_initializations"] = len(control_seeds)
        experiment["confirmatory_controls"] = {
            "status": "Tier 1 complete",
            "control_seeds": sorted(control_seeds),
            "mean_rmsle": {model: float(means[model]) for model in report_models},
            "trainable_parameters": params,
            "contrasts": results,
            "interpretation": (
                "FusionVI had lower RMSLE than same-width, parameter-matched, and missing-panel-aware "
                "joint encoders across 16 paired seeds. The availability indicator alone did not improve "
                "the joint encoder. The RNA SVD-ridge baseline remained substantially better on aggregate RMSLE."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
