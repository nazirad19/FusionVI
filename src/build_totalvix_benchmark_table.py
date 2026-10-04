"""Build the focused totalVI protein-efficiency readout audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ["rmsle", "abs_bias", "residual_sd", "spearman", "pearson", "within_celltype_spearman"]


def main() -> None:
    seeds = pd.read_csv(RESULTS / "totalvix_paper_seed_summary.csv")
    rows = []
    neural_rows = (
        ("totalVI helper-uncorrected", "totalVI helper output (efficiency omitted)"),
        ("totalVI likelihood-consistent", "Likelihood-consistent totalVI"),
        ("D1 affine head", "D1 affine head"),
        ("totalVI likelihood-consistent + D1 head", "Likelihood-consistent totalVI + D1 head"),
    )
    for readout, method in neural_rows:
        frame = seeds[seeds["readout"] == readout]
        row = {"method": method, "n_runs": len(frame), "source": "16 saved updated-library totalVI initializations"}
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            row[f"{metric}_ci95_low"] = float(values.mean() - half)
            row[f"{metric}_ci95_high"] = float(values.mean() + half)
        rows.append(row)

    baselines = pd.read_csv(RESULTS / "method_comparison_runs_totalvix_baselines.csv")
    choices = [
        ("seurat_v3_official", "raw", "Official Seurat v3"),
        ("rna_ridge", "calibrated", "RNA ridge + D1 calibration"),
        ("knn", "calibrated", "RNA kNN + D1 calibration"),
    ]
    for method_key, scoring, label in choices:
        candidates = baselines[(baselines["method"] == method_key) & (baselines["scoring"] == scoring)]
        if method_key == "knn":
            candidates = candidates[candidates["readout"] == "log"]
        if len(candidates) != 1:
            raise SystemExit(f"Expected one row for {method_key}/{scoring}, found {len(candidates)}")
        source = "authors' published target prediction" if method_key == "seurat_v3_official" else "deterministic source-trained baseline"
        row = {"method": label, "n_runs": 1, "source": source}
        src = candidates.iloc[0]
        for metric in METRICS:
            key = "within_ct_spearman" if metric == "within_celltype_spearman" else metric
            row[metric] = float(src[key])
        rows.append(row)

    out = pd.DataFrame(rows)
    out["rmsle_rank"] = out["rmsle"].rank(method="min").astype(int)
    ranked = out.sort_values("rmsle_rank")
    ranked.to_csv(RESULTS / "totalvi_efficiency_method_benchmark.csv", index=False)
    ranked.to_csv(RESULTS / "totalvix_method_benchmark.csv", index=False)
    reference = out[out["method"] == "Likelihood-consistent totalVI"].iloc[0]
    comparison = out[out["method"] != "Likelihood-consistent totalVI"].copy()
    comparison["reference_rmsle"] = reference["rmsle"]
    comparison["rmsle_method_minus_reference"] = comparison["rmsle"] - reference["rmsle"]
    comparison["method_better_rmsle"] = comparison["rmsle_method_minus_reference"] < 0
    comparison["reference_within_celltype_spearman"] = reference["within_celltype_spearman"]
    comparison["within_ct_method_minus_reference"] = (
        comparison["within_celltype_spearman"] - reference["within_celltype_spearman"]
    )
    comparison.to_csv(RESULTS / "totalvi_efficiency_vs_methods.csv", index=False)
    (RESULTS / "totalvix_method_benchmark.json").write_text(json.dumps({
        "primary_metric": "mean per-protein RMSLE on held-out SLN111-D2 protein counts",
        "best_method": out.loc[out["rmsle"].idxmin(), "method"],
        "conclusion": "Restoring the model-owned protein efficiency outperforms the D1 affine head; the affine-head improvement claim is rejected.",
        "seurat_scope": "raw target prediction from the authors' repository; no D1 cross-fitted calibration",
        "table": out.sort_values("rmsle_rank").to_dict(orient="records"),
    }, indent=2))
    (RESULTS / "totalvi_efficiency_method_benchmark.json").write_text(
        (RESULTS / "totalvix_method_benchmark.json").read_text()
    )
    print(out.sort_values("rmsle_rank")[["method", "n_runs", "rmsle", "spearman", "within_celltype_spearman"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
