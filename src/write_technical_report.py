"""Update Experiment 4 in the consolidated report from protected outputs.

The rest of TECHNICAL_REPORT.md contains manually curated Experiments 1–3 and
5. This updater deliberately replaces only Experiment 4 so those sections are
never lost when the likelihood-consistent benchmark is regenerated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "TECHNICAL_REPORT.md"
TAG = "likelihood_correct"


def main() -> None:
    summary_path = ROOT / "results" / f"calibration_summary_{TAG}.csv"
    contrast_path = ROOT / "results" / f"calibration_contrasts_{TAG}.csv"
    if not summary_path.exists() or not contrast_path.exists():
        raise FileNotFoundError("Run evaluate_calibration.py with --tag likelihood_correct first")

    summary = pd.read_csv(summary_path).set_index(["arm", "readout"])
    contrasts = pd.read_csv(contrast_path)
    required = {"likelihood_mean", "likelihood_pred_log", "calibrated", "helper_mean"}
    missing = required - set(summary.index.get_level_values("readout"))
    if missing:
        raise ValueError(f"Missing required readouts: {sorted(missing)}")

    def value(arm: str, metric: str, readout: str = "likelihood_mean") -> float:
        return float(summary.loc[(arm, readout), metric])

    def paired(candidate: str, reference: str, metric: str) -> dict:
        rows = contrasts[
            (contrasts["readout"] == "likelihood_mean")
            & (contrasts["metric"] == metric)
            & (contrasts["candidate"] == candidate)
            & (contrasts["reference"] == reference)
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one {metric} contrast for {candidate} vs {reference}; found {len(rows)}")
        return rows.iloc[0].to_dict()

    rmsle_tv = paired("fusionvi", "totalvi", "rmsle")
    rmsle_width = paired("fusionvi", "totalvi_w128", "rmsle")
    rmsle_pmatch = paired("fusionvi", "totalvi_pmatch", "rmsle")
    rmsle_avail = paired("fusionvi", "totalvi_avail_w128", "rmsle")
    availability = paired("totalvi_avail_w128", "totalvi_w128", "rmsle")
    spearman_tv = paired("fusionvi", "totalvi", "spearman")
    within_tv = paired("fusionvi", "totalvi", "within_celltype_spearman")
    within_width = paired("fusionvi", "totalvi_w128", "within_celltype_spearman")

    section = f"""## Experiment 4 paper-aligned complete missing panel

This experiment follows the totalVI paper's Figure 3 missing-protein test. SLN111-D1 retained RNA and all 110 proteins, while every D2 protein was hidden during training and preserved only for scoring. All neural models used the same 20-dimensional latent space, totalVI decoder and likelihoods, optimizer, split, 500-epoch budget and 25 posterior samples. Sixteen paired seeds were evaluated; the paper used 30 initializations.

The primary output is the likelihood-consistent expected protein count from `py_norm_`, including each checkpoint's learned per-protein efficiency. The efficiency-omitting helper output is retained only as an audit and is excluded from encoder claims.

| Encoder | Mean RMSLE | Overall Spearman | Within-cell-type Spearman |
|---|---:|---:|---:|
| Published totalVI | **{value('totalvi', 'rmsle'):.4f}** | {value('totalvi', 'spearman'):.4f} | {value('totalvi', 'within_celltype_spearman'):.4f} |
| Joint, same width | {value('totalvi_w128', 'rmsle'):.4f} | **{value('totalvi_w128', 'spearman'):.4f}** | **{value('totalvi_w128', 'within_celltype_spearman'):.4f}** |
| Joint plus availability | {value('totalvi_avail_w128', 'rmsle'):.4f} | {value('totalvi_avail_w128', 'spearman'):.4f} | {value('totalvi_avail_w128', 'within_celltype_spearman'):.4f} |
| **FusionVI** | {value('fusionvi', 'rmsle'):.4f} | {value('fusionvi', 'spearman'):.4f} | {value('fusionvi', 'within_celltype_spearman'):.4f} |
| Joint, parameter matched | {value('totalvi_pmatch', 'rmsle'):.4f} | {value('totalvi_pmatch', 'spearman'):.4f} | {value('totalvi_pmatch', 'within_celltype_spearman'):.4f} |

FusionVI was worse than published totalVI by {rmsle_tv['candidate_minus_ref']:+.4f} RMSLE (95% CI {rmsle_tv['ci95_low']:+.4f} to {rmsle_tv['ci95_high']:+.4f}; {int(rmsle_tv['seeds_candidate_better'])}/16 seed wins; Holm-adjusted p={rmsle_tv['p_holm_within_readout_metric']:.2g}). It was also worse than the same-width joint encoder by {rmsle_width['candidate_minus_ref']:+.4f} (95% CI {rmsle_width['ci95_low']:+.4f} to {rmsle_width['ci95_high']:+.4f}; Holm p={rmsle_width['p_holm_within_readout_metric']:.3g}) and worse than the availability-aware encoder by {rmsle_avail['candidate_minus_ref']:+.4f} (95% CI {rmsle_avail['ci95_low']:+.4f} to {rmsle_avail['ci95_high']:+.4f}; Holm p={rmsle_avail['p_holm_within_readout_metric']:.3g}). FusionVI beat only the smaller parameter-matched width-70 control by {abs(rmsle_pmatch['candidate_minus_ref']):.4f} RMSLE (95% CI for FusionVI-minus-control {rmsle_pmatch['ci95_low']:+.4f} to {rmsle_pmatch['ci95_high']:+.4f}; Holm p={rmsle_pmatch['p_holm_within_readout_metric']:.3g}).

The explicit availability indicator did not improve the same-width joint encoder: difference {availability['candidate_minus_ref']:+.4f} RMSLE (95% CI {availability['ci95_low']:+.4f} to {availability['ci95_high']:+.4f}; Holm p={availability['p_holm_within_readout_metric']:.3f}).

FusionVI also did not recover stronger biological ranking. Its overall Spearman difference versus published totalVI was {spearman_tv['candidate_minus_ref']:+.4f} (95% CI {spearman_tv['ci95_low']:+.4f} to {spearman_tv['ci95_high']:+.4f}; Holm p={spearman_tv['p_holm_within_readout_metric']:.3f}). Its within-cell-type Spearman was lower by {abs(within_tv['candidate_minus_ref']):.4f} versus published totalVI (95% CI {within_tv['ci95_low']:+.4f} to {within_tv['ci95_high']:+.4f}; Holm p={within_tv['p_holm_within_readout_metric']:.2g}) and lower by {abs(within_width['candidate_minus_ref']):.4f} versus the same-width control (Holm p={within_width['p_holm_within_readout_metric']:.2g}).

The D1 affine head did not improve the neural models after the learned efficiency was restored: totalVI changed from {value('totalvi', 'rmsle'):.4f} to {value('totalvi', 'rmsle', 'calibrated'):.4f}, and FusionVI changed from {value('fusionvi', 'rmsle'):.4f} to {value('fusionvi', 'rmsle', 'calibrated'):.4f}. The historical helper output gave {value('totalvi', 'rmsle', 'helper_mean'):.4f} for totalVI and {value('fusionvi', 'rmsle', 'helper_mean'):.4f} for FusionVI because it omitted the learned efficiency.

Likelihood-consistent totalVI therefore reproduces the paper's qualitative ordering and remains the best of the tested neural encoders on aggregate RMSLE. Experiment 4 is a negative result for FusionVI. The useful technical finding is that evaluating current scvi-tools protein imputations against observed counts requires the likelihood-consistent `py_norm_` readout.

![Likelihood-consistent complete-panel evaluation](results/figures/calibration_benchmark_likelihood_correct.png)

"""

    report = REPORT.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"## Experiment 4 paper-aligned complete missing panel\n.*?(?=## Experiment 5 original totalVI partial panel transfer)",
        section,
        report,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"Expected one Experiment 4 section; replaced {count}")

    updated, finding_count = re.subn(
        r"1\. On the complete missing-panel benchmark, .*?(?=\n2\.)",
        "1. On the complete missing-panel benchmark, FusionVI does not improve correctly read totalVI. The initial apparent advantage came from the efficiency-omitting helper output; likelihood-consistent totalVI has lower RMSLE and stronger within-cell-type ranking.",
        updated,
        flags=re.S,
    )
    if finding_count != 1:
        raise RuntimeError(f"Expected one combined Experiment 4 finding; replaced {finding_count}")
    updated = updated.replace(
        "Calibration was fitted only on source cells routed through the corresponding missing-target pattern.",
        "Calibration was fitted only on source cells routed through the corresponding missing-target pattern. Experiment 5 is unaffected by the Experiment 4 readout correction: positive per-protein efficiency scaling cannot change its per-protein Spearman metrics, and its D1 affine calibration absorbs the constant scale.",
    )
    updated = updated.replace(
        "`results/calibration_summary.csv`, `calibration_contrasts.csv`, `calibration_within_celltype.csv` and `calibration_marker_auroc.csv` contain the scale-matched evaluation.",
        "`results/calibration_summary_likelihood_correct.csv`, `calibration_contrasts_likelihood_correct.csv`, `calibration_within_celltype_likelihood_correct.csv` and `calibration_marker_auroc_likelihood_correct.csv` contain the likelihood-consistent Experiment 4 evaluation.",
    )
    REPORT.write_text(updated, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
